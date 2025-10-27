/******************************************************************************
 * QKV Fusion Kernel - Lightweight Implementation (Phase 3)
 * Approach: GEMM + Simple Bias Add + PyTorch Transpose/Split
 * 
 * This approach minimizes custom CUDA code and leverages PyTorch's
 * highly optimized tensor operations for reshape/transpose.
 ******************************************************************************/

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "../qkv_fused_params.h"

namespace qkv_fusion {

/******************************************************************************
 * Simple Bias Add Kernel (Element-wise)
 * 
 * Input:  qkv_buf [batch * seq_len, qkv_dim] (GEMM output)
 * Bias:   qkv_bias [qkv_dim]
 * Output: qkv_buf [batch * seq_len, qkv_dim] (in-place)
 * 
 * This is a very simple, memory bandwidth-bound kernel that should be
 * nearly free compared to GEMM.
 ******************************************************************************/
template<typename T>
__global__ void add_bias_kernel(
    T* __restrict__ data,
    const T* __restrict__ bias,
    const int total_elements,
    const int dim
) {
    // Each thread processes multiple elements for better efficiency
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int idx = tid; idx < total_elements; idx += stride) {
        const int bias_idx = idx % dim;
        data[idx] = __hadd(data[idx], bias[bias_idx]);
    }
}

/******************************************************************************
 * cuBLAS GEMM (reused from optimized version)
 ******************************************************************************/
// Use cublasGemmEx (current approach - no bias fusion)
void launch_fused_qkv_gemm(
    const half* hidden_states,      // [M, K] row-major
    const half* qkv_weight,         // [K, N] row-major
    half* qkv_buf,                  // [M, N] row-major
    int M,                          // batch_size * seq_len
    int N,                          // qkv_output_dim
    int K,                          // hidden_dim
    cublasHandle_t cublas_handle,
    cudaStream_t stream
) {
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    
    cublasSetStream(cublas_handle, stream);
    
    // Row-major C = A @ B becomes column-major C^T = B^T @ A^T
    cublasStatus_t status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_N,        // Don't transpose qkv_weight^T
        CUBLAS_OP_N,        // Don't transpose hidden_states^T
        N,                  // Rows of qkv_weight^T[N, K]
        M,                  // Columns of hidden_states^T[K, M]
        K,                  // Inner dimension
        &alpha,
        qkv_weight,         // First matrix: qkv_weight^T[N, K]
        CUDA_R_16F,
        N,                  // Leading dimension
        hidden_states,      // Second matrix: hidden_states^T[K, M]
        CUDA_R_16F,
        K,                  // Leading dimension
        &beta,
        qkv_buf,            // Output: qkv_buf^T[N, M]
        CUDA_R_16F,
        N,                  // Leading dimension
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS GEMM failed with status: %d\n", status);
    }
}

// TODO: Use cublasLt with epilogue fusion for optimal performance
// This would fuse bias into GEMM and match F.linear's performance
// See: https://docs.nvidia.com/cuda/cublas/#cublasltmatmul

/******************************************************************************
 * Host function: GEMM + Bias (no transpose/split)
 * 
 * This function only does:
 * 1. cuBLAS GEMM: hidden_states @ qkv_weight
 * 2. Simple bias add (if bias exists)
 * 
 * The Python side will handle:
 * 3. Split Q, K, V
 * 4. Reshape and transpose
 ******************************************************************************/
void run_qkv_fusion_lightweight(
    QKVFusedParams &params,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
) {
    const int batch_size = params.batch_size;
    const int seq_len = params.seqlen;
    const int hidden_dim = params.hidden_dim;
    const int num_q_heads = params.num_q_heads;
    const int num_kv_heads = params.num_kv_heads;
    const int head_dim = params.head_dim;
    
    // Calculate dimensions
    const int qkv_output_dim = (num_q_heads + 2 * num_kv_heads) * head_dim;
    const int token_num = batch_size * seq_len;
    const int M = token_num;
    const int N = qkv_output_dim;
    const int K = hidden_dim;
    
    // Use pre-allocated workspace for GEMM output
    half* qkv_buf = reinterpret_cast<half*>(params.workspace_ptr);
    
    // Step 1: cuBLAS GEMM
    // hidden_states [M, K] @ qkv_weight [K, N] = qkv_buf [M, N]
    launch_fused_qkv_gemm(
        reinterpret_cast<const half*>(params.hidden_states_ptr),
        reinterpret_cast<const half*>(params.qkv_fused_weight_ptr),
        qkv_buf,
        M, N, K,
        cublas_handle,
        stream
    );
    
    // Step 2: Add bias (if exists)
    if (params.has_bias) {
        const int total_elements = M * N;
        const int threads = 256;
        const int blocks = (total_elements + threads - 1) / threads;
        const int max_blocks = 1024;  // Limit for better occupancy
        
        add_bias_kernel<half><<<min(blocks, max_blocks), threads, 0, stream>>>(
            qkv_buf,
            reinterpret_cast<const half*>(params.qkv_fused_bias_ptr),
            total_elements,
            N
        );
    }
    
    // Step 3 & 4: Split + Transpose done in PyTorch (Python side)
    // No need for custom kernel here!
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
}

} // namespace qkv_fusion

