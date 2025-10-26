/******************************************************************************
 * QKV Fusion Kernel - Optimized Implementation (Phase 2)
 * Based on FasterTransformer's approach:
 * 1. Single GEMM with concatenated QKV weights
 * 2. Fused split + bias + transpose kernel
 ******************************************************************************/

 #include <cstdio>
 #include <cuda_runtime.h>
 #include <cuda_fp16.h>
 #include <cublas_v2.h>
 
 // CUTLASS includes for optimized GEMM
 // #include <cute/tensor.hpp>
 // #include <cutlass/cutlass.h>
 // #include <cutlass/numeric_types.h>
 // #include <cutlass/gemm/device/gemm.h>
 
 #include "../qkv_fused_params.h"
 
 namespace qkv_fusion {
 
 // using namespace cute;
 
 /******************************************************************************
  * Kernel 1: Fused QKV GEMM using CUTLASS
  * Input:  hidden_states [batch * seq_len, hidden_dim]
  * Weight: qkv_weight [hidden_dim, (num_q_heads + 2*num_kv_heads) * head_dim]
  * Output: qkv_buf [batch * seq_len, (num_q_heads + 2*num_kv_heads) * head_dim]
  * 
  * Matrix multiplication: qkv_buf = hidden_states @ qkv_weight
  * Dimensions: [M, K] @ [K, N] = [M, N]
  *   where M = batch_size * seq_len
  *         K = hidden_dim
  *         N = qkv_output_dim
  ******************************************************************************/
 
 #include <cublas_v2.h>
 #include <cuda_fp16.h>
 
 // CUTLASS GEMM implementation using tensor cores
 void launch_fused_qkv_gemm_cutlass(
     const half* hidden_states,      // [M, K]
     const half* qkv_weight,         // [K, N]
     half* qkv_buf,                  // [M, N]
     int M,                          // batch_size * seq_len
     int N,                          // qkv_output_dim
     int K,                          // hidden_dim
     cublasHandle_t cublas_handle,
     cudaStream_t stream
 ) {
     // Use cuBLAS for now (CUTLASS integration can be done later for more control)
     // cuBLAS is highly optimized and uses tensor cores automatically
     
     const half alpha = __float2half(1.0f);
     const half beta = __float2half(0.0f);
     
     // Set stream for cuBLAS
     cublasSetStream(cublas_handle, stream);
     
     // Matrix multiplication: C = alpha * A * B + beta * C
     // A = hidden_states [M, K]
     // B = qkv_weight [K, N]
     // C = qkv_buf [M, N]
     //
     // cuBLAS uses column-major, so we need to transpose:
     // C^T = alpha * B^T * A^T + beta * C^T
     // which is: [N, M] = [N, K] * [K, M]
     
     cublasStatus_t status = cublasGemmEx(
         cublas_handle,
         CUBLAS_OP_N,        // B^T is not transposed (B is already in column-major)
         CUBLAS_OP_N,        // A^T is not transposed (A is already in column-major)
         N,                  // rows of B^T (columns of B)
         M,                  // columns of A^T (rows of A)
         K,                  // columns of B^T (rows of B) = rows of A^T (columns of A)
         &alpha,
         qkv_weight,         // B in column-major
         CUDA_R_16F,         // data type
         N,                  // leading dimension of B
         hidden_states,      // A in column-major
         CUDA_R_16F,         // data type
         K,                  // leading dimension of A
         &beta,
         qkv_buf,            // C in column-major
         CUDA_R_16F,         // data type
         N,                  // leading dimension of C
         CUBLAS_COMPUTE_16F, // compute type (uses tensor cores on Ampere+)
         CUBLAS_GEMM_DEFAULT_TENSOR_OP  // algorithm (tensor core enabled)
     );
     
     if (status != CUBLAS_STATUS_SUCCESS) {
         printf("cuBLAS GEMM failed with status: %d\n", status);
     }
 }
 
/******************************************************************************
 * Kernel 2: Split QKV + Add Bias + Transpose (Optimized)
 * 
 * Optimization: Process in chunks to improve memory coalescing and reduce branching
 * 
 * Input:  qkv_buf [batch * seq_len, (num_q_heads + 2*num_kv_heads) * head_dim]
 * Output: q_out [batch, num_q_heads, seq_len, head_dim]
 *         k_out [batch, num_kv_heads, seq_len, head_dim]
 *         v_out [batch, num_kv_heads, seq_len, head_dim]
 ******************************************************************************/

template<typename T>
__global__ void split_qkv_bias_transpose_kernel_optimized(
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    T* __restrict__ v_out,
    const T* __restrict__ qkv_buf,
    const T* __restrict__ qkv_bias,
    const int batch_size,
    const int seq_len,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim
) {
    const int token_num = batch_size * seq_len;
    const int q_size = num_q_heads * head_dim;
    const int kv_size = num_kv_heads * head_dim;
    
    // Grid-stride loop over tokens
    for (int token_idx = blockIdx.x; token_idx < token_num; token_idx += gridDim.x) {
        const int batch_id = token_idx / seq_len;
        const int seq_id = token_idx % seq_len;
        
        const T* qkv_row = qkv_buf + token_idx * (q_size + 2 * kv_size);
        
        // Process Q (coalesced access within warp)
        for (int i = threadIdx.x; i < q_size; i += blockDim.x) {
            const int head_id = i / head_dim;
            const int dim_id = i % head_dim;
            
            T val = qkv_row[i];
            if (qkv_bias != nullptr) {
                val = __hadd(val, qkv_bias[i]);
            }
            
            const int out_idx = batch_id * num_q_heads * seq_len * head_dim +
                               head_id * seq_len * head_dim +
                               seq_id * head_dim +
                               dim_id;
            q_out[out_idx] = val;
        }
        
        // Process K
        for (int i = threadIdx.x; i < kv_size; i += blockDim.x) {
            const int head_id = i / head_dim;
            const int dim_id = i % head_dim;
            
            T val = qkv_row[q_size + i];
            if (qkv_bias != nullptr) {
                val = __hadd(val, qkv_bias[q_size + i]);
            }
            
            const int out_idx = batch_id * num_kv_heads * seq_len * head_dim +
                               head_id * seq_len * head_dim +
                               seq_id * head_dim +
                               dim_id;
            k_out[out_idx] = val;
        }
        
        // Process V
        for (int i = threadIdx.x; i < kv_size; i += blockDim.x) {
            const int head_id = i / head_dim;
            const int dim_id = i % head_dim;
            
            T val = qkv_row[q_size + kv_size + i];
            if (qkv_bias != nullptr) {
                val = __hadd(val, qkv_bias[q_size + kv_size + i]);
            }
            
            const int out_idx = batch_id * num_kv_heads * seq_len * head_dim +
                               head_id * seq_len * head_dim +
                               seq_id * head_dim +
                               dim_id;
            v_out[out_idx] = val;
        }
    }
}
 
 /******************************************************************************
  * Host function: Orchestrates the two-kernel approach
  ******************************************************************************/
 
 void run_qkv_fusion_optimized(
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
     
     // Total output dimension: Q + K + V
     const int qkv_output_dim = (num_q_heads + 2 * num_kv_heads) * head_dim;
    const int token_num = batch_size * seq_len;
    const int M = token_num;
    const int N = qkv_output_dim;
    const int K = hidden_dim;
    
    // Use pre-allocated workspace (no cudaMalloc overhead!)
    half* qkv_buf = reinterpret_cast<half*>(params.workspace_ptr);

    // Step 1: Single GEMM for all Q, K, V projections
     // hidden_states [M, K] @ qkv_weight [K, N] = qkv_buf [M, N]
     // where M = batch_size * seq_len
     //       K = hidden_dim
     //       N = (num_q_heads + 2*num_kv_heads) * head_dim
     //
     // For Qwen3: M = batch*seq, K = 3584, N = 5120
     
     launch_fused_qkv_gemm_cutlass(
         reinterpret_cast<const half*>(params.hidden_states_ptr),
         reinterpret_cast<const half*>(params.qkv_fused_weight_ptr),
         qkv_buf,
         M, N, K,
         cublas_handle,
         stream
     );
     
    // Step 2: Split QKV + Add Bias + Transpose (Optimized)
    // Use optimized kernel with better memory coalescing
    const int threads = 256;
    const int blocks = min(token_num, 512);  // Limit blocks for better occupancy
    
    split_qkv_bias_transpose_kernel_optimized<half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<half*>(params.q_out_ptr),
        reinterpret_cast<half*>(params.k_out_ptr),
        reinterpret_cast<half*>(params.v_out_ptr),
        qkv_buf,
        params.has_bias ? reinterpret_cast<const half*>(params.qkv_fused_bias_ptr) : nullptr,
        batch_size,
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim
    );

    // No cudaFree needed - workspace is managed by PyTorch!
    
    // Check for errors
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
     }
 }
 
 /******************************************************************************
  * Alternative: Fully Fused Single-Kernel Approach (Advanced)
  * Combines GEMM + Split + Transpose in one kernel
  * Requires more complex implementation but eliminates intermediate buffer
  ******************************************************************************/
 
 template<int kBlockM, int kBlockN, int kHeadDim>
 __global__ void qkv_fusion_fully_fused_kernel(
     const half* __restrict__ hidden_states,
     const half* __restrict__ qkv_weight,
     const half* __restrict__ qkv_bias,
     half* __restrict__ q_out,
     half* __restrict__ k_out,
     half* __restrict__ v_out,
     const int batch_size,
     const int seq_len,
     const int hidden_dim,
     const int num_q_heads,
     const int num_kv_heads
 ) {
     // This is an advanced optimization that fuses everything into one kernel
     // Benefits:
     // - No intermediate buffer needed
     // - Better data locality
     // - Fewer kernel launches
     //
     // Implementation strategy:
     // 1. Each thread block loads a tile of hidden_states into shared memory
     // 2. Computes Q, K, V for that tile using shared memory
     // 3. Directly writes to final transposed output layout
     //
     // TODO: Implement in Phase 2.5 after basic optimization works
 }
 
 } // namespace qkv_fusion
 
 