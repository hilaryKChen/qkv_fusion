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
 #include <cublasLt.h>
 
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

// Optimized GEMM using cuBLASLt with bias epilogue fusion
void launch_fused_qkv_gemm_cutlass(
    const half* hidden_states,      // [M, K] row-major
    const half* qkv_weight,         // [K, N] row-major
    const half* qkv_bias,           // [N] or nullptr
    half* qkv_buf,                  // [M, N] row-major
    int M,                          // batch_size * seq_len
    int N,                          // qkv_output_dim
    int K,                          // hidden_dim
    cublasLtHandle_t cublaslt_handle,
    cudaStream_t stream
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Describe the GEMM in row-major form so we can pass PyTorch-contiguous tensors directly.
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_16F);

    cublasOperation_t opA = CUBLAS_OP_N;  // hidden_states[M, K]
    cublasOperation_t opB = CUBLAS_OP_N;  // qkv_weight[K, N]
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    if (qkv_bias != nullptr) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
        cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &qkv_bias, sizeof(void*));
    }

    cublasLtMatrixLayout_t aLayout, bLayout, cLayout;

    // Matrix A (hidden_states) : [M, K] row-major
    cublasLtMatrixLayoutCreate(&aLayout, CUDA_R_16F, M, K, K);
    // Matrix B (fused weight)  : [K, N] row-major
    cublasLtMatrixLayoutCreate(&bLayout, CUDA_R_16F, K, N, N);
    // Matrix C/D (output)      : [M, N] row-major
    cublasLtMatrixLayoutCreate(&cLayout, CUDA_R_16F, M, N, N);

    // Tell cuBLASLt that the layouts are row-major.
    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
    cublasLtMatrixLayoutSetAttribute(
        aLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
    cublasLtMatrixLayoutSetAttribute(
        bLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
    cublasLtMatrixLayoutSetAttribute(
        cLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));

    // Select an algorithm
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);

    size_t maxWorkspaceSize = 8 * 1024 * 1024;  // cap at 8MB for now
    cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &maxWorkspaceSize,
        sizeof(maxWorkspaceSize));

    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(
        cublaslt_handle,
        matmulDesc,
        aLayout,
        bLayout,
        cLayout,
        cLayout,
        preference,
        1,
        &heuristicResult,
        &returnedResults);

    if (returnedResults == 0) {
        printf("cuBLASLt heuristic failed, no algorithm found\n");
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(aLayout);
        cublasLtMatrixLayoutDestroy(bLayout);
        cublasLtMatrixLayoutDestroy(cLayout);
        cublasLtMatmulDescDestroy(matmulDesc);
        return;
    }

    size_t workspaceSize = heuristicResult.workspaceSize;
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        cudaMalloc(&workspace, workspaceSize);
    }

    // Execute the matmul directly on row-major tensors.
    cublasStatus_t status = cublasLtMatmul(
        cublaslt_handle,
        matmulDesc,
        &alpha,
        hidden_states,
        aLayout,
        qkv_weight,
        bLayout,
        &beta,
        qkv_buf,
        cLayout,
        qkv_buf,
        cLayout,
        &heuristicResult.algo,
        workspace,
        workspaceSize,
        stream);

    if (workspace != nullptr) {
        cudaFree(workspace);
    }
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(aLayout);
    cublasLtMatrixLayoutDestroy(bLayout);
    cublasLtMatrixLayoutDestroy(cLayout);
    cublasLtMatmulDescDestroy(matmulDesc);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLASLt matmul failed with status: %d\n", status);
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
    cublasLtHandle_t cublaslt_handle,
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

   // Step 1: Single cuBLASLt GEMM with bias epilogue fusion
    // hidden_states [M, K] @ qkv_weight [K, N] = qkv_buf [M, N]
    // Bias is fused in the GEMM epilogue (no separate kernel needed!)
    //
    // For Qwen3: M = batch*seq, K = 2048, N = 5120
    
    launch_fused_qkv_gemm_cutlass(
        reinterpret_cast<const half*>(params.hidden_states_ptr),
        reinterpret_cast<const half*>(params.qkv_fused_weight_ptr),
        params.has_bias ? reinterpret_cast<const half*>(params.qkv_fused_bias_ptr) : nullptr,
        qkv_buf,
        M, N, K,
        cublaslt_handle,
        stream
    );
    
    // Step 2: NO SPLIT KERNEL NEEDED!
    // Bias is already added by cuBLASLt epilogue
    // Python will do the reshape/slice (zero-copy operations)
    //
    // Output qkv_buf is now [batch*seq, 5120] with bias already applied
    // Python will:
    //   1. view as [batch, seq, 5120]
    //   2. slice into Q[batch, seq, 4096], K[batch, seq, 512], V[batch, seq, 512]
    //   3. view as [batch, seq, heads, head_dim]
    //   4. transpose to [batch, heads, seq, head_dim]
    //
    // All of these are zero-copy view operations except the final transpose!
    
    /* SPLIT KERNEL REMOVED - Now handled in Python
    const int threads = 256;
    const int blocks = min(token_num, 512);
    
    split_qkv_bias_transpose_kernel_optimized<half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<half*>(params.q_out_ptr),
        reinterpret_cast<half*>(params.k_out_ptr),
        reinterpret_cast<half*>(params.v_out_ptr),
        qkv_buf,
        nullptr,  // Bias already added by cuBLASLt
        batch_size,
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim
    );
    */

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
 
 
