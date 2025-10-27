/******************************************************************************
 * QKV Fusion Kernel - Optimized Implementation (Phase 2)
 * Based on FasterTransformer's approach:
 * 1. Single GEMM with concatenated QKV weights
 * 2. Fused split + bias + transpose kernel
 ******************************************************************************/

 #include <cstdio>
 #include <cstdint>
 #include <vector>
 #include <algorithm>
 #include <ATen/cuda/CUDAContext.h>
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
  * Debug helpers for cuBLASLt setup
  ******************************************************************************/
 
 #ifndef LT_CHECK
 #define LT_CHECK(stmt)                                                                          \
     do {                                                                                        \
         cublasStatus_t _lt_status = (stmt);                                                     \
         if (_lt_status != CUBLAS_STATUS_SUCCESS) {                                              \
             printf("[LtErr] %s:%d status=%d\n", __FILE__, __LINE__, static_cast<int>(_lt_status));\
             return;                                                                             \
         }                                                                                       \
     } while (0)
 #endif
 
 #ifndef CUDA_CHECK
 #define CUDA_CHECK(stmt)                                                                         \
     do {                                                                                        \
         cudaError_t _cuda_status = (stmt);                                                       \
         if (_cuda_status != cudaSuccess) {                                                       \
             printf("[CudaErr] %s:%d status=%d (%s)\n", __FILE__, __LINE__,                      \
                    static_cast<int>(_cuda_status), cudaGetErrorString(_cuda_status));           \
             return;                                                                             \
         }                                                                                       \
     } while (0)
 #endif
 
static inline void dump_layout(const char* name, cublasLtMatrixLayout_t desc) {
    uint32_t order = 0;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t ld = 0;
    size_t attr_size = 0;
    cublasLtMatrixLayoutGetAttribute(desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order), &attr_size);
    cublasLtMatrixLayoutGetAttribute(desc, CUBLASLT_MATRIX_LAYOUT_ROWS, &rows, sizeof(rows), &attr_size);
    cublasLtMatrixLayoutGetAttribute(desc, CUBLASLT_MATRIX_LAYOUT_COLS, &cols, sizeof(cols), &attr_size);
    cublasLtMatrixLayoutGetAttribute(desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld), &attr_size);
    const char* order_str = (order == CUBLASLT_ORDER_ROW) ? "ROW" : (order == CUBLASLT_ORDER_COL) ? "COL" : "UNK";
    printf("[LAYOUT] %s order=%s rows=%lld cols=%lld ld=%lld\n",
           name,
           order_str,
           static_cast<long long>(rows),
           static_cast<long long>(cols),
           static_cast<long long>(ld));
}

static inline void dump_call_info(const void* A, const void* B, const void* D,
                                  const void* bias, int M, int K, int N) {
    printf("[LtCall] A=%p B=%p D=%p bias=%p M=%d K=%d N=%d\n",
           A, B, D, bias, M, K, N);
}
 
 static inline void memset_fp16(void* ptr, size_t elems, uint8_t pattern = 0x00) {
     CUDA_CHECK(cudaMemset(ptr, pattern, elems * sizeof(uint16_t)));
 }
 
__global__ void add_bias_rowmajor_kernel(half* __restrict__ data,
                                         const half* __restrict__ bias,
                                         int M,
                                         int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    for (int linear = idx; linear < total; linear += blockDim.x * gridDim.x) {
        const int col = linear % N;
        data[linear] = __hadd(data[linear], bias[col]);
    }
}

// Pack B from row-major to column-major
// B_row: [K, N] row-major, idx = k*N + n
// B_col: [K, N] col-major, idx = n*K + k
__global__ void pack_row_to_col_kernel(const half* __restrict__ B_row,
                                       half* __restrict__ B_col,
                                       int K, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (k < K && n < N) {
        B_col[n * K + k] = B_row[k * N + n];
    }
}

// Host wrapper: pack B_row to B_col and verify numerically
static bool pack_and_verify_col_major(const half* B_row_dev, half* B_col_dev,
                                       int K, int N, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (K + block.y - 1) / block.y);
    pack_row_to_col_kernel<<<grid, block, 0, stream>>>(B_row_dev, B_col_dev, K, N);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[Pack] Kernel launch failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    cudaStreamSynchronize(stream);
    
    // Numerical verification on small sample
    const int sample_k = std::min(K, 8);
    const int sample_n = std::min(N, 8);
    std::vector<half> h_row(sample_k * sample_n);
    std::vector<half> h_col(sample_k * sample_n);
    
    for (int k = 0; k < sample_k; ++k) {
        cudaMemcpyAsync(&h_row[k * sample_n], B_row_dev + k * N, 
                        sample_n * sizeof(half), cudaMemcpyDeviceToHost, stream);
    }
    for (int n = 0; n < sample_n; ++n) {
        cudaMemcpyAsync(&h_col[n * sample_k], B_col_dev + n * K,
                        sample_k * sizeof(half), cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
    
    float max_diff = 0.0f;
    for (int k = 0; k < sample_k; ++k) {
        for (int n = 0; n < sample_n; ++n) {
            float v_row = __half2float(h_row[k * sample_n + n]);
            float v_col = __half2float(h_col[n * sample_k + k]);
            float diff = fabsf(v_row - v_col);
            max_diff = std::max(max_diff, diff);
        }
    }
    
    printf("[Pack] Numerical check: max_abs_diff = %.6e (sample %dx%d)\n", max_diff, sample_k, sample_n);
    if (max_diff > 1e-4f) {  // FP16 tolerance
        printf("[Pack] FAIL: pack verification exceeded tolerance\n");
        return false;
    }
    return true;
}
 
 static bool run_cublas_gemm_ex_with_bias(
     const half* hidden_states,
     const half* qkv_weight,
     const half* qkv_bias,
     half* qkv_buf,
     int M,
     int N,
     int K,
     cudaStream_t stream
 ) {
     cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
     cublasStatus_t status = cublasSetStream(handle, stream);
     if (status != CUBLAS_STATUS_SUCCESS) {
         printf("[cuBLAS] cublasSetStream failed: %d\n", static_cast<int>(status));
         return false;
     }
 
     const float alpha = 1.0f;
     const float beta = 0.0f;
 
     status = cublasGemmEx(
         handle,
         CUBLAS_OP_N,               // B^T (handled by layout trick)
         CUBLAS_OP_N,               // A^T
         N,                         // m
         M,                         // n
         K,                         // k
         &alpha,
         qkv_weight,
         CUDA_R_16F,
         N,                         // lda
         hidden_states,
         CUDA_R_16F,
         K,                         // ldb
         &beta,
         qkv_buf,
         CUDA_R_16F,
         N,                         // ldc
         CUBLAS_COMPUTE_32F,
         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
 
     if (status != CUBLAS_STATUS_SUCCESS) {
         printf("[cuBLAS] cublasGemmEx failed: %d\n", static_cast<int>(status));
         return false;
     }
 
     if (qkv_bias != nullptr) {
         const int threads = 256;
         const int blocks = std::min(
             static_cast<int>((static_cast<long long>(M) * N + threads - 1) / threads),
             65535);
         add_bias_rowmajor_kernel<<<blocks, threads, 0, stream>>>(
             qkv_buf, qkv_bias, M, N);
     }
 
     return true;
 }
  
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

    const size_t totalElements = static_cast<size_t>(M) * static_cast<size_t>(N);
    // Initialize D to a non-zero pattern to catch silent zero outputs when beta == 0
    memset_fp16(qkv_buf, totalElements, 0x11);

    // Allocate and pack B to true column-major layout
    half* qkv_weight_col = nullptr;
    cudaError_t alloc_err = cudaMalloc(&qkv_weight_col, static_cast<size_t>(K) * N * sizeof(half));
    if (alloc_err != cudaSuccess) {
        printf("[cuBLASLt] Failed to allocate B_col buffer: %s\n", cudaGetErrorString(alloc_err));
        return;
    }

    printf("[Pack] Packing B from row-major to column-major...\n");
    bool pack_ok = pack_and_verify_col_major(qkv_weight, qkv_weight_col, K, N, stream);
    if (!pack_ok) {
        printf("[cuBLASLt] Weight packing failed; aborting\n");
        cudaFree(qkv_weight_col);
        return;
    }

    cublasLtMatrixLayout_t aLayout = nullptr;
    cublasLtMatrixLayout_t bLayoutRow = nullptr; // B row-major (ld = N) for fallback
    cublasLtMatrixLayout_t bLayoutCol = nullptr; // B col-major (ld = K) for bias epilogue
    cublasLtMatrixLayout_t cLayout = nullptr;
    cublasLtMatrixLayout_t dLayout = nullptr;
    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
    cublasLtOrder_t colOrder = CUBLASLT_ORDER_COL;
 
    bool layouts_ok = (cublasLtMatrixLayoutCreate(&aLayout, CUDA_R_16F, M, K, K) ==
                       CUBLAS_STATUS_SUCCESS) &&
                      // B row-major (fallback path): ld = N
                      (cublasLtMatrixLayoutCreate(&bLayoutRow, CUDA_R_16F, K, N, N) ==
                       CUBLAS_STATUS_SUCCESS) &&
                      // B column-major (bias epilogue path): ld = K
                      (cublasLtMatrixLayoutCreate(&bLayoutCol, CUDA_R_16F, K, N, K) ==
                       CUBLAS_STATUS_SUCCESS) &&
                      (cublasLtMatrixLayoutCreate(&cLayout, CUDA_R_16F, M, N, N) ==
                       CUBLAS_STATUS_SUCCESS) &&
                      (cublasLtMatrixLayoutCreate(&dLayout, CUDA_R_16F, M, N, N) ==
                       CUBLAS_STATUS_SUCCESS);

    if (!layouts_ok) {
        printf("[cuBLASLt] Failed to create matrix layouts\n");
    } else {
        cublasLtMatrixLayoutSetAttribute(
            aLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
        // Set orders for both B layouts
        cublasLtMatrixLayoutSetAttribute(
            bLayoutRow, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
        cublasLtMatrixLayoutSetAttribute(
            bLayoutCol, CUBLASLT_MATRIX_LAYOUT_ORDER, &colOrder, sizeof(colOrder));
        cublasLtMatrixLayoutSetAttribute(
            cLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
        cublasLtMatrixLayoutSetAttribute(
            dLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));

        dump_layout("A", aLayout);
        dump_layout("B_row", bLayoutRow);
        dump_layout("B_col", bLayoutCol);
        dump_layout("C", cLayout);
        dump_layout("D", dLayout);
    }
 
    auto try_matmul = [&](const half* B_ptr, cublasLtMatrixLayout_t bLayoutTry, cublasOperation_t transB,
                          const char* mode_desc, bool fuse_bias, bool& bias_fused) -> bool {
        bias_fused = false;
 
         cublasLtMatmulDesc_t matmulDesc = nullptr;
         cublasStatus_t status =
             cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
         if (status != CUBLAS_STATUS_SUCCESS) {
             printf("[cuBLASLt] MatmulDescCreate failed: %d\n", static_cast<int>(status));
             return false;
         }
 
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        // Set B transpose according to the mode
        cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));
 
        if (fuse_bias && qkv_bias != nullptr) {
             cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
             status = cublasLtMatmulDescSetAttribute(
                 matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
             if (status != CUBLAS_STATUS_SUCCESS) {
                 printf("[cuBLASLt] Failed to set epilogue attribute: %d\n",
                        static_cast<int>(status));
                 cublasLtMatmulDescDestroy(matmulDesc);
                 return false;
             }
             const void* bias_ptr = qkv_bias;
             status = cublasLtMatmulDescSetAttribute(
                 matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr));
             if (status != CUBLAS_STATUS_SUCCESS) {
                 printf("[cuBLASLt] Failed to set bias pointer attribute: %d\n",
                        static_cast<int>(status));
                 cublasLtMatmulDescDestroy(matmulDesc);
                 return false;
             }
             bias_fused = true;
         }
 
         cublasLtMatmulPreference_t preference = nullptr;
         status = cublasLtMatmulPreferenceCreate(&preference);
         if (status != CUBLAS_STATUS_SUCCESS) {
             printf("[cuBLASLt] PreferenceCreate failed: %d\n", static_cast<int>(status));
             cublasLtMatmulDescDestroy(matmulDesc);
             return false;
         }
 
         const size_t workspaceCaps[] = {
             static_cast<size_t>(64) * 1024 * 1024,
             static_cast<size_t>(16) * 1024 * 1024,
             0
         };
 
         constexpr int maxAlgos = 32;
         std::vector<cublasLtMatmulHeuristicResult_t> heuristicResults(maxAlgos);
 
         bool launchSuccess = false;
         cublasStatus_t matmulStatus = CUBLAS_STATUS_SUCCESS;
 
        for (size_t workspaceCap : workspaceCaps) {
             status = cublasLtMatmulPreferenceSetAttribute(
                 preference,
                 CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                 &workspaceCap,
                 sizeof(workspaceCap));
             if (status != CUBLAS_STATUS_SUCCESS) {
                 printf("[cuBLASLt] PreferenceSetAttribute failed: %d (cap %zu)\n",
                        static_cast<int>(status), workspaceCap);
                 continue;
             }
 
             int returnedResults = 0;
            status = cublasLtMatmulAlgoGetHeuristic(
                 cublaslt_handle,
                 matmulDesc,
                 aLayout,
                bLayoutTry,
                 cLayout,
                 dLayout,
                 preference,
                 maxAlgos,
                 heuristicResults.data(),
                 &returnedResults);
 
            printf("[LtHeur] returned=%d ws_cap=%zu bytes (bias_try=%d)\n",
                   returnedResults, workspaceCap, static_cast<int>(fuse_bias));

            if (status == CUBLAS_STATUS_NOT_SUPPORTED || returnedResults == 0) {
                printf("[cuBLASLt] No algorithms for workspace cap %zu bytes\n", workspaceCap);
                 continue;
             }
             if (status != CUBLAS_STATUS_SUCCESS) {
                 printf("[cuBLASLt] Heuristic query failed: %d (cap %zu)\n",
                        static_cast<int>(status), workspaceCap);
                 continue;
             }
 
             for (int algoIdx = 0; algoIdx < returnedResults; ++algoIdx) {
                 size_t workspaceSize = heuristicResults[algoIdx].workspaceSize;
                 void* workspace = nullptr;
                 if (workspaceSize > 0) {
                     cudaError_t allocStatus = cudaMalloc(&workspace, workspaceSize);
                     if (allocStatus != cudaSuccess) {
                         printf("[cuBLASLt] Workspace alloc failed (%zu bytes) status=%d\n",
                                workspaceSize, static_cast<int>(allocStatus));
                         workspace = nullptr;
                         workspaceSize = 0;
                     }
                 }
 
                printf("[cuBLASLt] Trying algo %d (mode=%s bias=%d) workspace=%zu bytes\n",
                       algoIdx, mode_desc, static_cast<int>(bias_fused), workspaceSize);
 
                matmulStatus = cublasLtMatmul(
                    cublaslt_handle,
                    matmulDesc,
                    &alpha,
                    hidden_states,
                    aLayout,
                    B_ptr,
                    bLayoutTry,
                    &beta,
                    qkv_buf,
                    cLayout,
                    qkv_buf,
                    dLayout,
                    &heuristicResults[algoIdx].algo,
                    workspace,
                    workspaceSize,
                    stream);
 
                 if (workspace != nullptr) {
                     cudaFree(workspace);
                 }
 
                if (matmulStatus == CUBLAS_STATUS_SUCCESS) {
                     launchSuccess = true;
                    printf("[cuBLASLt] Selected algo %d (mode=%s bias=%d)\n",
                           algoIdx, mode_desc, static_cast<int>(bias_fused));
                     break;
                 } else {
                    printf("[cuBLASLt] Algo %d failed with status %d (mode=%s bias=%d)\n",
                           algoIdx, static_cast<int>(matmulStatus), mode_desc, static_cast<int>(bias_fused));
                 }
             }
 
             if (launchSuccess) {
                 break;
             }
         }
 
         cublasLtMatmulPreferenceDestroy(preference);
         if (!launchSuccess) {
             bias_fused = false;
         }
 
         cublasLtMatmulDescDestroy(matmulDesc);
         return launchSuccess;
     };
 
    bool bias_fused = false;
    bool success = false;
    if (layouts_ok) {
        dump_call_info(hidden_states, qkv_weight, qkv_buf, qkv_bias, M, K, N);
        
        // 1) Try B column-major with opB = N (best shot at bias epilogue)
        printf("[cuBLASLt] Trying configuration: B=column-major (packed), opB=N\n");
        success = try_matmul(qkv_weight_col, bLayoutCol, CUBLAS_OP_N, "Bcol_N", qkv_bias != nullptr, bias_fused);

        if (!success && qkv_bias != nullptr) {
            printf("[cuBLASLt] Retrying (B=col) without bias epilogue\n");
            success = try_matmul(qkv_weight_col, bLayoutCol, CUBLAS_OP_N, "Bcol_N", false, bias_fused);
        }

        // 2) Fallback to B row-major with opB = T
        if (!success) {
            printf("[cuBLASLt] Switching to configuration: B=row-major, opB=T\n");
            success = try_matmul(qkv_weight, bLayoutRow, CUBLAS_OP_T, "Brow_T", qkv_bias != nullptr, bias_fused);

            if (!success && qkv_bias != nullptr) {
                printf("[cuBLASLt] Retrying (B=row) without bias epilogue\n");
                success = try_matmul(qkv_weight, bLayoutRow, CUBLAS_OP_T, "Brow_T", false, bias_fused);
            }
        }

        if (success && qkv_bias != nullptr && !bias_fused) {
            printf("[cuBLASLt] Bias epilogue not available; launching fallback bias kernel\n");
            const int threads = 256;
            const int blocks = std::min(
                static_cast<int>((static_cast<long long>(M) * N + threads - 1) / threads),
                65535);
            add_bias_rowmajor_kernel<<<blocks, threads, 0, stream>>>(
                qkv_buf, qkv_bias, M, N);
        }
    }
 
     if (!success) {
         printf("[cuBLASLt] Falling back to cublasGemmEx path\n");
         bool cublas_ok = run_cublas_gemm_ex_with_bias(
             hidden_states,
             qkv_weight,
             qkv_bias,
             qkv_buf,
             M,
             N,
             K,
             stream);
         if (!cublas_ok) {
             printf("cuBLAS fallback also failed; output remains uninitialized\n");
         }
     }
 
    if (aLayout) cublasLtMatrixLayoutDestroy(aLayout);
    if (bLayoutRow) cublasLtMatrixLayoutDestroy(bLayoutRow);
    if (bLayoutCol) cublasLtMatrixLayoutDestroy(bLayoutCol);
    if (cLayout) cublasLtMatrixLayoutDestroy(cLayout);
    if (dLayout) cublasLtMatrixLayoutDestroy(dLayout);
    
    // Cleanup packed weight buffer
    if (qkv_weight_col) {
        cudaFree(qkv_weight_col);
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
  
  
 