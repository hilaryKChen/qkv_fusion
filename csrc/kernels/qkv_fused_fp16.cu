/******************************************************************************
 * QKV Fusion Kernel - FP16 Implementation
 * Phase 1: Simple baseline using CUTLASS GEMM
 ******************************************************************************/

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUTLASS includes
// #include <cute/tensor.hpp>
// #include <cutlass/cutlass.h>
// #include <cutlass/numeric_types.h>
// #include <cutlass/gemm/device/gemm.h>

#include "../qkv_fused_params.h"

namespace qkv_fusion {

// using namespace cute;

// Simple kernel that does Q, K, V projections sequentially
// This is Phase 1 - not yet optimized, but correct
template<int kBlockM, int kBlockN, int kHeadDim>
__global__ void qkv_projection_kernel_simple(
    const QKVFusedParams params
) {
    // Block and thread indices
    const int batch_idx = blockIdx.z;
    const int seq_block_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Check bounds
    if (batch_idx >= params.batch_size) return;
    
    const int seq_start = seq_block_idx * kBlockM;
    if (seq_start >= params.seqlen) return;
    
    // Shared memory for input hidden states
    extern __shared__ half smem[];
    half *smem_hidden = smem;
    half *smem_output = smem + kBlockM * params.hidden_dim;
    
    // Load hidden states for this sequence block into shared memory
    for (int i = tid; i < kBlockM * params.hidden_dim; i += blockDim.x) {
        int seq_offset = i / params.hidden_dim;
        int hidden_offset = i % params.hidden_dim;
        int global_seq_idx = seq_start + seq_offset;
        
        if (global_seq_idx < params.seqlen) {
            half *hidden_ptr = reinterpret_cast<half*>(params.hidden_states_ptr);
            smem_hidden[i] = hidden_ptr[
                batch_idx * params.hidden_batch_stride + 
                global_seq_idx * params.hidden_row_stride + 
                hidden_offset
            ];
        } else {
            smem_hidden[i] = __float2half(0.0f);
        }
    }
    __syncthreads();
    
    // Process Q projection for this head
    if (head_idx < params.num_q_heads) {
        half *q_weight = reinterpret_cast<half*>(params.q_weight_ptr);
        half *q_out = reinterpret_cast<half*>(params.q_out_ptr);
        
        // Simple matrix multiply: hidden_states @ q_weight
        // For each sequence position in this block
        for (int seq_offset = 0; seq_offset < kBlockM; seq_offset++) {
            int global_seq_idx = seq_start + seq_offset;
            if (global_seq_idx >= params.seqlen) break;
            
            // For each dimension in head
            for (int d = tid; d < kHeadDim; d += blockDim.x) {
                float acc = 0.0f;
                
                // Dot product with weight column
                for (int h = 0; h < params.hidden_dim; h++) {
                    float hidden_val = __half2float(smem_hidden[seq_offset * params.hidden_dim + h]);
                    float weight_val = __half2float(q_weight[h * (params.num_q_heads * kHeadDim) + head_idx * kHeadDim + d]);
                    acc += hidden_val * weight_val;
                }
                
                // Add bias if present
                if (params.has_bias && params.q_bias_ptr != nullptr) {
                    half *q_bias = reinterpret_cast<half*>(params.q_bias_ptr);
                    acc += __half2float(q_bias[head_idx * kHeadDim + d]);
                }
                
                // Write output
                int out_idx = batch_idx * params.q_batch_stride +
                             head_idx * params.q_head_stride +
                             global_seq_idx * params.q_row_stride +
                             d;
                q_out[out_idx] = __float2half(acc);
            }
        }
    }
    
    // Process K projection for this head (if within KV heads)
    if (head_idx < params.num_kv_heads) {
        half *k_weight = reinterpret_cast<half*>(params.k_weight_ptr);
        half *k_out = reinterpret_cast<half*>(params.k_out_ptr);
        
        for (int seq_offset = 0; seq_offset < kBlockM; seq_offset++) {
            int global_seq_idx = seq_start + seq_offset;
            if (global_seq_idx >= params.seqlen) break;
            
            for (int d = tid; d < kHeadDim; d += blockDim.x) {
                float acc = 0.0f;
                
                for (int h = 0; h < params.hidden_dim; h++) {
                    float hidden_val = __half2float(smem_hidden[seq_offset * params.hidden_dim + h]);
                    float weight_val = __half2float(k_weight[h * (params.num_kv_heads * kHeadDim) + head_idx * kHeadDim + d]);
                    acc += hidden_val * weight_val;
                }
                
                if (params.has_bias && params.k_bias_ptr != nullptr) {
                    half *k_bias = reinterpret_cast<half*>(params.k_bias_ptr);
                    acc += __half2float(k_bias[head_idx * kHeadDim + d]);
                }
                
                int out_idx = batch_idx * params.k_batch_stride +
                             head_idx * params.k_head_stride +
                             global_seq_idx * params.k_row_stride +
                             d;
                k_out[out_idx] = __float2half(acc);
            }
        }
    }
    
    // Process V projection for this head (if within KV heads)
    if (head_idx < params.num_kv_heads) {
        half *v_weight = reinterpret_cast<half*>(params.v_weight_ptr);
        half *v_out = reinterpret_cast<half*>(params.v_out_ptr);
        
        for (int seq_offset = 0; seq_offset < kBlockM; seq_offset++) {
            int global_seq_idx = seq_start + seq_offset;
            if (global_seq_idx >= params.seqlen) break;
            
            for (int d = tid; d < kHeadDim; d += blockDim.x) {
                float acc = 0.0f;
                
                for (int h = 0; h < params.hidden_dim; h++) {
                    float hidden_val = __half2float(smem_hidden[seq_offset * params.hidden_dim + h]);
                    float weight_val = __half2float(v_weight[h * (params.num_kv_heads * kHeadDim) + head_idx * kHeadDim + d]);
                    acc += hidden_val * weight_val;
                }
                
                if (params.has_bias && params.v_bias_ptr != nullptr) {
                    half *v_bias = reinterpret_cast<half*>(params.v_bias_ptr);
                    acc += __half2float(v_bias[head_idx * kHeadDim + d]);
                }
                
                int out_idx = batch_idx * params.v_batch_stride +
                             head_idx * params.v_head_stride +
                             global_seq_idx * params.v_row_stride +
                             d;
                v_out[out_idx] = __float2half(acc);
            }
        }
    }
}

// Host function to launch the kernel
void run_qkv_fusion_fp16(QKVFusedParams &params, cudaStream_t stream) {
    constexpr int kBlockM = 64;    // Process 64 sequence positions per block
    constexpr int kBlockN = 128;   // Not used in simple version
    constexpr int kHeadDim = 128;  // Qwen3 uses 128
    constexpr int kThreads = 256;
    
    // Grid dimensions
    // x: heads (use max of q_heads and kv_heads)
    // y: sequence blocks
    // z: batch
    int max_heads = std::max(params.num_q_heads, params.num_kv_heads);
    dim3 grid(
        max_heads,
        (params.seqlen + kBlockM - 1) / kBlockM,
        params.batch_size
    );
    dim3 block(kThreads);
    
    // Shared memory size
    size_t smem_size = (kBlockM * params.hidden_dim + kBlockM * kHeadDim) * sizeof(half);
    
    qkv_projection_kernel_simple<kBlockM, kBlockN, kHeadDim><<<grid, block, smem_size, stream>>>(
        params
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

} // namespace qkv_fusion

