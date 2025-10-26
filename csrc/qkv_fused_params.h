/******************************************************************************
 * QKV Fusion Kernel - Parameter Structures
 * Based on FlashAttention architecture
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <stdint.h>

namespace qkv_fusion {

struct QKVFusedParams {
    using index_t = int64_t;
    
    // Input: hidden states from transformer layer
    void *__restrict__ hidden_states_ptr;  // [batch, seqlen, hidden_dim]
    
    // QKV weight matrices (separate - for Phase 1)
    void *__restrict__ q_weight_ptr;  // [hidden_dim, num_q_heads * head_dim]
    void *__restrict__ k_weight_ptr;  // [hidden_dim, num_kv_heads * head_dim]
    void *__restrict__ v_weight_ptr;  // [hidden_dim, num_kv_heads * head_dim]
    
    // Optional biases (separate - for Phase 1)
    void *__restrict__ q_bias_ptr;    // [num_q_heads * head_dim] or nullptr
    void *__restrict__ k_bias_ptr;    // [num_kv_heads * head_dim] or nullptr
    void *__restrict__ v_bias_ptr;    // [num_kv_heads * head_dim] or nullptr
    
    // Fused QKV weight and bias (for Phase 2 optimized path)
    void *__restrict__ qkv_fused_weight_ptr;  // [hidden_dim, qkv_out_dim]
    void *__restrict__ qkv_fused_bias_ptr;    // [qkv_out_dim] or nullptr
    
    // Output: Q, K, V tensors
    void *__restrict__ q_out_ptr;     // [batch, num_q_heads, seqlen, head_dim]
    void *__restrict__ k_out_ptr;     // [batch, num_kv_heads, seqlen, head_dim]
    void *__restrict__ v_out_ptr;     // [batch, num_kv_heads, seqlen, head_dim]
    
    // Workspace buffer (pre-allocated by PyTorch)
    void *__restrict__ workspace_ptr;  // [batch * seqlen, qkv_out_dim]
    
    // Strides for hidden states
    index_t hidden_batch_stride;
    index_t hidden_row_stride;
    
    // Strides for outputs
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    
    // Dimensions
    int batch_size;
    int seqlen;
    int hidden_dim;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    
    // For GQA: num_q_heads / num_kv_heads
    int num_groups;
    
    // Quantization parameters (for INT4 version)
    void *__restrict__ q_scales_ptr;  // Quantization scales
    void *__restrict__ k_scales_ptr;
    void *__restrict__ v_scales_ptr;
    void *__restrict__ q_zeros_ptr;   // Zero points
    void *__restrict__ k_zeros_ptr;
    void *__restrict__ v_zeros_ptr;
    int group_size;                    // GPTQ group size (e.g., 128)
    
    // Flags
    bool is_quantized;                 // true for INT4, false for FP16
    bool has_bias;                     // true if biases are provided
};

} // namespace qkv_fusion

