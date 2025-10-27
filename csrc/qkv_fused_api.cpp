/******************************************************************************
 * QKV Fusion Kernel - C++ API and PyTorch Bindings
 ******************************************************************************/

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include "qkv_fused_params.h"

namespace qkv_fusion {

// Forward declarations of CUDA kernel launchers
// void run_qkv_fusion_fp16(QKVFusedParams &params, cudaStream_t stream);  // Not used
void run_qkv_fusion_optimized(QKVFusedParams &params, cublasLtHandle_t cublaslt_handle, cudaStream_t stream);

// Python-facing function (baseline - not used, commented out)
/*
std::vector<torch::Tensor> qkv_fused_forward(
    torch::Tensor hidden_states,      // [batch, seqlen, hidden_dim]
    torch::Tensor q_weight,           // [hidden_dim, num_q_heads * head_dim]
    torch::Tensor k_weight,           // [hidden_dim, num_kv_heads * head_dim]
    torch::Tensor v_weight,           // [hidden_dim, num_kv_heads * head_dim]
    c10::optional<torch::Tensor> q_bias,
    c10::optional<torch::Tensor> k_bias,
    c10::optional<torch::Tensor> v_bias,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim
) {
    // Input validation
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be on CUDA");
    TORCH_CHECK(q_weight.is_cuda(), "q_weight must be on CUDA");
    TORCH_CHECK(k_weight.is_cuda(), "k_weight must be on CUDA");
    TORCH_CHECK(v_weight.is_cuda(), "v_weight must be on CUDA");
    
    TORCH_CHECK(hidden_states.dtype() == torch::kFloat16, "Only FP16 supported in this version");
    TORCH_CHECK(hidden_states.dim() == 3, "hidden_states must be 3D [batch, seqlen, hidden_dim]");
    
    auto batch_size = hidden_states.size(0);
    auto seqlen = hidden_states.size(1);
    auto hidden_dim = hidden_states.size(2);
    
    // Validate weight dimensions
    TORCH_CHECK(q_weight.size(0) == hidden_dim, "q_weight dim 0 must match hidden_dim");
    TORCH_CHECK(q_weight.size(1) == num_q_heads * head_dim, "q_weight dim 1 mismatch");
    TORCH_CHECK(k_weight.size(0) == hidden_dim, "k_weight dim 0 must match hidden_dim");
    TORCH_CHECK(k_weight.size(1) == num_kv_heads * head_dim, "k_weight dim 1 mismatch");
    TORCH_CHECK(v_weight.size(0) == hidden_dim, "v_weight dim 0 must match hidden_dim");
    TORCH_CHECK(v_weight.size(1) == num_kv_heads * head_dim, "v_weight dim 1 mismatch");
    
    // Allocate output tensors
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(hidden_states.device());
    
    // Output format: [batch, num_heads, seqlen, head_dim]
    auto q_out = torch::empty({batch_size, num_q_heads, seqlen, head_dim}, options);
    auto k_out = torch::empty({batch_size, num_kv_heads, seqlen, head_dim}, options);
    auto v_out = torch::empty({batch_size, num_kv_heads, seqlen, head_dim}, options);
    
    // Set up parameters
    QKVFusedParams params;
    
    // Input pointers
    params.hidden_states_ptr = hidden_states.data_ptr();
    params.q_weight_ptr = q_weight.data_ptr();
    params.k_weight_ptr = k_weight.data_ptr();
    params.v_weight_ptr = v_weight.data_ptr();
    
    // Bias pointers
    params.has_bias = q_bias.has_value() && k_bias.has_value() && v_bias.has_value();
    params.q_bias_ptr = params.has_bias ? q_bias.value().data_ptr() : nullptr;
    params.k_bias_ptr = params.has_bias ? k_bias.value().data_ptr() : nullptr;
    params.v_bias_ptr = params.has_bias ? v_bias.value().data_ptr() : nullptr;
    
    // Output pointers
    params.q_out_ptr = q_out.data_ptr();
    params.k_out_ptr = k_out.data_ptr();
    params.v_out_ptr = v_out.data_ptr();
    
    // Strides for hidden states
    params.hidden_batch_stride = hidden_states.stride(0);
    params.hidden_row_stride = hidden_states.stride(1);
    
    // Strides for outputs
    params.q_batch_stride = q_out.stride(0);
    params.k_batch_stride = k_out.stride(0);
    params.v_batch_stride = v_out.stride(0);
    params.q_head_stride = q_out.stride(1);
    params.k_head_stride = k_out.stride(1);
    params.v_head_stride = v_out.stride(1);
    params.q_row_stride = q_out.stride(2);
    params.k_row_stride = k_out.stride(2);
    params.v_row_stride = v_out.stride(2);
    
    // Dimensions
    params.batch_size = batch_size;
    params.seqlen = seqlen;
    params.hidden_dim = hidden_dim;
    params.num_q_heads = num_q_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.num_groups = num_q_heads / num_kv_heads;
    
    // Quantization (not used in FP16 version)
    params.is_quantized = false;
    params.q_scales_ptr = nullptr;
    params.k_scales_ptr = nullptr;
    params.v_scales_ptr = nullptr;
    params.q_zeros_ptr = nullptr;
    params.k_zeros_ptr = nullptr;
    params.v_zeros_ptr = nullptr;
    params.group_size = 0;
    
    // Get CUDA stream
    at::cuda::CUDAGuard device_guard(hidden_states.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    run_qkv_fusion_fp16(params, stream);
    
    // Return Q, K, V tensors
    return {q_out, k_out, v_out};
}
*/

// Optimized forward function using fused weights
// Returns raw GEMM output [batch, seqlen, qkv_out_dim] for Python reshaping
torch::Tensor qkv_fused_forward_optimized(
    torch::Tensor hidden_states,      // [batch, seqlen, hidden_dim]
    torch::Tensor qkv_fused_weight,   // [hidden_dim, qkv_out_dim]
    c10::optional<torch::Tensor> qkv_fused_bias,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim
) {
    // Input validation
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be on CUDA");
    TORCH_CHECK(qkv_fused_weight.is_cuda(), "qkv_fused_weight must be on CUDA");
    TORCH_CHECK(hidden_states.dtype() == torch::kFloat16, "Only FP16 supported");
    TORCH_CHECK(hidden_states.dim() == 3, "hidden_states must be 3D [batch, seqlen, hidden_dim]");
    
    auto batch_size = hidden_states.size(0);
    auto seqlen = hidden_states.size(1);
    auto hidden_dim = hidden_states.size(2);
    
    const int qkv_out_dim = (num_q_heads + 2 * num_kv_heads) * head_dim;
    TORCH_CHECK(qkv_fused_weight.size(0) == hidden_dim, "qkv_fused_weight dim 0 must match hidden_dim");
    TORCH_CHECK(qkv_fused_weight.size(1) == qkv_out_dim, "qkv_fused_weight dim 1 mismatch");
    
    // Allocate output tensor for raw GEMM result
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(hidden_states.device());
    
    // Output shape: [batch, seqlen, qkv_out_dim]
    // Python will reshape this to [batch, seqlen, total_heads, head_dim]
    // then slice into Q, K, V
    auto qkv_output = torch::empty({batch_size, seqlen, qkv_out_dim}, options);
    
    // Set up parameters
    QKVFusedParams params;
    
    // Input pointers
    params.hidden_states_ptr = hidden_states.data_ptr();
    params.qkv_fused_weight_ptr = qkv_fused_weight.data_ptr();
    params.qkv_fused_bias_ptr = qkv_fused_bias.has_value() ? qkv_fused_bias.value().data_ptr() : nullptr;
    
    // Workspace pointer (output goes here)
    params.workspace_ptr = qkv_output.data_ptr();
    
    // Dimensions
    params.batch_size = batch_size;
    params.seqlen = seqlen;
    params.hidden_dim = hidden_dim;
    params.num_q_heads = num_q_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.num_groups = num_q_heads / num_kv_heads;
    
    // Flags
    params.has_bias = qkv_fused_bias.has_value();
    params.is_quantized = false;
    
    // Get CUDA stream and create cuBLASLt handle
    at::cuda::CUDAGuard device_guard(hidden_states.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Create cuBLASLt handle (lightweight, but ideally should be cached)
    cublasLtHandle_t cublaslt_handle;
    cublasLtCreate(&cublaslt_handle);
    
    // Launch optimized kernel (just GEMM with bias epilogue, no split)
    run_qkv_fusion_optimized(params, cublaslt_handle, stream);
    
    // Cleanup
    cublasLtDestroy(cublaslt_handle);
    
    // Return raw output - Python will do the reshaping (zero-copy!)
    return qkv_output;
}

} // namespace qkv_fusion

// PyTorch bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Baseline qkv_fused_forward is commented out (not used)
    /*
    m.def("qkv_fused_forward", &qkv_fusion::qkv_fused_forward,
          "QKV Fused Forward Pass (FP16, Phase 1)",
          py::arg("hidden_states"),
          py::arg("q_weight"),
          py::arg("k_weight"),
          py::arg("v_weight"),
          py::arg("q_bias") = py::none(),
          py::arg("k_bias") = py::none(),
          py::arg("v_bias") = py::none(),
          py::arg("num_q_heads"),
          py::arg("num_kv_heads"),
          py::arg("head_dim")
    );
    */
    
    m.def("qkv_fused_forward_optimized", &qkv_fusion::qkv_fused_forward_optimized,
          "QKV Fused Forward Pass (Optimized with CUTLASS, Phase 2)",
          py::arg("hidden_states"),
          py::arg("qkv_fused_weight"),
          py::arg("qkv_fused_bias") = py::none(),
          py::arg("num_q_heads"),
          py::arg("num_kv_heads"),
          py::arg("head_dim")
    );
}

