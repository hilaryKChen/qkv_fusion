"""
Weight Preparation Utilities for QKV Fusion
Based on FasterTransformer's approach
"""

import torch
from typing import Tuple, Optional

def prepare_fused_qkv_weights(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_bias: Optional[torch.Tensor] = None,
    k_bias: Optional[torch.Tensor] = None,
    v_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Concatenate Q, K, V weights for fused GEMM operation.
    
    This follows FasterTransformer's approach of concatenating weights
    along the output dimension to enable a single GEMM for all projections.
    
    Args:
        q_weight: Query weight [hidden_dim, num_q_heads * head_dim]
        k_weight: Key weight [hidden_dim, num_kv_heads * head_dim]
        v_weight: Value weight [hidden_dim, num_kv_heads * head_dim]
        q_bias: Optional query bias [num_q_heads * head_dim]
        k_bias: Optional key bias [num_kv_heads * head_dim]
        v_bias: Optional value bias [num_kv_heads * head_dim]
    
    Returns:
        fused_weight: Concatenated weight [hidden_dim, (num_q_heads + 2*num_kv_heads) * head_dim]
        fused_bias: Concatenated bias or None
    
    Example:
        >>> # Qwen3: 32 Q heads, 4 KV heads, head_dim=128
        >>> q_weight.shape  # [3584, 4096]  (32 * 128)
        >>> k_weight.shape  # [3584, 512]   (4 * 128)
        >>> v_weight.shape  # [3584, 512]   (4 * 128)
        >>> 
        >>> fused_weight, fused_bias = prepare_fused_qkv_weights(
        ...     q_weight, k_weight, v_weight
        ... )
        >>> fused_weight.shape  # [3584, 5120]  (4096 + 512 + 512)
    """
    # Validate shapes
    assert q_weight.shape[0] == k_weight.shape[0] == v_weight.shape[0], \
        "All weights must have same hidden_dim"
    assert k_weight.shape[1] == v_weight.shape[1], \
        "K and V must have same output dim (num_kv_heads * head_dim)"
    
    # Concatenate weights along output dimension (dim=1)
    # Layout: [Q_all | K_all | V_all]
    fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=1)
    
    # Concatenate biases if present
    fused_bias = None
    if q_bias is not None and k_bias is not None and v_bias is not None:
        fused_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
    
    return fused_weight, fused_bias


def prepare_fused_qkv_from_model(
    model_layer,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract and fuse QKV weights from a transformer attention layer.
    
    Args:
        model_layer: Attention layer with q_proj, k_proj, v_proj attributes
        device: Target device (default: same as weights)
        dtype: Target dtype (default: same as weights)
    
    Returns:
        fused_weight: Concatenated QKV weight
        fused_bias: Concatenated QKV bias or None
    
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
        >>> 
        >>> # Prepare fused weights for first layer
        >>> attn_layer = model.model.layers[0].self_attn
        >>> fused_weight, fused_bias = prepare_fused_qkv_from_model(attn_layer)
        >>> 
        >>> # Now use with fused kernel
        >>> q, k, v = qkv_fused_forward_optimized(
        ...     hidden_states, fused_weight, fused_bias, ...
        ... )
    """
    # Extract weights (need to transpose for our GEMM layout)
    q_weight = model_layer.q_proj.weight.t().contiguous()
    k_weight = model_layer.k_proj.weight.t().contiguous()
    v_weight = model_layer.v_proj.weight.t().contiguous()
    
    # Extract biases if present
    q_bias = model_layer.q_proj.bias if hasattr(model_layer.q_proj, 'bias') else None
    k_bias = model_layer.k_proj.bias if hasattr(model_layer.k_proj, 'bias') else None
    v_bias = model_layer.v_proj.bias if hasattr(model_layer.v_proj, 'bias') else None
    
    # Prepare fused weights
    fused_weight, fused_bias = prepare_fused_qkv_weights(
        q_weight, k_weight, v_weight,
        q_bias, k_bias, v_bias
    )
    
    # Move to target device/dtype if specified
    if device is not None:
        fused_weight = fused_weight.to(device)
        if fused_bias is not None:
            fused_bias = fused_bias.to(device)
    
    if dtype is not None:
        fused_weight = fused_weight.to(dtype)
        if fused_bias is not None:
            fused_bias = fused_bias.to(dtype)
    
    return fused_weight, fused_bias


def split_fused_qkv_output(
    qkv_output: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split fused QKV output into separate Q, K, V tensors.
    
    This is a PyTorch reference implementation for testing.
    In production, use the CUDA kernel for better performance.
    
    Args:
        qkv_output: Fused output [batch, seq_len, (num_q_heads + 2*num_kv_heads) * head_dim]
        num_q_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_dim: Dimension per head
    
    Returns:
        q: [batch, num_q_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        v: [batch, num_kv_heads, seq_len, head_dim]
    """
    batch_size, seq_len, _ = qkv_output.shape
    
    # Split along last dimension
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    
    q_flat = qkv_output[:, :, :q_size]
    k_flat = qkv_output[:, :, q_size:q_size + kv_size]
    v_flat = qkv_output[:, :, q_size + kv_size:]
    
    # Reshape and transpose to [batch, num_heads, seq_len, head_dim]
    q = q_flat.view(batch_size, seq_len, num_q_heads, head_dim).transpose(1, 2)
    k = k_flat.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v_flat.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    
    return q, k, v


def benchmark_weight_fusion():
    """
    Benchmark to show memory savings from weight fusion.
    """
    import time
    
    # Qwen3 configuration
    hidden_dim = 3584
    num_q_heads = 32
    num_kv_heads = 4
    head_dim = 128
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    # Create separate weights
    q_weight = torch.randn(hidden_dim, num_q_heads * head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    v_weight = torch.randn(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    
    # Memory usage
    separate_memory = (q_weight.numel() + k_weight.numel() + v_weight.numel()) * 2  # FP16 = 2 bytes
    print(f"Separate weights memory: {separate_memory / 1024 / 1024:.2f} MB")
    
    # Fuse weights
    fused_weight, _ = prepare_fused_qkv_weights(q_weight, k_weight, v_weight)
    fused_memory = fused_weight.numel() * 2
    print(f"Fused weight memory: {fused_memory / 1024 / 1024:.2f} MB")
    print(f"Memory overhead: {(fused_memory - separate_memory) / 1024:.2f} KB")
    print(f"(Overhead is just from contiguous storage, negligible)")
    
    # Benchmark access time
    batch_size = 4
    seq_len = 512
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(10):
        _ = torch.matmul(hidden_states, q_weight)
        _ = torch.matmul(hidden_states, fused_weight)
    torch.cuda.synchronize()
    
    # Benchmark separate
    num_iters = 100
    start = time.time()
    for _ in range(num_iters):
        q = torch.matmul(hidden_states, q_weight)
        k = torch.matmul(hidden_states, k_weight)
        v = torch.matmul(hidden_states, v_weight)
    torch.cuda.synchronize()
    separate_time = (time.time() - start) / num_iters * 1000
    
    # Benchmark fused
    start = time.time()
    for _ in range(num_iters):
        qkv = torch.matmul(hidden_states, fused_weight)
    torch.cuda.synchronize()
    fused_time = (time.time() - start) / num_iters * 1000
    
    print(f"\nBenchmark (PyTorch matmul):")
    print(f"  Separate (3 GEMMs): {separate_time:.3f} ms")
    print(f"  Fused (1 GEMM):     {fused_time:.3f} ms")
    print(f"  Speedup:            {separate_time / fused_time:.2f}x")
    print(f"\nNote: CUDA kernel will be even faster due to fused split+transpose!")


if __name__ == "__main__":
    benchmark_weight_fusion()

