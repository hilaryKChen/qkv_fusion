"""
QKV Fusion Module - Fused QKV Projection for Transformer Attention

This module provides optimized CUDA kernels for fusing Q, K, V linear projections
in transformer attention layers, specifically designed for:
- Grouped Query Attention (GQA)
- Quantized models (INT4 GPTQ)
- Integration with FlashAttention-2

Usage:
    from qkv_fusion import qkv_fused_forward
    
    q, k, v = qkv_fused_forward(
        hidden_states,
        q_weight, k_weight, v_weight,
        num_q_heads=32,
        num_kv_heads=4,
        head_dim=128
    )
"""

import torch
from typing import Optional, Tuple

# Import the compiled CUDA extension
try:
    import qkv_fusion_cuda
except ImportError as e:
    raise ImportError(
        "Failed to import qkv_fusion_cuda. "
        "Make sure you have compiled the extension with: "
        "pip install -e ."
    ) from e

# Baseline qkv_fused_forward is commented out (not used)
"""
def qkv_fused_forward(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_bias: Optional[torch.Tensor] = None,
    k_bias: Optional[torch.Tensor] = None,
    v_bias: Optional[torch.Tensor] = None,
    num_q_heads: int = 32,
    num_kv_heads: int = 4,
    head_dim: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    Fused QKV projection for transformer attention.
    
    Args:
        hidden_states: Input tensor [batch, seqlen, hidden_dim]
        q_weight: Query weight matrix [hidden_dim, num_q_heads * head_dim]
        k_weight: Key weight matrix [hidden_dim, num_kv_heads * head_dim]
        v_weight: Value weight matrix [hidden_dim, num_kv_heads * head_dim]
        q_bias: Optional query bias [num_q_heads * head_dim]
        k_bias: Optional key bias [num_kv_heads * head_dim]
        v_bias: Optional value bias [num_kv_heads * head_dim]
        num_q_heads: Number of query heads (e.g., 32 for Qwen3)
        num_kv_heads: Number of key/value heads (e.g., 4 for Qwen3 GQA)
        head_dim: Dimension per head (e.g., 128 for Qwen3)
    
    Returns:
        Tuple of (Q, K, V) tensors:
        - Q: [batch, num_q_heads, seqlen, head_dim]
        - K: [batch, num_kv_heads, seqlen, head_dim]
        - V: [batch, num_kv_heads, seqlen, head_dim]
    
    return qkv_fusion_cuda.qkv_fused_forward(
        hidden_states,
        q_weight,
        k_weight,
        v_weight,
        q_bias,
        k_bias,
        v_bias,
        num_q_heads,
        num_kv_heads,
        head_dim
    )
"""

def qkv_fused_forward_optimized(
    hidden_states: torch.Tensor,
    qkv_fused_weight: torch.Tensor,
    qkv_fused_bias: Optional[torch.Tensor] = None,
    num_q_heads: int = 32,
    num_kv_heads: int = 4,
    head_dim: int = 128,
) :
    """
    Optimized fused QKV projection using FasterTransformer approach.
    
    This uses:
    1. Single GEMM with concatenated weights (cuBLAS with tensor cores)
    2. Efficient split + bias + transpose kernel
    
    Expected speedup: 2-3x vs baseline
    
    Args:
        hidden_states: Input tensor [batch, seqlen, hidden_dim]
        qkv_fused_weight: Concatenated weight [hidden_dim, qkv_out_dim]
                         where qkv_out_dim = (num_q_heads + 2*num_kv_heads) * head_dim
        qkv_fused_bias: Optional concatenated bias [qkv_out_dim]
        num_q_heads: Number of query heads (e.g., 32 for Qwen3)
        num_kv_heads: Number of key/value heads (e.g., 4 for Qwen3 GQA)
        head_dim: Dimension per head (e.g., 128 for Qwen3)
    
    Returns:
        Tuple of (Q, K, V) tensors:
        - Q: [batch, num_q_heads, seqlen, head_dim]
        - K: [batch, num_kv_heads, seqlen, head_dim]
        - V: [batch, num_kv_heads, seqlen, head_dim]
    
    Example:
        >>> from qkv_fusion.weight_utils import prepare_fused_qkv_weights
        >>> 
        >>> # Prepare fused weights
        >>> fused_weight, fused_bias = prepare_fused_qkv_weights(
        ...     q_weight, k_weight, v_weight
        ... )
        >>> 
        >>> # Run optimized kernel
        >>> q, k, v = qkv_fused_forward_optimized(
        ...     hidden_states, fused_weight, fused_bias,
        ...     num_q_heads=32, num_kv_heads=4, head_dim=128
        ... )
    """
    return qkv_fusion_cuda.qkv_fused_forward_optimized(
        hidden_states,
        qkv_fused_weight,
        qkv_fused_bias,
        num_q_heads,
        num_kv_heads,
        head_dim
    )

def qkv_fused_forward_lightweight(
    hidden_states: torch.Tensor,
    qkv_fused_weight: torch.Tensor,
    qkv_fused_bias: Optional[torch.Tensor] = None,
    num_q_heads: int = 32,
    num_kv_heads: int = 4,
    head_dim: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Lightweight fused QKV projection: GEMM+Bias in CUDA, split+transpose in PyTorch.
    
    This approach minimizes custom CUDA code and leverages PyTorch's highly
    optimized tensor operations for reshape/transpose.
    
    Strategy:
    1. cuBLAS GEMM with concatenated weights (highly optimized)
    2. Simple element-wise bias add kernel (memory bandwidth bound, ~free)
    3. Split Q/K/V and transpose in PyTorch (nearly free view operations)
    
    Expected performance: Similar to 3 separate nn.Linear, much faster than
    the complex split+bias+transpose CUDA kernel.
    
    Args:
        hidden_states: Input tensor [batch, seqlen, hidden_dim]
        qkv_fused_weight: Concatenated weight [hidden_dim, qkv_out_dim]
                         where qkv_out_dim = (num_q_heads + 2*num_kv_heads) * head_dim
        qkv_fused_bias: Optional concatenated bias [qkv_out_dim]
        num_q_heads: Number of query heads (e.g., 32 for Qwen3)
        num_kv_heads: Number of key/value heads (e.g., 4 for Qwen3 GQA)
        head_dim: Dimension per head (e.g., 128 for Qwen3)
    
    Returns:
        Tuple of (Q, K, V) tensors:
        - Q: [batch, num_q_heads, seqlen, head_dim]
        - K: [batch, num_kv_heads, seqlen, head_dim]
        - V: [batch, num_kv_heads, seqlen, head_dim]
    
    Example:
        >>> from qkv_fusion.weight_utils import prepare_fused_qkv_weights
        >>> 
        >>> # Prepare fused weights
        >>> fused_weight, fused_bias = prepare_fused_qkv_weights(
        ...     q_weight, k_weight, v_weight
        ... )
        >>> 
        >>> # Run lightweight kernel
        >>> q, k, v = qkv_fused_forward_lightweight(
        ...     hidden_states, fused_weight, fused_bias,
        ...     num_q_heads=32, num_kv_heads=4, head_dim=128
        ... )
    """
    batch_size, seqlen, hidden_dim = hidden_states.shape
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    
    # Step 1: GEMM only (no bias in CUDA)
    # Output: [batch, seqlen, q_dim + 2*kv_dim]
    qkv_output = qkv_fusion_cuda.qkv_fused_forward_lightweight(
        hidden_states,
        qkv_fused_weight,
        None,  # Pass None for bias - we'll add it in PyTorch
        num_q_heads,
        num_kv_heads,
        head_dim
    )
    
    # Step 2: Add bias using PyTorch (highly optimized)
    if qkv_fused_bias is not None:
        qkv_output = qkv_output + qkv_fused_bias
    
    # Step 3: Split Q, K, V (PyTorch view operation - no data copy)
    q = qkv_output[:, :, :q_dim]
    k = qkv_output[:, :, q_dim:q_dim + kv_dim]
    v = qkv_output[:, :, q_dim + kv_dim:]
    
    # Step 4: Reshape and transpose (PyTorch view + transpose - minimal overhead)
    # [batch, seqlen, num_heads * head_dim] -> [batch, seqlen, num_heads, head_dim] -> [batch, num_heads, seqlen, head_dim]
    q = q.view(batch_size, seqlen, num_q_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
    
    return q, k, v

__version__ = "0.1.0"

# Import high-level interfaces
from .qkv_interface import (
    qkv_projection_for_flash_attention,
    fused_qkv_attention,
    FusedQKVAttention,
)

__all__ = [
    # Low-level kernel interfaces
    # "qkv_fused_forward",  # Baseline - not used
    "qkv_fused_forward_optimized",
    "qkv_fused_forward_lightweight",
    # High-level FlashAttention integration
    "qkv_projection_for_flash_attention",
    "fused_qkv_attention",
    "FusedQKVAttention",
]

