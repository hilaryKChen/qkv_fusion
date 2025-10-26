"""
High-level interface for QKV fusion with FlashAttention integration.

This module provides convenient wrappers that handle:
- QKV projection with fused kernel
- Transpose for FlashAttention compatibility
- Integration with attention mechanisms
"""

import torch
from typing import Optional, Tuple
import math

from . import qkv_fused_forward_optimized
from .weight_utils import prepare_fused_qkv_weights


def qkv_projection_for_flash_attention(
    hidden_states: torch.Tensor,
    qkv_fused_weight: torch.Tensor,
    qkv_fused_bias: Optional[torch.Tensor] = None,
    num_q_heads: int = 32,
    num_kv_heads: int = 4,
    head_dim: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute QKV projections using fused kernel and prepare for FlashAttention.
    
    This function:
    1. Runs the optimized cuBLAS fused QKV kernel
    2. Transposes outputs to FlashAttention's expected format [batch, seq, heads, head_dim]
    
    Args:
        hidden_states: Input tensor [batch, seq, hidden_dim]
        qkv_fused_weight: Concatenated QKV weight [hidden_dim, qkv_out_dim]
        qkv_fused_bias: Optional concatenated bias [qkv_out_dim]
        num_q_heads: Number of query heads (e.g., 32 for Qwen3)
        num_kv_heads: Number of key/value heads (e.g., 4 for Qwen3 GQA)
        head_dim: Dimension per head (e.g., 128 for Qwen3)
    
    Returns:
        Tuple of (Q, K, V) tensors, each with shape:
        - Q: [batch, seq, num_q_heads, head_dim]
        - K: [batch, seq, num_kv_heads, head_dim]
        - V: [batch, seq, num_kv_heads, head_dim]
        
        These can be directly fed to flash_attn_func().
    
    Example:
        >>> from flash_attn import flash_attn_func
        >>> 
        >>> # Prepare fused weights
        >>> fused_weight, fused_bias = prepare_fused_qkv_weights(q_w, k_w, v_w)
        >>> 
        >>> # Get Q, K, V for FlashAttention
        >>> q, k, v = qkv_projection_for_flash_attention(
        ...     hidden_states, fused_weight, fused_bias,
        ...     num_q_heads=32, num_kv_heads=4, head_dim=128
        ... )
        >>> 
        >>> # Run FlashAttention
        >>> attn_output = flash_attn_func(q, k, v, causal=True)
    """
    # Step 1: Run fused QKV kernel
    # Output: [batch, heads, seq, head_dim]
    q, k, v = qkv_fused_forward_optimized(
        hidden_states,
        qkv_fused_weight,
        qkv_fused_bias,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )
    
    # Step 2: Transpose to FlashAttention format [batch, seq, heads, head_dim]
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    
    return q, k, v


def fused_qkv_attention(
    hidden_states: torch.Tensor,
    qkv_fused_weight: torch.Tensor,
    qkv_fused_bias: Optional[torch.Tensor] = None,
    num_q_heads: int = 32,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """
    Complete fused QKV projection + FlashAttention in one call.
    
    This is a convenience function that combines:
    1. Fused QKV projection (your optimized kernel)
    2. FlashAttention-2 (efficient attention computation)
    
    Args:
        hidden_states: Input tensor [batch, seq, hidden_dim]
        qkv_fused_weight: Concatenated QKV weight [hidden_dim, qkv_out_dim]
        qkv_fused_bias: Optional concatenated bias [qkv_out_dim]
        num_q_heads: Number of query heads
        num_kv_heads: Number of key/value heads (for GQA)
        head_dim: Dimension per head
        causal: Whether to apply causal mask (for autoregressive models)
        softmax_scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
        dropout_p: Dropout probability (0.0 for inference)
        window_size: Sliding window attention (default: infinite context)
    
    Returns:
        Attention output tensor [batch, seq, num_q_heads * head_dim]
    
    Example:
        >>> # Single function call for complete attention
        >>> attn_output = fused_qkv_attention(
        ...     hidden_states, fused_weight, fused_bias,
        ...     num_q_heads=32, num_kv_heads=4, head_dim=128,
        ...     causal=True
        ... )
    """
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        raise ImportError(
            "FlashAttention is required for this function. "
            "Install with: pip install flash-attn --no-build-isolation"
        )
    
    # Default softmax scale
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    
    # Get Q, K, V in FlashAttention format
    q, k, v = qkv_projection_for_flash_attention(
        hidden_states,
        qkv_fused_weight,
        qkv_fused_bias,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )
    
    # Run FlashAttention
    attn_output = flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
    )
    # Output: [batch, seq, num_q_heads, head_dim]
    
    # Reshape to [batch, seq, num_q_heads * head_dim]
    batch, seq, heads, dim = attn_output.shape
    attn_output = attn_output.reshape(batch, seq, heads * dim)
    
    return attn_output


class FusedQKVAttention(torch.nn.Module):
    """
    PyTorch module for fused QKV projection + FlashAttention.
    
    This module can be used as a drop-in replacement for standard attention layers
    in transformer models like Qwen3.
    
    Args:
        hidden_dim: Input hidden dimension (e.g., 3584 for Qwen3-7B)
        num_q_heads: Number of query heads (e.g., 32)
        num_kv_heads: Number of key/value heads (e.g., 4 for GQA)
        head_dim: Dimension per head (e.g., 128)
        bias: Whether to use bias in projections
        dropout: Dropout probability (only used during training)
        causal: Whether to apply causal masking
    
    Example:
        >>> # Create attention module
        >>> attn = FusedQKVAttention(
        ...     hidden_dim=3584,
        ...     num_q_heads=32,
        ...     num_kv_heads=4,
        ...     head_dim=128,
        ...     bias=True,
        ...     causal=True
        ... )
        >>> 
        >>> # Use in forward pass
        >>> hidden_states = torch.randn(2, 512, 3584, device='cuda', dtype=torch.float16)
        >>> output = attn(hidden_states)
        >>> print(output.shape)  # [2, 512, 4096]
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bias: bool = True,
        dropout: float = 0.0,
        causal: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.causal = causal
        self.dropout = dropout
        
        # Calculate output dimensions
        q_out_dim = num_q_heads * head_dim
        kv_out_dim = num_kv_heads * head_dim
        qkv_out_dim = q_out_dim + 2 * kv_out_dim
        
        # Create fused QKV weight and bias
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.qkv_weight = torch.nn.Parameter(
            torch.empty((hidden_dim, qkv_out_dim), **factory_kwargs)
        )
        if bias:
            self.qkv_bias = torch.nn.Parameter(
                torch.empty(qkv_out_dim, **factory_kwargs)
            )
        else:
            self.register_parameter('qkv_bias', None)
        
        # Output projection
        self.o_proj = torch.nn.Linear(
            q_out_dim, hidden_dim, bias=bias, **factory_kwargs
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using standard initialization."""
        torch.nn.init.xavier_uniform_(self.qkv_weight)
        if self.qkv_bias is not None:
            torch.nn.init.zeros_(self.qkv_bias)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        if self.o_proj.bias is not None:
            torch.nn.init.zeros_(self.o_proj.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with fused QKV projection and FlashAttention.
        
        Args:
            hidden_states: Input tensor [batch, seq, hidden_dim]
            attention_mask: Not used (FlashAttention handles causal masking internally)
        
        Returns:
            Output tensor [batch, seq, hidden_dim]
        """
        # Fused QKV + FlashAttention
        attn_output = fused_qkv_attention(
            hidden_states,
            self.qkv_weight,
            self.qkv_bias,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            causal=self.causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output
    
    @classmethod
    def from_separate_projections(
        cls,
        q_proj: torch.nn.Linear,
        k_proj: torch.nn.Linear,
        v_proj: torch.nn.Linear,
        o_proj: torch.nn.Linear,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        causal: bool = True,
        dropout: float = 0.0,
    ):
        """
        Create FusedQKVAttention from existing separate Q, K, V projection layers.
        
        This is useful for converting existing models to use fused QKV.
        
        Args:
            q_proj: Existing query projection layer
            k_proj: Existing key projection layer
            v_proj: Existing value projection layer
            o_proj: Existing output projection layer
            num_q_heads: Number of query heads
            num_kv_heads: Number of key/value heads
            head_dim: Dimension per head
            causal: Whether to use causal masking
            dropout: Dropout probability
        
        Returns:
            FusedQKVAttention module with weights copied from input layers
        
        Example:
            >>> # Convert existing Qwen3 attention layer
            >>> original_attn = model.layers[0].self_attn
            >>> fused_attn = FusedQKVAttention.from_separate_projections(
            ...     original_attn.q_proj,
            ...     original_attn.k_proj,
            ...     original_attn.v_proj,
            ...     original_attn.o_proj,
            ...     num_q_heads=32,
            ...     num_kv_heads=4,
            ...     head_dim=128
            ... )
        """
        hidden_dim = q_proj.in_features
        has_bias = q_proj.bias is not None
        
        # Create module
        module = cls(
            hidden_dim=hidden_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bias=has_bias,
            dropout=dropout,
            causal=causal,
            device=q_proj.weight.device,
            dtype=q_proj.weight.dtype,
        )
        
        # Prepare fused weights from separate projections
        q_weight = q_proj.weight.t().contiguous()  # [in, out] -> [out, in] -> transpose
        k_weight = k_proj.weight.t().contiguous()
        v_weight = v_proj.weight.t().contiguous()
        
        fused_weight, fused_bias = prepare_fused_qkv_weights(
            q_weight, k_weight, v_weight,
            q_proj.bias if has_bias else None,
            k_proj.bias if has_bias else None,
            v_proj.bias if has_bias else None,
        )
        
        # Copy weights
        module.qkv_weight.data.copy_(fused_weight)
        if has_bias:
            module.qkv_bias.data.copy_(fused_bias)
        
        # Copy output projection
        module.o_proj.weight.data.copy_(o_proj.weight)
        if o_proj.bias is not None:
            module.o_proj.bias.data.copy_(o_proj.bias)
        
        return module


__all__ = [
    'qkv_projection_for_flash_attention',
    'fused_qkv_attention',
    'FusedQKVAttention',
]

