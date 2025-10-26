#!/usr/bin/env python3
"""
Example: Using QKV Fusion with FlashAttention-2

This script demonstrates three ways to use the QKV fusion kernel with FlashAttention:
1. Low-level: Manual kernel call + transpose
2. Mid-level: Using qkv_projection_for_flash_attention()
3. High-level: Using FusedQKVAttention module
"""

import torch
import math
from flash_attn import flash_attn_func

from qkv_fusion import (
    qkv_fused_forward_optimized,
    qkv_projection_for_flash_attention,
    fused_qkv_attention,
    FusedQKVAttention,
)
from qkv_fusion.weight_utils import prepare_fused_qkv_weights


def example_1_low_level():
    """
    Example 1: Low-level usage with manual transpose
    
    This gives you full control over each step.
    """
    print("=" * 80)
    print("Example 1: Low-level Usage")
    print("=" * 80)
    
    # Configuration (Qwen3-7B)
    batch_size = 2
    seqlen = 512
    hidden_dim = 3584
    num_q_heads = 32
    num_kv_heads = 4
    head_dim = 128
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    # Create inputs
    hidden_states = torch.randn(batch_size, seqlen, hidden_dim, dtype=dtype, device=device)
    
    # Create weight matrices
    q_weight = torch.randn(hidden_dim, num_q_heads * head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    v_weight = torch.randn(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    
    # Prepare fused weights
    fused_weight, fused_bias = prepare_fused_qkv_weights(q_weight, k_weight, v_weight)
    
    print(f"Input: {hidden_states.shape}")
    print(f"Fused weight: {fused_weight.shape}")
    
    # Step 1: Run fused QKV kernel
    q, k, v = qkv_fused_forward_optimized(
        hidden_states,
        fused_weight,
        fused_bias,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )
    print(f"\nAfter QKV kernel:")
    print(f"  Q: {q.shape}")  # [batch, 32, seq, 128]
    print(f"  K: {k.shape}")  # [batch, 4, seq, 128]
    print(f"  V: {v.shape}")  # [batch, 4, seq, 128]
    
    # Step 2: Transpose for FlashAttention
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    print(f"\nAfter transpose:")
    print(f"  Q: {q.shape}")  # [batch, seq, 32, 128]
    print(f"  K: {k.shape}")  # [batch, seq, 4, 128]
    print(f"  V: {v.shape}")  # [batch, seq, 4, 128]
    
    # Step 3: Run FlashAttention
    attn_output = flash_attn_func(
        q, k, v,
        causal=True,
        softmax_scale=1.0 / math.sqrt(head_dim)
    )
    print(f"\nFlashAttention output: {attn_output.shape}")  # [batch, seq, 32, 128]
    
    # Step 4: Reshape for output projection
    batch, seq, heads, dim = attn_output.shape
    attn_output = attn_output.reshape(batch, seq, heads * dim)
    print(f"Final output: {attn_output.shape}")  # [batch, seq, 4096]
    
    print("\n✓ Low-level example completed successfully!\n")
    return attn_output


def example_2_mid_level():
    """
    Example 2: Mid-level usage with qkv_projection_for_flash_attention()
    
    This handles the transpose automatically.
    """
    print("=" * 80)
    print("Example 2: Mid-level Usage")
    print("=" * 80)
    
    # Configuration
    batch_size = 2
    seqlen = 512
    hidden_dim = 3584
    num_q_heads = 32
    num_kv_heads = 4
    head_dim = 128
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    # Create inputs
    hidden_states = torch.randn(batch_size, seqlen, hidden_dim, dtype=dtype, device=device)
    q_weight = torch.randn(hidden_dim, num_q_heads * head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    v_weight = torch.randn(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    
    fused_weight, fused_bias = prepare_fused_qkv_weights(q_weight, k_weight, v_weight)
    
    print(f"Input: {hidden_states.shape}")
    
    # Single function call: QKV kernel + transpose
    q, k, v = qkv_projection_for_flash_attention(
        hidden_states,
        fused_weight,
        fused_bias,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )
    print(f"\nQ, K, V ready for FlashAttention:")
    print(f"  Q: {q.shape}")  # [batch, seq, 32, 128]
    print(f"  K: {k.shape}")  # [batch, seq, 4, 128]
    print(f"  V: {v.shape}")  # [batch, seq, 4, 128]
    
    # Run FlashAttention
    attn_output = flash_attn_func(q, k, v, causal=True)
    print(f"\nFlashAttention output: {attn_output.shape}")
    
    print("\n✓ Mid-level example completed successfully!\n")
    return attn_output


def example_3_high_level():
    """
    Example 3: High-level usage with fused_qkv_attention()
    
    This is the simplest - one function call for everything.
    """
    print("=" * 80)
    print("Example 3: High-level Usage (Single Function)")
    print("=" * 80)
    
    # Configuration
    batch_size = 2
    seqlen = 512
    hidden_dim = 3584
    num_q_heads = 32
    num_kv_heads = 4
    head_dim = 128
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    # Create inputs
    hidden_states = torch.randn(batch_size, seqlen, hidden_dim, dtype=dtype, device=device)
    q_weight = torch.randn(hidden_dim, num_q_heads * head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    v_weight = torch.randn(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    
    fused_weight, fused_bias = prepare_fused_qkv_weights(q_weight, k_weight, v_weight)
    
    print(f"Input: {hidden_states.shape}")
    
    # Single function call: QKV kernel + FlashAttention + reshape
    attn_output = fused_qkv_attention(
        hidden_states,
        fused_weight,
        fused_bias,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=True
    )
    print(f"Attention output: {attn_output.shape}")  # [batch, seq, 4096]
    
    print("\n✓ High-level example completed successfully!\n")
    return attn_output


def example_4_pytorch_module():
    """
    Example 4: Using FusedQKVAttention as a PyTorch module
    
    This can be used as a drop-in replacement for standard attention layers.
    """
    print("=" * 80)
    print("Example 4: PyTorch Module Usage")
    print("=" * 80)
    
    # Configuration
    batch_size = 2
    seqlen = 512
    hidden_dim = 3584
    num_q_heads = 32
    num_kv_heads = 4
    head_dim = 128
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    # Create attention module
    attn_module = FusedQKVAttention(
        hidden_dim=hidden_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bias=True,
        causal=True,
        device=device,
        dtype=dtype
    )
    
    print(f"Created FusedQKVAttention module:")
    print(f"  Parameters: {sum(p.numel() for p in attn_module.parameters()):,}")
    print(f"  QKV weight shape: {attn_module.qkv_weight.shape}")
    print(f"  Output proj weight shape: {attn_module.o_proj.weight.shape}")
    
    # Create input
    hidden_states = torch.randn(batch_size, seqlen, hidden_dim, dtype=dtype, device=device)
    
    # Forward pass
    output = attn_module(hidden_states)
    print(f"\nInput: {hidden_states.shape}")
    print(f"Output: {output.shape}")  # [batch, seq, hidden_dim]
    
    print("\n✓ PyTorch module example completed successfully!\n")
    return output


def example_5_convert_from_model():
    """
    Example 5: Converting existing model layers to use fused QKV
    
    This shows how to replace standard attention layers with fused versions.
    """
    print("=" * 80)
    print("Example 5: Converting Existing Model Layers")
    print("=" * 80)
    
    # Simulate existing model layers (like Qwen3)
    hidden_dim = 3584
    num_q_heads = 32
    num_kv_heads = 4
    head_dim = 128
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    # Create "existing" layers
    q_proj = torch.nn.Linear(hidden_dim, num_q_heads * head_dim, dtype=dtype, device=device)
    k_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    v_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    o_proj = torch.nn.Linear(num_q_heads * head_dim, hidden_dim, dtype=dtype, device=device)
    
    print("Original layers:")
    print(f"  q_proj: {q_proj}")
    print(f"  k_proj: {k_proj}")
    print(f"  v_proj: {v_proj}")
    print(f"  o_proj: {o_proj}")
    
    # Convert to fused version
    fused_attn = FusedQKVAttention.from_separate_projections(
        q_proj, k_proj, v_proj, o_proj,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=True
    )
    
    print(f"\nConverted to FusedQKVAttention:")
    print(f"  Parameters: {sum(p.numel() for p in fused_attn.parameters()):,}")
    
    # Test that outputs match
    hidden_states = torch.randn(2, 128, hidden_dim, dtype=dtype, device=device)
    
    # Original (separate projections)
    with torch.no_grad():
        q = q_proj(hidden_states).view(2, 128, num_q_heads, head_dim).transpose(1, 2)
        k = k_proj(hidden_states).view(2, 128, num_kv_heads, head_dim).transpose(1, 2)
        v = v_proj(hidden_states).view(2, 128, num_kv_heads, head_dim).transpose(1, 2)
        q_t = q.transpose(1, 2).contiguous()
        k_t = k.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()
        attn_out_orig = flash_attn_func(q_t, k_t, v_t, causal=True)
        output_orig = o_proj(attn_out_orig.reshape(2, 128, -1))
    
    # Fused version
    with torch.no_grad():
        output_fused = fused_attn(hidden_states)
    
    # Compare
    diff = (output_orig - output_fused).abs().max().item()
    print(f"\nOutput comparison:")
    print(f"  Original output: {output_orig.shape}")
    print(f"  Fused output: {output_fused.shape}")
    print(f"  Max difference: {diff:.6f}")
    
    if diff < 1e-2:
        print("  ✓ Outputs match!")
    else:
        print("  ✗ Outputs differ (may need to check weight conversion)")
    
    print("\n✓ Conversion example completed successfully!\n")
    return fused_attn


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("QKV Fusion + FlashAttention Integration Examples")
    print("=" * 80 + "\n")
    
    # Run examples
    example_1_low_level()
    example_2_mid_level()
    example_3_high_level()
    example_4_pytorch_module()
    example_5_convert_from_model()
    
    print("=" * 80)
    print("All examples completed successfully! 🎉")
    print("=" * 80)


if __name__ == "__main__":
    main()

