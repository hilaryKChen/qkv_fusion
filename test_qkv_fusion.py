#!/usr/bin/env python3
"""
Test script for QKV Fusion kernel
Tests correctness against PyTorch baseline
"""

import torch
import time
from qkv_fusion import qkv_fused_forward

def test_qkv_fusion_correctness():
    """Test that fused kernel produces same results as separate linear layers"""
    print("=" * 80)
    print("Testing QKV Fusion Correctness")
    print("=" * 80)
    
    # Qwen3-7B configuration
    batch_size = 2
    seqlen = 128
    hidden_dim = 3584
    num_q_heads = 32
    num_kv_heads = 4
    head_dim = 128
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    # Create random inputs
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seqlen, hidden_dim, dtype=dtype, device=device)
    
    # Create weight matrices (transposed for PyTorch linear layers)
    q_weight = torch.randn(num_q_heads * head_dim, hidden_dim, dtype=dtype, device=device)
    k_weight = torch.randn(num_kv_heads * head_dim, hidden_dim, dtype=dtype, device=device)
    v_weight = torch.randn(num_kv_heads * head_dim, hidden_dim, dtype=dtype, device=device)
    
    # Optional biases
    q_bias = torch.randn(num_q_heads * head_dim, dtype=dtype, device=device)
    k_bias = torch.randn(num_kv_heads * head_dim, dtype=dtype, device=device)
    v_bias = torch.randn(num_kv_heads * head_dim, dtype=dtype, device=device)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Q weight shape: {q_weight.shape}")
    print(f"K weight shape: {k_weight.shape}")
    print(f"V weight shape: {v_weight.shape}")
    
    # Baseline: PyTorch linear layers
    print("\n[1] Running PyTorch baseline...")
    q_baseline = torch.nn.functional.linear(hidden_states, q_weight, q_bias)
    k_baseline = torch.nn.functional.linear(hidden_states, k_weight, k_bias)
    v_baseline = torch.nn.functional.linear(hidden_states, v_weight, v_bias)
    
    # Reshape to [batch, num_heads, seqlen, head_dim]
    q_baseline = q_baseline.view(batch_size, seqlen, num_q_heads, head_dim).transpose(1, 2)
    k_baseline = k_baseline.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
    v_baseline = v_baseline.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
    
    print(f"Q baseline shape: {q_baseline.shape}")
    print(f"K baseline shape: {k_baseline.shape}")
    print(f"V baseline shape: {v_baseline.shape}")
    
    # Fused kernel (need to transpose weights back)
    print("\n[2] Running fused QKV kernel...")
    q_weight_t = q_weight.t().contiguous()
    k_weight_t = k_weight.t().contiguous()
    v_weight_t = v_weight.t().contiguous()
    
    q_fused, k_fused, v_fused = qkv_fused_forward(
        hidden_states,
        q_weight_t,
        k_weight_t,
        v_weight_t,
        q_bias,
        k_bias,
        v_bias,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )
    
    print(f"Q fused shape: {q_fused.shape}")
    print(f"K fused shape: {k_fused.shape}")
    print(f"V fused shape: {v_fused.shape}")
    
    # Compare results
    print("\n[3] Comparing results...")
    q_diff = (q_fused - q_baseline).abs().max().item()
    k_diff = (k_fused - k_baseline).abs().max().item()
    v_diff = (v_fused - v_baseline).abs().max().item()
    
    q_rel_err = ((q_fused - q_baseline).abs() / (q_baseline.abs() + 1e-5)).mean().item()
    k_rel_err = ((k_fused - k_baseline).abs() / (k_baseline.abs() + 1e-5)).mean().item()
    v_rel_err = ((v_fused - v_baseline).abs() / (v_baseline.abs() + 1e-5)).mean().item()
    
    print(f"Q max absolute diff: {q_diff:.6f}")
    print(f"K max absolute diff: {k_diff:.6f}")
    print(f"V max absolute diff: {v_diff:.6f}")
    print(f"Q mean relative error: {q_rel_err:.6f}")
    print(f"K mean relative error: {k_rel_err:.6f}")
    print(f"V mean relative error: {v_rel_err:.6f}")
    
    # Check if results are close (allow some FP16 numerical error)
    threshold = 1e-2  # FP16 has limited precision
    if q_diff < threshold and k_diff < threshold and v_diff < threshold:
        print("\n✓ PASS: Fused kernel matches baseline!")
        return True
    else:
        print("\n✗ FAIL: Results differ too much!")
        return False

def benchmark_qkv_fusion():
    """Benchmark fused kernel vs baseline"""
    print("\n" + "=" * 80)
    print("Benchmarking QKV Fusion Performance")
    print("=" * 80)
    
    batch_size = 4
    seqlen = 512
    hidden_dim = 3584
    num_q_heads = 32
    num_kv_heads = 4
    head_dim = 128
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    # Create inputs
    hidden_states = torch.randn(batch_size, seqlen, hidden_dim, dtype=dtype, device=device)
    q_weight = torch.randn(num_q_heads * head_dim, hidden_dim, dtype=dtype, device=device)
    k_weight = torch.randn(num_kv_heads * head_dim, hidden_dim, dtype=dtype, device=device)
    v_weight = torch.randn(num_kv_heads * head_dim, hidden_dim, dtype=dtype, device=device)
    
    q_weight_t = q_weight.t().contiguous()
    k_weight_t = k_weight.t().contiguous()
    v_weight_t = v_weight.t().contiguous()
    
    # Warmup
    for _ in range(10):
        _ = torch.nn.functional.linear(hidden_states, q_weight)
        q_fused, k_fused, v_fused = qkv_fused_forward(
            hidden_states, q_weight_t, k_weight_t, v_weight_t,
            num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
    torch.cuda.synchronize()
    
    # Benchmark baseline
    num_iters = 100
    start = time.time()
    for _ in range(num_iters):
        q = torch.nn.functional.linear(hidden_states, q_weight)
        k = torch.nn.functional.linear(hidden_states, k_weight)
        v = torch.nn.functional.linear(hidden_states, v_weight)
    torch.cuda.synchronize()
    baseline_time = (time.time() - start) / num_iters * 1000  # ms
    
    # Benchmark fused
    start = time.time()
    for _ in range(num_iters):
        q_fused, k_fused, v_fused = qkv_fused_forward(
            hidden_states, q_weight_t, k_weight_t, v_weight_t,
            num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
    torch.cuda.synchronize()
    fused_time = (time.time() - start) / num_iters * 1000  # ms
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seqlen}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Q heads: {num_q_heads}, KV heads: {num_kv_heads}")
    print(f"\nResults:")
    print(f"  Baseline (3 separate linear): {baseline_time:.3f} ms")
    print(f"  Fused QKV kernel:             {fused_time:.3f} ms")
    print(f"  Speedup:                      {baseline_time/fused_time:.2f}x")

if __name__ == "__main__":
    # Run tests
    success = test_qkv_fusion_correctness()
    
    if success:
        benchmark_qkv_fusion()
    else:
        print("\nSkipping benchmark due to correctness test failure.")

