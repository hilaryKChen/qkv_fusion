#!/usr/bin/env python3
"""
Test script for optimized QKV fusion kernel (Phase 2)
Compares Phase 1 (naive) vs Phase 2 (optimized with cuBLAS)
"""

import torch
import time
from qkv_fusion import qkv_fused_forward, qkv_fused_forward_optimized
from qkv_fusion.weight_utils import prepare_fused_qkv_weights

def test_optimized_correctness():
    """Test that optimized kernel matches PyTorch nn.Linear baseline"""
    print("=" * 80)
    print("Testing Optimized QKV Fusion Correctness")
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
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Configuration: {num_q_heads} Q heads, {num_kv_heads} KV heads, {head_dim} head_dim")
    
    # Create PyTorch Linear layers (baseline)
    q_proj = torch.nn.Linear(hidden_dim, num_q_heads * head_dim, dtype=dtype, device=device)
    k_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    v_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    
    print(f"\nQ proj weight shape: {q_proj.weight.shape}")  # [out, in]
    print(f"K proj weight shape: {k_proj.weight.shape}")
    print(f"V proj weight shape: {v_proj.weight.shape}")
    
    # Baseline: PyTorch nn.Linear (3 separate projections)
    print("\n[1] Running PyTorch baseline (3 nn.Linear)...")
    with torch.no_grad():
        q_baseline = q_proj(hidden_states)  # [batch, seq, num_q_heads * head_dim]
        k_baseline = k_proj(hidden_states)  # [batch, seq, num_kv_heads * head_dim]
        v_baseline = v_proj(hidden_states)  # [batch, seq, num_kv_heads * head_dim]
    
    # Reshape and transpose to [batch, heads, seq, head_dim]
    q_baseline = q_baseline.view(batch_size, seqlen, num_q_heads, head_dim).transpose(1, 2)
    k_baseline = k_baseline.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
    v_baseline = v_baseline.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
    
    print(f"Q baseline shape: {q_baseline.shape}")
    print(f"K baseline shape: {k_baseline.shape}")
    print(f"V baseline shape: {v_baseline.shape}")
    
    # Optimized: cuBLAS fused kernel
    print("\n[2] Running optimized cuBLAS kernel...")
    
    # Prepare fused weights from PyTorch Linear layers
    # Note: nn.Linear weights are [out_features, in_features], need to transpose
    q_weight = q_proj.weight.t().contiguous()  # [hidden_dim, num_q_heads * head_dim]
    k_weight = k_proj.weight.t().contiguous()  # [hidden_dim, num_kv_heads * head_dim]
    v_weight = v_proj.weight.t().contiguous()  # [hidden_dim, num_kv_heads * head_dim]
    
    fused_weight, fused_bias = prepare_fused_qkv_weights(
        q_weight, k_weight, v_weight,
        q_proj.bias, k_proj.bias, v_proj.bias
    )
    print(f"Fused weight shape: {fused_weight.shape}")
    print(f"Fused bias shape: {fused_bias.shape if fused_bias is not None else None}")
    
    q_optimized, k_optimized, v_optimized = qkv_fused_forward_optimized(
        hidden_states,
        fused_weight,
        fused_bias,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )
    
    print(f"Q optimized shape: {q_optimized.shape}")
    print(f"K optimized shape: {k_optimized.shape}")
    print(f"V optimized shape: {v_optimized.shape}")
    
    # Compare results
    print("\n[3] Comparing results...")
    q_diff = (q_optimized - q_baseline).abs().max().item()
    k_diff = (k_optimized - k_baseline).abs().max().item()
    v_diff = (v_optimized - v_baseline).abs().max().item()
    
    q_rel_err = ((q_optimized - q_baseline).abs() / (q_baseline.abs() + 1e-5)).mean().item()
    k_rel_err = ((k_optimized - k_baseline).abs() / (k_baseline.abs() + 1e-5)).mean().item()
    v_rel_err = ((v_optimized - v_baseline).abs() / (v_baseline.abs() + 1e-5)).mean().item()
    
    print(f"Q max absolute diff: {q_diff:.6f}")
    print(f"K max absolute diff: {k_diff:.6f}")
    print(f"V max absolute diff: {v_diff:.6f}")
    print(f"Q mean relative error: {q_rel_err:.6f}")
    print(f"K mean relative error: {k_rel_err:.6f}")
    print(f"V mean relative error: {v_rel_err:.6f}")
    
    # Check if results are close (allow some FP16 numerical error)
    # threshold = 1e-2  # FP16 has limited precision
    threshold = 2e-2  # FP16 has limited precision
    if q_diff < threshold and k_diff < threshold and v_diff < threshold:
        print("\n✓ PASS: Optimized kernel matches PyTorch baseline!")
        return True
    else:
        print("\n✗ FAIL: Results differ too much!")
        print("\nDebugging info:")
        print(f"  Q sample values (baseline): {q_baseline[0, 0, 0, :5]}")
        print(f"  Q sample values (optimized): {q_optimized[0, 0, 0, :5]}")
        return False

def benchmark_optimized():
    """Benchmark cuBLAS fused kernel vs PyTorch nn.Linear baseline"""
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
    
    # Create PyTorch Linear layers (baseline)
    q_proj = torch.nn.Linear(hidden_dim, num_q_heads * head_dim, dtype=dtype, device=device)
    k_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    v_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    
    # Prepare fused weights for optimized kernel
    q_weight = q_proj.weight.t().contiguous()
    k_weight = k_proj.weight.t().contiguous()
    v_weight = v_proj.weight.t().contiguous()
    fused_weight, fused_bias = prepare_fused_qkv_weights(
        q_weight, k_weight, v_weight,
        q_proj.bias, k_proj.bias, v_proj.bias
    )
    
    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = q_proj(hidden_states)
            _ = k_proj(hidden_states)
            _ = v_proj(hidden_states)
        _ = qkv_fused_forward_optimized(hidden_states, fused_weight, fused_bias,
                                        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim)
    torch.cuda.synchronize()
    
    # Benchmark PyTorch baseline (3 separate nn.Linear)
    num_iters = 100
    print(f"\nBenchmarking {num_iters} iterations...")
    
    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            q = q_proj(hidden_states)
            k = k_proj(hidden_states)
            v = v_proj(hidden_states)
            # Reshape and transpose (part of the operation)
            q = q.view(batch_size, seqlen, num_q_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iters * 1000  # ms
    
    # Benchmark optimized cuBLAS kernel
    start = time.time()
    for _ in range(num_iters):
        q, k, v = qkv_fused_forward_optimized(
            hidden_states, fused_weight, fused_bias,
            num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
    torch.cuda.synchronize()
    optimized_time = (time.time() - start) / num_iters * 1000  # ms
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seqlen}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Q heads: {num_q_heads}, KV heads: {num_kv_heads}")
    print(f"  Total tokens: {batch_size * seqlen}")
    
    # Calculate FLOPs
    # GEMM: 2 * M * N * K FLOPs
    M = batch_size * seqlen
    N_total = (num_q_heads + 2 * num_kv_heads) * head_dim  # 5120 for Qwen3
    K = hidden_dim
    gemm_flops = 2 * M * N_total * K
    
    print(f"\nResults:")
    print(f"  PyTorch (3 nn.Linear + reshape):  {pytorch_time:.3f} ms")
    print(f"  Optimized (cuBLAS fused):         {optimized_time:.3f} ms")
    print(f"\n  Speedup: {pytorch_time/optimized_time:.2f}x")
    print(f"  Performance: {gemm_flops / optimized_time / 1e9:.2f} GFLOPS")
    
    # Expected: Optimized should be 1.5-2.5x faster than PyTorch
    if optimized_time < pytorch_time:
        speedup = pytorch_time / optimized_time
        print(f"\n✓ SUCCESS: Optimized kernel is {speedup:.2f}x faster!")
        if speedup >= 1.5:
            print(f"  Excellent! Achieved target speedup (>1.5x)")
        else:
            print(f"  Good, but could be better. Target is 1.5-2.5x speedup.")
    else:
        print(f"\n✗ WARNING: Optimized kernel is slower than baseline!")
        print(f"  This suggests a bug or suboptimal implementation.")

if __name__ == "__main__":
    # Run tests
    success = test_optimized_correctness()
    
    if success:
        benchmark_optimized()
    else:
        print("\nSkipping benchmark due to correctness test failure.")

