#!/usr/bin/env python3
"""
Comprehensive benchmark comparing all QKV fusion approaches:
1. PyTorch baseline (3 separate nn.Linear)
2. Phase 2 optimized (cuBLAS GEMM + custom split/bias/transpose kernel)
3. Phase 3 lightweight (cuBLAS GEMM + simple bias + PyTorch split/transpose)

This will help identify which approach gives the best performance.
"""

import torch
import time
from typing import Tuple

try:
    from qkv_fusion import qkv_fused_forward_optimized, qkv_fused_forward_lightweight
    from qkv_fusion.weight_utils import prepare_fused_qkv_weights
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease recompile the extension first:")
    print("  cd /home/kchenbx/attention_optimization/qkv_fusion")
    print("  pip install -e . --force-reinstall --no-deps")
    exit(1)


def benchmark_pytorch_baseline(
    hidden_states: torch.Tensor,
    q_proj: torch.nn.Linear,
    k_proj: torch.nn.Linear,
    v_proj: torch.nn.Linear,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_iters: int = 100
) -> Tuple[float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Benchmark PyTorch baseline: 3 separate nn.Linear + reshape/transpose"""
    batch_size, seqlen, _ = hidden_states.shape
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            q = q_proj(hidden_states)
            k = k_proj(hidden_states)
            v = v_proj(hidden_states)
            q = q.view(batch_size, seqlen, num_q_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            q = q_proj(hidden_states)
            k = k_proj(hidden_states)
            v = v_proj(hidden_states)
            q = q.view(batch_size, seqlen, num_q_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
    torch.cuda.synchronize()
    elapsed_time = (time.time() - start) / num_iters * 1000  # ms
    
    return elapsed_time, (q, k, v)


def benchmark_optimized_kernel(
    hidden_states: torch.Tensor,
    fused_weight: torch.Tensor,
    fused_bias: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_iters: int = 100
) -> Tuple[float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Benchmark Phase 2: cuBLAS + custom split/bias/transpose kernel"""
    # Warmup
    for _ in range(10):
        q, k, v = qkv_fused_forward_optimized(
            hidden_states, fused_weight, fused_bias,
            num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(num_iters):
        q, k, v = qkv_fused_forward_optimized(
            hidden_states, fused_weight, fused_bias,
            num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
    torch.cuda.synchronize()
    elapsed_time = (time.time() - start) / num_iters * 1000  # ms
    
    return elapsed_time, (q, k, v)


def benchmark_lightweight_kernel(
    hidden_states: torch.Tensor,
    fused_weight: torch.Tensor,
    fused_bias: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_iters: int = 100
) -> Tuple[float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Benchmark Phase 3: cuBLAS + simple bias + PyTorch split/transpose"""
    # Warmup
    for _ in range(10):
        q, k, v = qkv_fused_forward_lightweight(
            hidden_states, fused_weight, fused_bias,
            num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(num_iters):
        q, k, v = qkv_fused_forward_lightweight(
            hidden_states, fused_weight, fused_bias,
            num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
    torch.cuda.synchronize()
    elapsed_time = (time.time() - start) / num_iters * 1000  # ms
    
    return elapsed_time, (q, k, v)


def verify_correctness(
    baseline_results: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    test_results: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    approach_name: str,
    threshold: float = 2e-2
) -> bool:
    """Verify that test results match baseline within threshold"""
    q_base, k_base, v_base = baseline_results
    q_test, k_test, v_test = test_results
    
    q_diff = (q_test - q_base).abs().max().item()
    k_diff = (k_test - k_base).abs().max().item()
    v_diff = (v_test - v_base).abs().max().item()
    
    print(f"\n{approach_name} correctness check:")
    print(f"  Q max diff: {q_diff:.6f}")
    print(f"  K max diff: {k_diff:.6f}")
    print(f"  V max diff: {v_diff:.6f}")
    
    if q_diff < threshold and k_diff < threshold and v_diff < threshold:
        print(f"  ✓ PASS")
        return True
    else:
        print(f"  ✗ FAIL (threshold: {threshold})")
        return False


def main():
    print("=" * 80)
    print("Comprehensive QKV Fusion Benchmark")
    print("=" * 80)
    
    # Configuration (Qwen3-30B-A3B-like)
    batch_size = 4
    seqlen = 512
    hidden_dim = 2048
    num_q_heads = 32
    num_kv_heads = 4
    head_dim = 128
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    num_iters = 100
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seqlen}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Q heads: {num_q_heads}, KV heads: {num_kv_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Total tokens: {batch_size * seqlen}")
    print(f"  Iterations: {num_iters}")
    
    # Create input
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seqlen, hidden_dim, dtype=dtype, device=device)
    
    # Create PyTorch Linear layers (baseline)
    q_proj = torch.nn.Linear(hidden_dim, num_q_heads * head_dim, dtype=dtype, device=device)
    k_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    v_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    
    # Prepare fused weights
    q_weight = q_proj.weight.t().contiguous()
    k_weight = k_proj.weight.t().contiguous()
    v_weight = v_proj.weight.t().contiguous()
    fused_weight, fused_bias = prepare_fused_qkv_weights(
        q_weight, k_weight, v_weight,
        q_proj.bias, k_proj.bias, v_proj.bias
    )
    
    print("\n" + "=" * 80)
    print("Benchmarking...")
    print("=" * 80)
    
    # Calculate FLOPs for performance reporting
    M = batch_size * seqlen
    N_total = (num_q_heads + 2 * num_kv_heads) * head_dim
    K = hidden_dim
    gemm_flops = 2 * M * N_total * K
    
    # Benchmark 1: PyTorch baseline
    print("\n[1] PyTorch Baseline (3 nn.Linear + reshape/transpose)")
    baseline_time, baseline_results = benchmark_pytorch_baseline(
        hidden_states, q_proj, k_proj, v_proj,
        num_q_heads, num_kv_heads, head_dim, num_iters
    )
    baseline_gflops = gemm_flops / (baseline_time * 1e6)
    print(f"    Time: {baseline_time:.3f} ms")
    print(f"    Performance: {baseline_gflops:.2f} GFLOPS")
    
    # Benchmark 2: Phase 2 optimized
    print("\n[2] Phase 2 Optimized (cuBLAS + custom split/bias/transpose)")
    try:
        optimized_time, optimized_results = benchmark_optimized_kernel(
            hidden_states, fused_weight, fused_bias,
            num_q_heads, num_kv_heads, head_dim, num_iters
        )
        optimized_gflops = gemm_flops / (optimized_time * 1e6)
        print(f"    Time: {optimized_time:.3f} ms")
        print(f"    Performance: {optimized_gflops:.2f} GFLOPS")
        print(f"    Speedup vs baseline: {baseline_time/optimized_time:.2f}x")
        
        # Verify correctness
        verify_correctness(baseline_results, optimized_results, "Phase 2 Optimized")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        optimized_time = float('inf')
    
    # Benchmark 3: Phase 3 lightweight
    print("\n[3] Phase 3 Lightweight (cuBLAS + bias + PyTorch split/transpose)")
    try:
        lightweight_time, lightweight_results = benchmark_lightweight_kernel(
            hidden_states, fused_weight, fused_bias,
            num_q_heads, num_kv_heads, head_dim, num_iters
        )
        lightweight_gflops = gemm_flops / (lightweight_time * 1e6)
        print(f"    Time: {lightweight_time:.3f} ms")
        print(f"    Performance: {lightweight_gflops:.2f} GFLOPS")
        print(f"    Speedup vs baseline: {baseline_time/lightweight_time:.2f}x")
        
        # Verify correctness
        verify_correctness(baseline_results, lightweight_results, "Phase 3 Lightweight")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        lightweight_time = float('inf')
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    results = [
        ("PyTorch Baseline", baseline_time, baseline_gflops, 1.0),
    ]
    
    if optimized_time != float('inf'):
        results.append(("Phase 2 Optimized", optimized_time, optimized_gflops, baseline_time/optimized_time))
    
    if lightweight_time != float('inf'):
        results.append(("Phase 3 Lightweight", lightweight_time, lightweight_gflops, baseline_time/lightweight_time))
    
    print(f"\n{'Approach':<30} {'Time (ms)':<12} {'GFLOPS':<12} {'Speedup':<10}")
    print("-" * 80)
    for name, time_ms, gflops, speedup in results:
        print(f"{name:<30} {time_ms:>10.3f}   {gflops:>10.2f}   {speedup:>8.2f}x")
    
    # Identify winner
    print("\n" + "=" * 80)
    if lightweight_time != float('inf') and optimized_time != float('inf'):
        if lightweight_time < optimized_time and lightweight_time < baseline_time:
            improvement = (optimized_time - lightweight_time) / optimized_time * 100
            print(f"✓ WINNER: Phase 3 Lightweight is {baseline_time/lightweight_time:.2f}x faster than baseline")
            print(f"  and {improvement:.1f}% faster than Phase 2 Optimized!")
            print(f"\n  Analysis: The custom split/bias/transpose kernel in Phase 2 was adding")
            print(f"  significant overhead. PyTorch's view/transpose operations are highly")
            print(f"  optimized and essentially free, making the lightweight approach superior.")
        elif optimized_time < lightweight_time and optimized_time < baseline_time:
            print(f"✓ WINNER: Phase 2 Optimized is {baseline_time/optimized_time:.2f}x faster than baseline")
        else:
            print(f"⚠ PyTorch baseline is still the fastest. Need to investigate bottlenecks.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

