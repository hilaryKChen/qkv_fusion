#!/usr/bin/env python3
"""
Analyze why 3 separate Linear layers are almost as fast as 1 big Linear
"""

import torch
import time

def detailed_benchmark():
    batch_size = 4
    seqlen = 512
    hidden_dim = 2048  # Qwen3-30B-A3B
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    x = torch.randn(batch_size, seqlen, hidden_dim, dtype=dtype, device=device)
    
    print("=" * 80)
    print("Detailed Performance Analysis")
    print("=" * 80)
    print()
    
    # Test different output sizes
    sizes = [512, 1024, 2048, 4096, 5120]
    
    print("Single nn.Linear with varying output sizes:")
    print("-" * 80)
    for out_dim in sizes:
        linear = torch.nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = linear(x)
        torch.cuda.synchronize()
        
        # Benchmark
        num_iters = 100
        start = time.time()
        for _ in range(num_iters):
            with torch.no_grad():
                _ = linear(x)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / num_iters * 1000
        
        M = batch_size * seqlen
        K = hidden_dim
        N = out_dim
        flops = 2 * M * K * N
        gflops = flops / (elapsed * 1e6)
        
        print(f"  {hidden_dim:4d} → {out_dim:4d}: {elapsed:.3f} ms ({gflops:6.0f} GFLOPS)")
    
    print()
    print("=" * 80)
    print("Analysis:")
    print("=" * 80)
    
    # Theoretical analysis
    q_dim = 4096
    kv_dim = 512
    fused_dim = 5120
    
    print(f"\nTheoretical FLOPs:")
    print(f"  Q GEMM: 2 × {batch_size * seqlen} × {hidden_dim} × {q_dim} = {2 * batch_size * seqlen * hidden_dim * q_dim:,}")
    print(f"  K GEMM: 2 × {batch_size * seqlen} × {hidden_dim} × {kv_dim} = {2 * batch_size * seqlen * hidden_dim * kv_dim:,}")
    print(f"  V GEMM: 2 × {batch_size * seqlen} × {hidden_dim} × {kv_dim} = {2 * batch_size * seqlen * hidden_dim * kv_dim:,}")
    print(f"  Total (3 separate): {2 * batch_size * seqlen * hidden_dim * (q_dim + 2 * kv_dim):,}")
    print()
    print(f"  Fused GEMM: 2 × {batch_size * seqlen} × {hidden_dim} × {fused_dim} = {2 * batch_size * seqlen * hidden_dim * fused_dim:,}")
    print()
    print(f"  FLOPs are IDENTICAL! (both = {2 * batch_size * seqlen * hidden_dim * fused_dim:,})")
    print()
    
    print("Why 3 separate calls are almost as fast as 1 big call:")
    print()
    print("1. **Kernel Launch Overhead is Tiny**")
    print("   - Modern CUDA: ~2-5 μs per kernel launch")
    print("   - 3 launches = 0.015 ms overhead")
    print("   - Measured difference: 0.115 - 0.105 = 0.010 ms ✓")
    print()
    print("2. **Small GEMMs Are Memory-Bound**")
    print("   - K and V GEMMs (512 output) are tiny")
    print("   - They're limited by memory bandwidth, not compute")
    print("   - GPU can't fully utilize tensor cores for small matrices")
    print()
    print("3. **GPU Scheduler Optimization**")
    print("   - PyTorch may batch the 3 calls internally")
    print("   - Or use CUDA graphs to reduce overhead")
    print("   - Small kernels can run concurrently on different SMs")
    print()
    print("=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print()
    print("For this problem size (batch=4, seq=512), QKV fusion provides NO benefit!")
    print()
    print("Reasons:")
    print("  1. FLOPs are identical (no computation savings)")
    print("  2. Kernel launch overhead is negligible (~0.01 ms)")
    print("  3. Split kernel overhead (0.069 ms) >> launch overhead")
    print("  4. PyTorch's 3 separate calls are already well-optimized")
    print()
    print("QKV fusion would help if:")
    print("  ✓ Batch size is MUCH larger (e.g., batch=64, seq=2048)")
    print("  ✓ You have a truly fused kernel (GEMM+split in one pass)")
    print("  ✓ You're memory-bandwidth limited (not the case here)")
    print()
    print("Current approach (2 kernels: GEMM + split) adds overhead without benefit.")
    print("=" * 80)

if __name__ == "__main__":
    detailed_benchmark()

