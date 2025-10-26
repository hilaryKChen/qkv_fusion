#!/usr/bin/env python3
"""
Profile individual kernels to find the bottleneck
"""

import torch
import time

# Test just cuBLAS GEMM performance
def benchmark_cublas_gemm():
    batch_size = 4
    seqlen = 512
    M = batch_size * seqlen  # 2048
    K = 3584
    N = 5120
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(10):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Benchmark
    num_iters = 100
    start = time.time()
    for _ in range(num_iters):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    gemm_time = (time.time() - start) / num_iters * 1000
    
    flops = 2 * M * N * K
    gflops = flops / (gemm_time * 1e6)
    
    print(f"Pure cuBLAS GEMM [{M}, {K}] @ [{K}, {N}]:")
    print(f"  Time: {gemm_time:.3f} ms")
    print(f"  Performance: {gflops:.2f} GFLOPS")
    print()

# Test PyTorch Linear (includes bias and optimizations)
def benchmark_pytorch_linear():
    batch_size = 4
    seqlen = 512
    hidden_dim = 3584
    out_dim = 5120
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    x = torch.randn(batch_size, seqlen, hidden_dim, dtype=dtype, device=device)
    linear = torch.nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(10):
        y = linear(x)
    torch.cuda.synchronize()
    
    # Benchmark
    num_iters = 100
    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            y = linear(x)
    torch.cuda.synchronize()
    linear_time = (time.time() - start) / num_iters * 1000
    
    M = batch_size * seqlen
    K = hidden_dim
    N = out_dim
    flops = 2 * M * N * K
    gflops = flops / (linear_time * 1e6)
    
    print(f"PyTorch nn.Linear [{batch_size}, {seqlen}, {hidden_dim}] -> [{batch_size}, {seqlen}, {out_dim}]:")
    print(f"  Time: {linear_time:.3f} ms")
    print(f"  Performance: {gflops:.2f} GFLOPS")
    print()

# Test 3 separate smaller GEMMs (Q, K, V)
def benchmark_three_linears():
    batch_size = 4
    seqlen = 512
    hidden_dim = 3584
    q_dim = 4096
    kv_dim = 512
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    x = torch.randn(batch_size, seqlen, hidden_dim, dtype=dtype, device=device)
    q_proj = torch.nn.Linear(hidden_dim, q_dim, dtype=dtype, device=device)
    k_proj = torch.nn.Linear(hidden_dim, kv_dim, dtype=dtype, device=device)
    v_proj = torch.nn.Linear(hidden_dim, kv_dim, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            q = q_proj(x)
            k = k_proj(x)
            v = v_proj(x)
    torch.cuda.synchronize()
    
    # Benchmark
    num_iters = 100
    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            q = q_proj(x)
            k = k_proj(x)
            v = v_proj(x)
    torch.cuda.synchronize()
    three_linear_time = (time.time() - start) / num_iters * 1000
    
    M = batch_size * seqlen
    K = hidden_dim
    total_flops = 2 * M * K * (q_dim + kv_dim + kv_dim)
    gflops = total_flops / (three_linear_time * 1e6)
    
    print(f"3 separate nn.Linear (Q={q_dim}, K={kv_dim}, V={kv_dim}):")
    print(f"  Time: {three_linear_time:.3f} ms")
    print(f"  Performance: {gflops:.2f} GFLOPS")
    print()

if __name__ == "__main__":
    print("=" * 80)
    print("Kernel Performance Profiling")
    print("=" * 80)
    print()
    
    benchmark_cublas_gemm()
    benchmark_pytorch_linear()
    benchmark_three_linears()
    
    print("=" * 80)
    print("Analysis:")
    print("If pure cuBLAS GEMM is fast but your fused kernel is slow,")
    print("the bottleneck is in the split/transpose kernel or memory layout.")
    print("=" * 80)

