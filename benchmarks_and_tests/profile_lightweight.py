#!/usr/bin/env python3
"""
Detailed profiling of the lightweight kernel to find bottlenecks.
"""

import torch
import time
from qkv_fusion import qkv_fused_forward_lightweight
from qkv_fusion.weight_utils import prepare_fused_qkv_weights

def profile_components():
    """Profile each component of the lightweight approach separately"""
    
    # Setup
    batch_size = 4
    seqlen = 512
    hidden_dim = 2048
    num_q_heads = 32
    num_kv_heads = 4
    head_dim = 128
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    num_iters = 100
    
    # Create inputs
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seqlen, hidden_dim, dtype=dtype, device=device)
    
    # Create weights
    q_proj = torch.nn.Linear(hidden_dim, num_q_heads * head_dim, dtype=dtype, device=device)
    k_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    v_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    
    q_weight = q_proj.weight.t().contiguous()
    k_weight = k_proj.weight.t().contiguous()
    v_weight = v_proj.weight.t().contiguous()
    fused_weight, fused_bias = prepare_fused_qkv_weights(
        q_weight, k_weight, v_weight,
        q_proj.bias, k_proj.bias, v_proj.bias
    )
    
    print("=" * 80)
    print("Component-by-Component Profiling")
    print("=" * 80)
    
    # 1. Just the CUDA kernel (GEMM + bias)
    print("\n[1] CUDA kernel only (GEMM + bias, no Python postprocessing)")
    import qkv_fusion_cuda
    
    # Warmup
    for _ in range(10):
        qkv_output = qkv_fusion_cuda.qkv_fused_forward_lightweight(
            hidden_states, fused_weight, fused_bias,
            num_q_heads, num_kv_heads, head_dim
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(num_iters):
        qkv_output = qkv_fusion_cuda.qkv_fused_forward_lightweight(
            hidden_states, fused_weight, fused_bias,
            num_q_heads, num_kv_heads, head_dim
        )
    torch.cuda.synchronize()
    cuda_time = (time.time() - start) / num_iters * 1000
    print(f"    Time: {cuda_time:.3f} ms")
    
    # 2. Full lightweight (CUDA + Python split/transpose)
    print("\n[2] Full lightweight (CUDA + Python split/transpose)")
    
    # Warmup
    for _ in range(10):
        q, k, v = qkv_fused_forward_lightweight(
            hidden_states, fused_weight, fused_bias,
            num_q_heads, num_kv_heads, head_dim
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(num_iters):
        q, k, v = qkv_fused_forward_lightweight(
            hidden_states, fused_weight, fused_bias,
            num_q_heads, num_kv_heads, head_dim
        )
    torch.cuda.synchronize()
    full_time = (time.time() - start) / num_iters * 1000
    print(f"    Time: {full_time:.3f} ms")
    
    # 3. Just PyTorch split/transpose on pre-computed output
    print("\n[3] PyTorch split/transpose only (on pre-computed tensor)")
    
    # Pre-compute output once
    qkv_output = qkv_fusion_cuda.qkv_fused_forward_lightweight(
        hidden_states, fused_weight, fused_bias,
        num_q_heads, num_kv_heads, head_dim
    )
    
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    
    # Warmup
    for _ in range(10):
        q = qkv_output[:, :, :q_dim]
        k = qkv_output[:, :, q_dim:q_dim + kv_dim]
        v = qkv_output[:, :, q_dim + kv_dim:]
        q = q.view(batch_size, seqlen, num_q_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(num_iters):
        q = qkv_output[:, :, :q_dim]
        k = qkv_output[:, :, q_dim:q_dim + kv_dim]
        v = qkv_output[:, :, q_dim + kv_dim:]
        q = q.view(batch_size, seqlen, num_q_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iters * 1000
    print(f"    Time: {pytorch_time:.3f} ms")
    
    # 4. Pure cuBLAS GEMM baseline
    print("\n[4] Pure cuBLAS GEMM (for comparison)")
    M = batch_size * seqlen
    K = hidden_dim
    N = (num_q_heads + 2 * num_kv_heads) * head_dim
    
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(10):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(num_iters):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    gemm_time = (time.time() - start) / num_iters * 1000
    print(f"    Time: {gemm_time:.3f} ms")
    
    # Analysis
    print("\n" + "=" * 80)
    print("Analysis")
    print("=" * 80)
    print(f"\nCUDA kernel (GEMM + bias):           {cuda_time:.3f} ms")
    print(f"Python split/transpose overhead:     {pytorch_time:.3f} ms")
    print(f"Expected total:                      {cuda_time + pytorch_time:.3f} ms")
    print(f"Actual total:                        {full_time:.3f} ms")
    print(f"Unexplained overhead:                {full_time - (cuda_time + pytorch_time):.3f} ms")
    print(f"\nComparison to pure GEMM:             {gemm_time:.3f} ms")
    print(f"CUDA kernel overhead vs pure GEMM:   {cuda_time - gemm_time:.3f} ms")
    
    if full_time - (cuda_time + pytorch_time) > 0.005:
        print("\n⚠ Significant unexplained overhead detected!")
        print("  Possible causes:")
        print("  - CUDA synchronization overhead")
        print("  - Kernel launch latency")
        print("  - Memory allocation/copy overhead")
    
    print("=" * 80)

if __name__ == "__main__":
    profile_components()

