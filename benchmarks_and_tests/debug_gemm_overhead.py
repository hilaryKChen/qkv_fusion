#!/usr/bin/env python3
"""
Debug: Why is the CUDA GEMM slower than pure PyTorch matmul?
"""

import torch
import time

def benchmark(name, func, num_iters=100):
    # Warmup
    for _ in range(10):
        func()
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(num_iters):
        func()
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / num_iters * 1000
    print(f"{name:50s} {elapsed:.3f} ms")
    return elapsed

def main():
    batch_size = 4
    seqlen = 512
    hidden_dim = 2048
    num_q_heads = 32
    num_kv_heads = 4
    head_dim = 128
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    M = batch_size * seqlen
    K = hidden_dim
    N = (num_q_heads + 2 * num_kv_heads) * head_dim
    
    print("=" * 80)
    print("GEMM Overhead Investigation")
    print("=" * 80)
    print(f"\nMatrix dimensions: [{M}, {K}] @ [{K}, {N}] = [{M}, {N}]")
    print()
    
    # Create inputs
    torch.manual_seed(42)
    hidden_states_3d = torch.randn(batch_size, seqlen, hidden_dim, dtype=dtype, device=device)
    hidden_states_2d = hidden_states_3d.view(M, K)
    
    # Weights
    q_proj = torch.nn.Linear(hidden_dim, num_q_heads * head_dim, dtype=dtype, device=device)
    k_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    v_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
    
    from qkv_fusion.weight_utils import prepare_fused_qkv_weights
    q_weight = q_proj.weight.t().contiguous()
    k_weight = k_proj.weight.t().contiguous()
    v_weight = v_proj.weight.t().contiguous()
    fused_weight, fused_bias = prepare_fused_qkv_weights(
        q_weight, k_weight, v_weight,
        q_proj.bias, k_proj.bias, v_proj.bias
    )
    
    # Test 1: Pure torch.matmul
    A = hidden_states_2d
    B = fused_weight
    t1 = benchmark("[1] torch.matmul (2D tensors)", 
                   lambda: torch.matmul(A, B))
    
    # Test 2: F.linear without bias
    t2 = benchmark("[2] F.linear without bias (3D input)",
                   lambda: torch.nn.functional.linear(hidden_states_3d, fused_weight.t()))
    
    # Test 3: F.linear with bias
    t3 = benchmark("[3] F.linear with bias (3D input)",
                   lambda: torch.nn.functional.linear(hidden_states_3d, fused_weight.t(), fused_bias))
    
    # Test 4: Your CUDA kernel without bias
    import qkv_fusion_cuda
    t4 = benchmark("[4] Your CUDA kernel (GEMM only, no bias)",
                   lambda: qkv_fusion_cuda.qkv_fused_forward_lightweight(
                       hidden_states_3d, fused_weight, None,
                       num_q_heads, num_kv_heads, head_dim))
    
    # Test 5: Your CUDA kernel with bias
    t5 = benchmark("[5] Your CUDA kernel (GEMM + bias in CUDA)",
                   lambda: qkv_fusion_cuda.qkv_fused_forward_lightweight(
                       hidden_states_3d, fused_weight, fused_bias,
                       num_q_heads, num_kv_heads, head_dim))
    
    # Test 6: Full lightweight Python wrapper
    from qkv_fusion import qkv_fused_forward_lightweight
    t6 = benchmark("[6] Full lightweight (GEMM + PyTorch bias/split)",
                   lambda: qkv_fused_forward_lightweight(
                       hidden_states_3d, fused_weight, fused_bias,
                       num_q_heads, num_kv_heads, head_dim))
    
    print("\n" + "=" * 80)
    print("Analysis")
    print("=" * 80)
    print(f"\nPure torch.matmul:                    {t1:.3f} ms (baseline)")
    print(f"F.linear without bias:                {t2:.3f} ms (+{t2-t1:.3f} ms)")
    print(f"F.linear with bias:                   {t3:.3f} ms (+{t3-t1:.3f} ms)")
    print(f"Your CUDA (no bias):                  {t4:.3f} ms (+{t4-t1:.3f} ms)")
    print(f"Your CUDA (with bias):                {t5:.3f} ms (+{t5-t1:.3f} ms)")
    print(f"Full lightweight:                     {t6:.3f} ms (+{t6-t1:.3f} ms)")
    
    print(f"\nBias add overhead (CUDA):             {t5-t4:.3f} ms")
    print(f"Bias add overhead (PyTorch):          {t3-t2:.3f} ms")
    
    if t4 > t1 + 0.010:
        overhead = t4 - t1
        print(f"\n⚠ Your CUDA kernel has {overhead:.3f} ms unexplained overhead!")
        print("  Possible causes:")
        print("  - Tensor reshape overhead (3D -> 2D)")
        print("  - cuBLAS setup/configuration")
        print("  - Memory layout (row-major vs column-major)")
        print("  - Workspace allocation")
    
    print("=" * 80)

if __name__ == "__main__":
    main()

