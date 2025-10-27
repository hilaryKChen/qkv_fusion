#!/usr/bin/env python3
"""
Find why the CUDA GEMM wrapper is slower than pure torch.matmul
"""

import torch
import time
import qkv_fusion_cuda

batch_size = 4
seqlen = 512
hidden_dim = 2048
num_q_heads = 32
num_kv_heads = 4
head_dim = 128

M = batch_size * seqlen  # 2048
K = hidden_dim           # 2048
N = (num_q_heads + 2 * num_kv_heads) * head_dim  # 5120

device = torch.device("cuda:0")
dtype = torch.float16

print("=" * 80)
print("CUDA GEMM Wrapper Overhead Investigation")
print("=" * 80)
print(f"\nMatrix dimensions: [{M}, {K}] @ [{K}, {N}] = [{M}, {N}]")

# Create inputs
torch.manual_seed(42)
hidden_states_3d = torch.randn(batch_size, seqlen, hidden_dim, dtype=dtype, device=device)
hidden_states_2d = hidden_states_3d.view(M, K)
weight_2d = torch.randn(K, N, dtype=dtype, device=device)

# For CUDA kernel
q_proj = torch.nn.Linear(hidden_dim, num_q_heads * head_dim, dtype=dtype, device=device)
k_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)
v_proj = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim, dtype=dtype, device=device)

from qkv_fusion.weight_utils import prepare_fused_qkv_weights
q_weight = q_proj.weight.t().contiguous()
k_weight = k_proj.weight.t().contiguous()
v_weight = v_proj.weight.t().contiguous()
fused_weight, _ = prepare_fused_qkv_weights(
    q_weight, k_weight, v_weight,
    None, None, None
)

def benchmark(name, func, warmup=10, iters=100):
    for _ in range(warmup):
        result = func()
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iters):
        result = func()
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / iters * 1000
    print(f"  {name:50s} {elapsed:.3f} ms")
    return elapsed

print("\n[1] Baseline GEMM operations:")
t1 = benchmark("[Baseline] torch.matmul (2D input)", 
               lambda: torch.matmul(hidden_states_2d, weight_2d))

t2 = benchmark("[Baseline] torch.matmul (3D input reshaped inline)",
               lambda: torch.matmul(hidden_states_3d.view(M, K), weight_2d))

print("\n[2] Your CUDA kernel:")
t3 = benchmark("[Your kernel] qkv_fused_forward_lightweight",
               lambda: qkv_fusion_cuda.qkv_fused_forward_lightweight(
                   hidden_states_3d, fused_weight, None,
                   num_q_heads, num_kv_heads, head_dim))

print("\n[3] Testing potential overhead sources:")

# Test if pre-allocating output helps
output_buf = torch.empty(batch_size, seqlen, N, dtype=dtype, device=device)
def test_with_preallocated():
    result = qkv_fusion_cuda.qkv_fused_forward_lightweight(
        hidden_states_3d, fused_weight, None,
        num_q_heads, num_kv_heads, head_dim)
    return result

t4 = benchmark("[Your kernel] With pre-allocated buffer check",
               test_with_preallocated)

# Test if input shape matters
hidden_states_2d_explicit = hidden_states_3d.reshape(M, K).contiguous()
t5 = benchmark("[Test] torch.matmul (3D->2D explicit reshape)",
               lambda: torch.matmul(hidden_states_2d_explicit, weight_2d))

print("\n" + "=" * 80)
print("Analysis")
print("=" * 80)
print(f"\nBaseline torch.matmul (2D):           {t1:.3f} ms")
print(f"Baseline torch.matmul (3D reshaped):  {t2:.3f} ms")
print(f"Your CUDA kernel:                     {t3:.3f} ms")
print(f"\nOverhead breakdown:")
print(f"  3D reshape overhead:                {t2 - t1:.3f} ms")
print(f"  Your kernel overhead:               {t3 - t1:.3f} ms")
print(f"  Unexplained overhead:               {t3 - t2:.3f} ms")

if t3 - t2 > 0.010:
    print(f"\n⚠ PROBLEM: Kernel has {t3-t2:.3f} ms unexplained overhead!")
    print(f"\nPossible causes:")
    print(f"  1. Output tensor allocation in C++ (torch::empty)")
    print(f"  2. Tensor view/reshape in C++ return path")
    print(f"  3. cuBLAS handle/stream initialization")
    print(f"  4. Python-C++ call overhead")
    print(f"  5. Memory layout mismatch")
    
    print(f"\nTo fix:")
    print(f"  - Check C++ binding for unnecessary allocations")
    print(f"  - Ensure input/output tensors are contiguous")
    print(f"  - Profile with torch.profiler to see exact overhead")

print("=" * 80)

