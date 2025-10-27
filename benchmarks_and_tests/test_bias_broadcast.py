#!/usr/bin/env python3
"""
Test different ways to add bias to see which is fastest
"""

import torch
import time

batch_size = 4
seqlen = 512
qkv_dim = 5120

device = torch.device("cuda:0")
dtype = torch.float16

# Create tensors
qkv_output = torch.randn(batch_size, seqlen, qkv_dim, dtype=dtype, device=device)
bias = torch.randn(qkv_dim, dtype=dtype, device=device)

print("=" * 80)
print("Testing different bias add methods")
print("=" * 80)
print(f"\nTensor shapes:")
print(f"  qkv_output: {qkv_output.shape}")
print(f"  bias: {bias.shape}")

def benchmark(name, func, warmup=10, iters=100):
    for _ in range(warmup):
        result = func()
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iters):
        result = func()
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / iters * 1000
    print(f"  {name:40s} {elapsed:.3f} ms")
    return elapsed

print(f"\n[1] Different bias addition methods:")
t1 = benchmark("qkv_output + bias", 
               lambda: qkv_output + bias)

t2 = benchmark("qkv_output.add(bias)", 
               lambda: qkv_output.add(bias))

t3 = benchmark("torch.add(qkv_output, bias)",
               lambda: torch.add(qkv_output, bias))

# Try unsqueezing bias
bias_unsqueezed = bias.unsqueeze(0).unsqueeze(0)  # [1, 1, 5120]
t4 = benchmark("qkv_output + bias.unsqueeze(0).unsqueeze(0)",
               lambda: qkv_output + bias_unsqueezed)

# Try different memory ordering
qkv_flat = qkv_output.view(-1, qkv_dim)  # [2048, 5120]
t5 = benchmark("qkv_flat + bias (2D view)",
               lambda: qkv_flat + bias)

# Compare to F.linear which includes bias
weight = torch.randn(qkv_dim, 2048, dtype=dtype, device=device)
hidden = torch.randn(batch_size, seqlen, 2048, dtype=dtype, device=device)
t6 = benchmark("F.linear with bias (for reference)",
               lambda: torch.nn.functional.linear(hidden, weight, bias))

print(f"\n[2] Analysis:")
print(f"  Baseline (qkv_output + bias):  {t1:.3f} ms")
print(f"  2D view approach:              {t5:.3f} ms")
print(f"  F.linear with bias:            {t6:.3f} ms")
print(f"  F.linear bias overhead:        {t6 - 0.063:.3f} ms (should be ~0.003ms)")

if t5 < t1:
    print(f"\n✓ 2D view is faster! Use qkv_output.view(-1, qkv_dim) + bias")
    print(f"  Speedup: {t1/t5:.2f}x")
else:
    print(f"\n⚠ No improvement from reshaping")

print(f"\nLikely issue: PyTorch broadcasts across 2 dimensions [batch, seq], which")
print(f"is slower than broadcasting across 1 dimension in F.linear")

print("=" * 80)

