#!/usr/bin/env python3
"""
Check if the CUDA kernel output is contiguous
"""

import torch
from qkv_fusion.weight_utils import prepare_fused_qkv_weights
import qkv_fusion_cuda

batch_size = 4
seqlen = 512
hidden_dim = 2048
num_q_heads = 32
num_kv_heads = 4
head_dim = 128

device = torch.device("cuda:0")
dtype = torch.float16

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
print("Checking tensor properties")
print("=" * 80)

# Call CUDA kernel
qkv_output = qkv_fusion_cuda.qkv_fused_forward_lightweight(
    hidden_states, fused_weight, None,
    num_q_heads, num_kv_heads, head_dim
)

print(f"\n[1] CUDA kernel output:")
print(f"    Shape: {qkv_output.shape}")
print(f"    Strides: {qkv_output.stride()}")
print(f"    Is contiguous: {qkv_output.is_contiguous()}")
print(f"    Memory format: {qkv_output.is_contiguous(memory_format=torch.contiguous_format)}")

# Now test bias add speed
import time

print(f"\n[2] Testing bias add performance:")

# Test on contiguous tensor
contiguous_output = qkv_output.contiguous()
print(f"    Contiguous output shape: {contiguous_output.shape}")
print(f"    Contiguous output strides: {contiguous_output.stride()}")

# Warmup
for _ in range(10):
    result1 = qkv_output + fused_bias
    result2 = contiguous_output + fused_bias
torch.cuda.synchronize()

# Benchmark original (potentially non-contiguous)
start = time.time()
for _ in range(100):
    result1 = qkv_output + fused_bias
torch.cuda.synchronize()
time_original = (time.time() - start) / 100 * 1000

# Benchmark contiguous
start = time.time()
for _ in range(100):
    result2 = contiguous_output + fused_bias
torch.cuda.synchronize()
time_contiguous = (time.time() - start) / 100 * 1000

print(f"\n    Bias add on original output:    {time_original:.3f} ms")
print(f"    Bias add on contiguous output:  {time_contiguous:.3f} ms")
print(f"    Speedup with contiguous:        {time_original/time_contiguous:.2f}x")

if time_contiguous < time_original * 0.5:
    print(f"\n✓ FOUND THE PROBLEM!")
    print(f"  The CUDA kernel output is non-contiguous, causing slow bias add!")
    print(f"  Solution: Make CUDA kernel output contiguous or call .contiguous() in Python")
else:
    print(f"\n⚠ Bias is slow for another reason")

print("=" * 80)

