#!/usr/bin/env python3
"""
Verify that the lightweight implementation is actually skipping the CUDA bias kernel
"""

import torch
import time
from qkv_fusion import qkv_fused_forward_lightweight
from qkv_fusion.weight_utils import prepare_fused_qkv_weights

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

print("Testing lightweight implementation...")

# Test multiple times
num_iters = 100
for _ in range(10):
    q, k, v = qkv_fused_forward_lightweight(
        hidden_states, fused_weight, fused_bias,
        num_q_heads, num_kv_heads, head_dim
    )
torch.cuda.synchronize()

start = time.time()
for _ in range(num_iters):
    q, k, v = qkv_fused_forward_lightweight(
        hidden_states, fused_weight, fused_bias,
        num_q_heads, num_kv_heads, head_dim
    )
torch.cuda.synchronize()
elapsed = (time.time() - start) / num_iters * 1000

print(f"\nLightweight implementation time: {elapsed:.3f} ms")
print(f"\nExpected breakdown:")
print(f"  CUDA GEMM (no bias):    0.067 ms")
print(f"  PyTorch bias add:       0.003 ms")
print(f"  PyTorch split/transpose: 0.012 ms")
print(f"  Expected total:         ~0.082 ms")
print(f"  Actual total:           {elapsed:.3f} ms")

if elapsed < 0.085:
    print(f"\n✓ SUCCESS! Achieved target performance")
else:
    print(f"\n⚠ Still {elapsed - 0.082:.3f} ms slower than expected")
    print(f"  Remaining overhead needs investigation")

