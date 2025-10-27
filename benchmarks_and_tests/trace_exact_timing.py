#!/usr/bin/env python3
"""
Use CUDA events to get exact timing of each operation
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

q_dim = num_q_heads * head_dim
kv_dim = num_kv_heads * head_dim

print("=" * 80)
print("Detailed CUDA Event Timing")
print("=" * 80)

# Warmup
for _ in range(20):
    qkv_output = qkv_fusion_cuda.qkv_fused_forward_lightweight(
        hidden_states, fused_weight, None,
        num_q_heads, num_kv_heads, head_dim
    )
    if fused_bias is not None:
        qkv_output = qkv_output + fused_bias
    q = qkv_output[:, :, :q_dim]
    k = qkv_output[:, :, q_dim:q_dim + kv_dim]
    v = qkv_output[:, :, q_dim + kv_dim:]
    q = q.view(batch_size, seqlen, num_q_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
torch.cuda.synchronize()

# Create CUDA events
start = torch.cuda.Event(enable_timing=True)
after_gemm = torch.cuda.Event(enable_timing=True)
after_bias = torch.cuda.Event(enable_timing=True)
after_split = torch.cuda.Event(enable_timing=True)
after_transpose = torch.cuda.Event(enable_timing=True)

# Time each operation
num_iters = 100
times = {"gemm": [], "bias": [], "split": [], "transpose": []}

for _ in range(num_iters):
    start.record()
    
    # Step 1: CUDA GEMM
    qkv_output = qkv_fusion_cuda.qkv_fused_forward_lightweight(
        hidden_states, fused_weight, None,
        num_q_heads, num_kv_heads, head_dim
    )
    after_gemm.record()
    
    # Step 2: PyTorch bias add
    if fused_bias is not None:
        qkv_output = qkv_output + fused_bias
    after_bias.record()
    
    # Step 3: Split
    q = qkv_output[:, :, :q_dim]
    k = qkv_output[:, :, q_dim:q_dim + kv_dim]
    v = qkv_output[:, :, q_dim + kv_dim:]
    after_split.record()
    
    # Step 4: Transpose
    q = q.view(batch_size, seqlen, num_q_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seqlen, num_kv_heads, head_dim).transpose(1, 2)
    after_transpose.record()
    
    torch.cuda.synchronize()
    
    times["gemm"].append(start.elapsed_time(after_gemm))
    times["bias"].append(after_gemm.elapsed_time(after_bias))
    times["split"].append(after_bias.elapsed_time(after_split))
    times["transpose"].append(after_split.elapsed_time(after_transpose))

# Calculate averages
avg_gemm = sum(times["gemm"]) / num_iters
avg_bias = sum(times["bias"]) / num_iters
avg_split = sum(times["split"]) / num_iters
avg_transpose = sum(times["transpose"]) / num_iters
total = avg_gemm + avg_bias + avg_split + avg_transpose

print(f"\nAverage times over {num_iters} iterations:")
print(f"  1. CUDA GEMM (no bias):      {avg_gemm:.3f} ms")
print(f"  2. PyTorch bias add:         {avg_bias:.3f} ms")
print(f"  3. Split (slicing):          {avg_split:.3f} ms")
print(f"  4. Transpose:                {avg_transpose:.3f} ms")
print(f"  ─────────────────────────────────────")
print(f"  Total:                       {total:.3f} ms")

print(f"\nExpected vs Actual:")
print(f"  Expected GEMM:               0.067 ms")
print(f"  Actual GEMM:                 {avg_gemm:.3f} ms")
print(f"  GEMM overhead:               {avg_gemm - 0.067:.3f} ms")

if avg_gemm > 0.075:
    print(f"\n⚠ PROBLEM: CUDA GEMM is slower than expected!")
    print(f"  Possible causes:")
    print(f"  - CUDA kernel is not actually skipping bias")
    print(f"  - Memory allocation overhead")
    print(f"  - cuBLAS configuration issue")

print("=" * 80)
