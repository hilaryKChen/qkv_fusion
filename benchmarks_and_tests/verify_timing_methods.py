#!/usr/bin/env python3
"""
Compare different timing methods to verify which is accurate
"""

import torch
import time
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
print("Comparing Timing Methods")
print("=" * 80)

# Method 1: time.time() with synchronize (debug_gemm_overhead.py style)
print("\n[Method 1] time.time() + synchronize (batch timing)")
for _ in range(10):
    qkv_output = qkv_fusion_cuda.qkv_fused_forward_lightweight(
        hidden_states, fused_weight, None,
        num_q_heads, num_kv_heads, head_dim
    )
torch.cuda.synchronize()

start = time.time()
num_iters = 100
for _ in range(num_iters):
    qkv_output = qkv_fusion_cuda.qkv_fused_forward_lightweight(
        hidden_states, fused_weight, None,
        num_q_heads, num_kv_heads, head_dim
    )
torch.cuda.synchronize()
method1_time = (time.time() - start) / num_iters * 1000
print(f"  Result: {method1_time:.3f} ms")

# Method 2: CUDA Events WITHOUT synchronize per iteration (correct way)
print("\n[Method 2] CUDA Events (batch timing, no per-iteration sync)")
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Warmup
for _ in range(10):
    qkv_output = qkv_fusion_cuda.qkv_fused_forward_lightweight(
        hidden_states, fused_weight, None,
        num_q_heads, num_kv_heads, head_dim
    )
torch.cuda.synchronize()

start_event.record()
for _ in range(num_iters):
    qkv_output = qkv_fusion_cuda.qkv_fused_forward_lightweight(
        hidden_states, fused_weight, None,
        num_q_heads, num_kv_heads, head_dim
    )
end_event.record()
torch.cuda.synchronize()
method2_time = start_event.elapsed_time(end_event) / num_iters
print(f"  Result: {method2_time:.3f} ms")

# Method 3: CUDA Events WITH synchronize per iteration (trace_exact_timing.py style - WRONG!)
print("\n[Method 3] CUDA Events (per-iteration recording + sync) - HAS OVERHEAD")
times = []
for _ in range(num_iters):
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    
    start_ev.record()
    qkv_output = qkv_fusion_cuda.qkv_fused_forward_lightweight(
        hidden_states, fused_weight, None,
        num_q_heads, num_kv_heads, head_dim
    )
    end_ev.record()
    torch.cuda.synchronize()
    
    times.append(start_ev.elapsed_time(end_ev))

method3_time = sum(times) / len(times)
print(f"  Result: {method3_time:.3f} ms")
print(f"  ⚠ This method adds ~{method3_time - method2_time:.3f} ms overhead per call!")

# Method 4: PyTorch profiler (ground truth)
print("\n[Method 4] torch.profiler (most accurate)")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False
) as prof:
    for _ in range(num_iters):
        qkv_output = qkv_fusion_cuda.qkv_fused_forward_lightweight(
            hidden_states, fused_weight, None,
            num_q_heads, num_kv_heads, head_dim
        )
torch.cuda.synchronize()

# Get average CUDA time
cuda_times = []
for event in prof.key_averages():
    if 'cuda' in event.key.lower() or 'gemm' in event.key.lower():
        cuda_times.append(event.cuda_time_total / event.count / 1000)  # Convert to ms

if cuda_times:
    method4_time = sum(cuda_times) / len(cuda_times)
    print(f"  Result: {method4_time:.3f} ms (CUDA kernel time)")
else:
    print(f"  Could not extract CUDA time from profiler")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"\nMethod 1 (time.time batch):          {method1_time:.3f} ms")
print(f"Method 2 (CUDA events batch):        {method2_time:.3f} ms")
print(f"Method 3 (CUDA events per-iter):     {method3_time:.3f} ms ⚠ HAS OVERHEAD")

print(f"\n✓ Accurate measurement: {min(method1_time, method2_time):.3f} ms")
print(f"\nConclusion:")
print(f"  - trace_exact_timing.py was using Method 3 (per-iteration events)")
print(f"  - This added ~{method3_time - method2_time:.3f} ms overhead")
print(f"  - debug_gemm_overhead.py was using Method 1 (correct!)")
print(f"  - The real GEMM time is ~{method1_time:.3f} ms, not {method3_time:.3f} ms")

print("=" * 80)

