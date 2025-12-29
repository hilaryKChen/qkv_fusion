#!/usr/bin/env python3
"""
Roofline Model Analysis for QKV Fusion Project

This script computes the theoretical performance bounds (roofline model) for
attention operations on your GPU platform, helping identify optimization targets.

The roofline model helps answer:
1. Is your kernel compute-bound or memory-bound?
2. What's the theoretical peak performance?
3. How close are you to the hardware limits?

Usage:
    python roofline_analysis.py [--gpu-info] [--profile-all]
    
Example output:
    GPU: NVIDIA H800 (Hopper)
    Peak FP16 Compute: 989 TFLOPS (Tensor Core)
    Peak Memory BW: 2000 GB/s (HBM3)
    
    QKV Projection Analysis (batch=4, seq=512, hidden=3584):
    - FLOPs: 42.9 GFLOP
    - Bytes: 21 MB (read) + 10.5 MB (write)
    - Arithmetic Intensity: 1.36 FLOPs/Byte
    - Ridge Point: 494 FLOPs/Byte
    - STATUS: MEMORY BOUND (by 363x!)
    - Theoretical Minimum Time: 0.016 ms (at memory BW limit)
    - PyTorch Achieved: 0.073 ms (22% of peak BW)
"""

import argparse
import contextlib
import datetime as _dt
from pathlib import Path
import sys
import torch
import time
from dataclasses import dataclass
from typing import Tuple, Optional
import math


@dataclass
class GPUSpecs:
    """Hardware specifications for roofline analysis"""
    name: str
    compute_capability: Tuple[int, int]
    peak_fp16_tflops: float        # FP16 Tensor Core peak TFLOPS
    peak_fp32_tflops: float        # FP32 TFLOPS (non-tensor core)
    memory_bandwidth_gbps: float   # Memory bandwidth in GB/s
    l2_cache_mb: float             # L2 cache size in MB
    sm_count: int                  # Number of SMs
    
    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity at ridge point (FLOPs/Byte)"""
        # Ridge point = Peak Compute / Peak Memory BW
        return self.peak_fp16_tflops * 1000 / self.memory_bandwidth_gbps


# Common GPU specifications (extend as needed)
GPU_SPECS = {
    "H800": GPUSpecs(
        name="NVIDIA H800",
        compute_capability=(9, 0),
        peak_fp16_tflops=989.0,
        peak_fp32_tflops=67.0,
        memory_bandwidth_gbps=2000.0,
        l2_cache_mb=50.0,
        sm_count=132
    ),
    "H100": GPUSpecs(
        name="NVIDIA H100",
        compute_capability=(9, 0),
        peak_fp16_tflops=989.0,
        peak_fp32_tflops=67.0,
        memory_bandwidth_gbps=3350.0,  # HBM3
        l2_cache_mb=50.0,
        sm_count=132
    ),
    "A100": GPUSpecs(
        name="NVIDIA A100",
        compute_capability=(8, 0),
        peak_fp16_tflops=312.0,
        peak_fp32_tflops=19.5,
        memory_bandwidth_gbps=2039.0,  # HBM2e
        l2_cache_mb=40.0,
        sm_count=108
    ),
    "A6000": GPUSpecs(
        name="NVIDIA RTX A6000",
        compute_capability=(8, 6),
        peak_fp16_tflops=77.4,
        peak_fp32_tflops=38.7,
        memory_bandwidth_gbps=768.0,
        l2_cache_mb=6.0,
        sm_count=84
    ),
    "4090": GPUSpecs(
        name="NVIDIA RTX 4090",
        compute_capability=(8, 9),
        peak_fp16_tflops=165.0,  # FP16 Tensor
        peak_fp32_tflops=82.6,
        memory_bandwidth_gbps=1008.0,
        l2_cache_mb=72.0,
        sm_count=128
    ),
    "3090": GPUSpecs(
        name="NVIDIA RTX 3090",
        compute_capability=(8, 6),
        peak_fp16_tflops=71.0,
        peak_fp32_tflops=35.6,
        memory_bandwidth_gbps=936.0,
        l2_cache_mb=6.0,
        sm_count=82
    ),
}

class _TeeIO:
    """File-like object that writes to multiple streams (e.g., stdout + file)."""

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()



def detect_gpu() -> GPUSpecs:
    """Auto-detect GPU and return specs"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {gpu_name}")
    
    # Match against known GPUs
    for key, specs in GPU_SPECS.items():
        if key in gpu_name.upper() or key in gpu_name.replace(" ", ""):
            print(f"Matched to: {specs.name}")
            return specs
    
    # Default fallback - estimate based on properties
    props = torch.cuda.get_device_properties(0)
    print(f"Unknown GPU, using estimated specs based on properties:")
    print(f"  Total memory: {props.total_memory / 1e9:.1f} GB")
    print(f"  SM count: {props.multi_processor_count}")
    
    # Conservative estimates
    return GPUSpecs(
        name=gpu_name,
        compute_capability=(props.major, props.minor),
        peak_fp16_tflops=100.0,  # Conservative estimate
        peak_fp32_tflops=30.0,
        memory_bandwidth_gbps=800.0,  # Conservative estimate
        l2_cache_mb=40.0,
        sm_count=props.multi_processor_count
    )


@dataclass
class OperationAnalysis:
    """Analysis of a single operation"""
    name: str
    flops: int                  # Total FLOPs
    bytes_read: int             # Bytes read from memory
    bytes_written: int          # Bytes written to memory
    
    @property
    def total_bytes(self) -> int:
        return self.bytes_read + self.bytes_written
    
    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte transferred"""
        return self.flops / self.total_bytes if self.total_bytes > 0 else float('inf')
    
    def is_memory_bound(self, gpu: GPUSpecs) -> bool:
        """Check if operation is memory bound"""
        return self.arithmetic_intensity < gpu.ridge_point
    
    def theoretical_time_ms(self, gpu: GPUSpecs) -> float:
        """Theoretical minimum execution time in ms"""
        # Time limited by memory bandwidth
        mem_time = self.total_bytes / (gpu.memory_bandwidth_gbps * 1e9) * 1000
        # Time limited by compute
        compute_time = self.flops / (gpu.peak_fp16_tflops * 1e12) * 1000
        return max(mem_time, compute_time)
    
    def limiting_factor(self, gpu: GPUSpecs) -> str:
        """Which resource is the bottleneck"""
        return "MEMORY" if self.is_memory_bound(gpu) else "COMPUTE"


def analyze_qkv_projection(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2  # FP16 = 2 bytes
) -> OperationAnalysis:
    """
    Analyze QKV projection operation
    
    Operation: hidden_states @ qkv_weight + bias
    Shapes:
      - hidden_states: [batch, seq, hidden_dim]
      - qkv_weight: [hidden_dim, (num_q_heads + 2*num_kv_heads) * head_dim]
      - qkv_output: [batch, seq, (num_q_heads + 2*num_kv_heads) * head_dim]
    """
    M = batch_size * seq_len
    K = hidden_dim
    N = (num_q_heads + 2 * num_kv_heads) * head_dim
    
    # GEMM FLOPs: 2 * M * N * K (multiply-add)
    gemm_flops = 2 * M * N * K
    
    # Bias addition FLOPs: M * N
    bias_flops = M * N
    
    # Total FLOPs
    total_flops = gemm_flops + bias_flops
    
    # Memory reads
    # - Input: M * K elements
    # - Weight: K * N elements
    # - Bias: N elements (broadcast)
    bytes_read = (M * K + K * N + N) * dtype_bytes
    
    # Memory writes
    # - Output: M * N elements
    bytes_written = M * N * dtype_bytes
    
    return OperationAnalysis(
        name=f"QKV Projection [{M}x{K}] @ [{K}x{N}]",
        flops=total_flops,
        bytes_read=bytes_read,
        bytes_written=bytes_written
    )


def analyze_attention_score(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2
) -> OperationAnalysis:
    """
    Analyze attention score computation: Q @ K^T
    
    Shapes (per head):
      - Q: [batch, heads, seq, head_dim]
      - K: [batch, heads, seq, head_dim]
      - Scores: [batch, heads, seq, seq]
    """
    # For each head: [seq, head_dim] @ [head_dim, seq] = [seq, seq]
    # Per head FLOPs: 2 * seq * seq * head_dim
    per_head_flops = 2 * seq_len * seq_len * head_dim
    total_flops = batch_size * num_heads * per_head_flops
    
    # Memory reads: Q and K
    bytes_read = 2 * batch_size * num_heads * seq_len * head_dim * dtype_bytes
    
    # Memory writes: Attention scores
    bytes_written = batch_size * num_heads * seq_len * seq_len * dtype_bytes
    
    return OperationAnalysis(
        name=f"Attention Scores (Q@K^T) [{seq_len}x{head_dim}]@[{head_dim}x{seq_len}]",
        flops=total_flops,
        bytes_read=bytes_read,
        bytes_written=bytes_written
    )


def analyze_attention_output(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2
) -> OperationAnalysis:
    """
    Analyze attention output computation: softmax(scores) @ V
    
    Shapes (per head):
      - Probs: [batch, heads, seq, seq]
      - V: [batch, heads, seq, head_dim]
      - Output: [batch, heads, seq, head_dim]
    """
    # Per head FLOPs: 2 * seq * head_dim * seq (matmul) + seq * seq (softmax ~= seq*seq ops)
    matmul_flops = 2 * seq_len * head_dim * seq_len
    softmax_flops = 5 * seq_len * seq_len  # exp, sum, div per element
    per_head_flops = matmul_flops + softmax_flops
    total_flops = batch_size * num_heads * per_head_flops
    
    # Memory reads: Scores + V
    scores_bytes = batch_size * num_heads * seq_len * seq_len * dtype_bytes
    v_bytes = batch_size * num_heads * seq_len * head_dim * dtype_bytes
    bytes_read = scores_bytes + v_bytes
    
    # Memory writes: Attention output
    bytes_written = batch_size * num_heads * seq_len * head_dim * dtype_bytes
    
    return OperationAnalysis(
        name=f"Attention Output (softmax@V)",
        flops=total_flops,
        bytes_read=bytes_read,
        bytes_written=bytes_written
    )


def analyze_full_attention(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2
) -> dict:
    """Analyze full attention layer"""
    analyses = {}
    
    # 1. QKV Projection
    analyses['qkv_proj'] = analyze_qkv_projection(
        batch_size, seq_len, hidden_dim,
        num_q_heads, num_kv_heads, head_dim, dtype_bytes
    )
    
    # 2. Attention Scores (Q @ K^T) - use num_q_heads for standard attention
    # For GQA, this would need adjustment
    analyses['attn_scores'] = analyze_attention_score(
        batch_size, seq_len, num_q_heads, head_dim, dtype_bytes
    )
    
    # 3. Attention Output (softmax @ V)
    analyses['attn_output'] = analyze_attention_output(
        batch_size, seq_len, num_q_heads, head_dim, dtype_bytes
    )
    
    # 4. Output Projection
    M = batch_size * seq_len
    N = hidden_dim
    K = num_q_heads * head_dim
    analyses['out_proj'] = OperationAnalysis(
        name=f"Output Projection [{M}x{K}] @ [{K}x{N}]",
        flops=2 * M * N * K,
        bytes_read=(M * K + K * N) * dtype_bytes,
        bytes_written=M * N * dtype_bytes
    )
    
    return analyses


def benchmark_memory_bandwidth(warmup: int = 10, iters: int = 100) -> float:
    """
    Measure actual achievable memory bandwidth on your GPU.
    Uses a simple copy kernel to measure bandwidth.
    
    Returns bandwidth in GB/s.
    """
    # Use large tensors to saturate memory bandwidth
    size = 256 * 1024 * 1024  # 256 MB
    src = torch.randn(size // 4, dtype=torch.float32, device='cuda')
    dst = torch.empty_like(src)
    
    # Warmup
    for _ in range(warmup):
        dst.copy_(src)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Calculate bandwidth (read + write)
    bytes_transferred = 2 * size * iters  # read + write
    bandwidth_gbps = bytes_transferred / elapsed / 1e9
    
    return bandwidth_gbps


def benchmark_compute_throughput(warmup: int = 10, iters: int = 100) -> float:
    """
    Measure actual achievable FP16 compute throughput.
    Uses matrix multiplication to measure TFLOPS.
    
    Returns throughput in TFLOPS.
    """
    # Use large matrices to saturate compute
    M = N = K = 4096
    a = torch.randn(M, K, dtype=torch.float16, device='cuda')
    b = torch.randn(K, N, dtype=torch.float16, device='cuda')
    
    # Warmup
    for _ in range(warmup):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Calculate TFLOPS
    flops_per_iter = 2 * M * N * K
    total_flops = flops_per_iter * iters
    tflops = total_flops / elapsed / 1e12
    
    return tflops


def print_roofline_analysis(
    gpu: GPUSpecs,
    analyses: dict,
    measured_bw: Optional[float] = None,
    measured_compute: Optional[float] = None
):
    """Print formatted roofline analysis"""
    print("\n" + "=" * 80)
    print("ROOFLINE MODEL ANALYSIS")
    print("=" * 80)
    
    print(f"\n{'GPU Information':-^80}")
    print(f"  Device: {gpu.name}")
    print(f"  Compute Capability: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
    print(f"  Peak FP16 Compute: {gpu.peak_fp16_tflops:.1f} TFLOPS (Tensor Core)")
    print(f"  Peak Memory BW: {gpu.memory_bandwidth_gbps:.0f} GB/s")
    print(f"  Ridge Point: {gpu.ridge_point:.1f} FLOPs/Byte")
    
    if measured_bw or measured_compute:
        print(f"\n{'Measured Performance':-^80}")
        if measured_bw:
            efficiency = measured_bw / gpu.memory_bandwidth_gbps * 100
            print(f"  Memory Bandwidth: {measured_bw:.1f} GB/s ({efficiency:.1f}% of peak)")
        if measured_compute:
            efficiency = measured_compute / gpu.peak_fp16_tflops * 100
            print(f"  FP16 Compute: {measured_compute:.1f} TFLOPS ({efficiency:.1f}% of peak)")
    
    print(f"\n{'Operation Analysis':-^80}")
    
    total_flops = 0
    total_bytes = 0
    total_theory_time = 0
    
    for name, analysis in analyses.items():
        total_flops += analysis.flops
        total_bytes += analysis.total_bytes
        theory_time = analysis.theoretical_time_ms(gpu)
        total_theory_time += theory_time
        
        bound = analysis.limiting_factor(gpu)
        ai = analysis.arithmetic_intensity
        
        print(f"\n  [{name}]")
        print(f"    Operation: {analysis.name}")
        print(f"    FLOPs: {analysis.flops / 1e9:.2f} GFLOP")
        print(f"    Memory: {analysis.bytes_read / 1e6:.1f} MB read, {analysis.bytes_written / 1e6:.1f} MB write")
        print(f"    Arithmetic Intensity: {ai:.2f} FLOPs/Byte")
        print(f"    Bound: {bound} BOUND", end="")
        if bound == "MEMORY":
            ratio = gpu.ridge_point / ai
            print(f" (by {ratio:.0f}x below ridge point)")
        else:
            print()
        print(f"    Theoretical Min Time: {theory_time * 1000:.2f} µs")
    
    print(f"\n{'Total Attention Layer':-^80}")
    total_ai = total_flops / total_bytes if total_bytes > 0 else 0
    print(f"  Total FLOPs: {total_flops / 1e9:.2f} GFLOP")
    print(f"  Total Memory: {total_bytes / 1e6:.1f} MB")
    print(f"  Overall Arithmetic Intensity: {total_ai:.2f} FLOPs/Byte")
    print(f"  Theoretical Min Time: {total_theory_time * 1000:.2f} µs")
    
    print(f"\n{'Optimization Guidance':-^80}")
    
    # Find the most memory-bound operation
    most_memory_bound = min(analyses.items(), 
                           key=lambda x: x[1].arithmetic_intensity)
    
    print(f"  Most Memory-Bound Operation: {most_memory_bound[0]}")
    print(f"    -> AI = {most_memory_bound[1].arithmetic_intensity:.2f} FLOPs/Byte")
    
    if most_memory_bound[1].is_memory_bound(gpu):
        print("\n  RECOMMENDATION: Focus on MEMORY OPTIMIZATION")
        print("    - Reduce memory traffic (fusion, caching)")
        print("    - Improve memory access patterns (coalescing)")
        print("    - Use larger batch sizes to amortize overhead")
    else:
        print("\n  RECOMMENDATION: Focus on COMPUTE OPTIMIZATION")
        print("    - Use Tensor Cores (ensure FP16 alignment)")
        print("    - Optimize kernel occupancy")
        print("    - Reduce warp divergence")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Roofline Model Analysis for QKV Fusion")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq", type=int, default=512, help="Sequence length")
    parser.add_argument("--hidden", type=int, default=2048, help="Hidden dimension (Qwen3-30B: 2048)")
    parser.add_argument("--q-heads", type=int, default=32, help="Number of Q heads (Qwen3-30B: 32)")
    parser.add_argument("--kv-heads", type=int, default=4, help="Number of KV heads (Qwen3-30B: 4)")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--measure", action="store_true", help="Measure actual HW performance")
    parser.add_argument("--gpu", type=str, default=None, 
                        help="GPU type override (H800, H100, A100, 4090, etc.)")
    parser.add_argument(
        "--output",
        type=str,
        default="/root/proj/qkv_fusion/benchmarks_and_tests/results_roofline/roofline_analysis.txt",
        help="Optional path to write the full report to (text). Still prints to console.",
    )
    args = parser.parse_args()

    report_path: Optional[Path] = Path(args.output).expanduser() if args.output else None
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    with contextlib.ExitStack() as stack:
        report_fh = None
        if report_path is not None:
            report_fh = stack.enter_context(report_path.open("w", encoding="utf-8"))
            tee = _TeeIO(sys.stdout, report_fh)
            stack.enter_context(contextlib.redirect_stdout(tee))

            # Add a tiny header to the file (and console) for provenance.
            ts = _dt.datetime.now().isoformat(timespec="seconds")
            print(f"[roofline_analysis] timestamp={ts}")
            print(f"[roofline_analysis] argv={' '.join(sys.argv)}")
            print()

        print("=" * 80)
        print("QKV Fusion Project - Roofline Model Analysis")
        print("=" * 80)

        # Detect or set GPU
        if args.gpu and args.gpu.upper() in GPU_SPECS:
            gpu = GPU_SPECS[args.gpu.upper()]
            print(f"Using specified GPU: {gpu.name}")
        else:
            gpu = detect_gpu()

        print(f"\nConfiguration:")
        print(f"  Batch size: {args.batch}")
        print(f"  Sequence length: {args.seq}")
        print(f"  Hidden dimension: {args.hidden}")
        print(f"  Q heads: {args.q_heads}, KV heads: {args.kv_heads}")
        print(f"  Head dimension: {args.head_dim}")

        # Measure actual performance if requested
        measured_bw = None
        measured_compute = None
        if args.measure:
            print("\nMeasuring actual hardware performance...")
            print("  Benchmarking memory bandwidth...")
            measured_bw = benchmark_memory_bandwidth()
            print(f"    -> {measured_bw:.1f} GB/s")

            print("  Benchmarking FP16 compute...")
            measured_compute = benchmark_compute_throughput()
            print(f"    -> {measured_compute:.1f} TFLOPS")

        # Analyze attention operations
        analyses = analyze_full_attention(
            args.batch, args.seq, args.hidden,
            args.q_heads, args.kv_heads, args.head_dim
        )

        # Print analysis
        print_roofline_analysis(gpu, analyses, measured_bw, measured_compute)

        # Also analyze QKV projection specifically (the focus of this project)
        print("\n" + "=" * 80)
        print("QKV PROJECTION SPECIFIC ANALYSIS")
        print("=" * 80)

        qkv_analysis = analyses['qkv_proj']
        print(f"\nThis is your optimization target:")
        print(f"  Operation: {qkv_analysis.name}")
        print(f"  FLOPs: {qkv_analysis.flops / 1e9:.2f} GFLOP")
        print(f"  Arithmetic Intensity: {qkv_analysis.arithmetic_intensity:.2f} FLOPs/Byte")

        # Calculate expected times
        theory_time_us = qkv_analysis.theoretical_time_ms(gpu) * 1000

        print(f"\n  Theoretical bounds:")
        print(f"    - Memory-bound time: {qkv_analysis.total_bytes / (gpu.memory_bandwidth_gbps * 1e6):.2f} µs")
        print(f"    - Compute-bound time: {qkv_analysis.flops / (gpu.peak_fp16_tflops * 1e9):.2f} µs")
        print(f"    - Expected min time: {theory_time_us:.2f} µs")

        # Compare with baseline
        baseline_ms = 0.073  # From your benchmark results
        baseline_us = baseline_ms * 1000
        achieved_efficiency = theory_time_us / baseline_us * 100

        print(f"\n  Your baseline (PyTorch 3x nn.Linear): {baseline_us:.1f} µs")
        print(f"  Theoretical efficiency: {achieved_efficiency:.1f}%")

        if achieved_efficiency < 50:
            print(f"\n  ⚠️  There's {100-achieved_efficiency:.0f}% room for improvement!")
            print(f"     But this doesn't mean you can achieve it with custom kernels.")
            print(f"     PyTorch uses highly optimized cuBLASLt with epilogue fusion.")
        else:
            print(f"\n  ✓ PyTorch is already achieving {achieved_efficiency:.0f}% of theoretical peak.")
            print(f"    Custom kernels may not provide significant speedup.")

        print("\n" + "=" * 80)

    if report_path is not None:
        print(f"\nWrote report to: {report_path}")


if __name__ == "__main__":
    main()

