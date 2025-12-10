# Benchmarks and Tests

This directory contains all testing, benchmarking, and profiling scripts used during development.

## 📁 Directory Structure

```
benchmarks_and_tests/
├── README.md                          # This file
├── results/                           # Test outputs and logs
│   ├── *.txt                         # Benchmark results
│   └── *.log                         # Compilation/test logs
├── roofline_analysis.py               # ⭐ Roofline model analysis (NEW)
├── benchmark_qwen3_baseline.py        # ⭐ Qwen3 baseline benchmark (NEW)
├── benchmark_all_approaches.py        # Main benchmark suite
├── profile_*.py                       # Profiling scripts
├── debug_*.py                         # Debugging/analysis scripts
├── test_*.py                          # Correctness tests
└── run_phase3_benchmark.sh           # Quick benchmark script
```

## 🎯 START HERE: New Analysis Scripts

### 1. Roofline Model Analysis
**Run this first** to understand your optimization target:
```bash
# Basic analysis (auto-detect GPU)
python roofline_analysis.py --batch 1 --seq 512

# With actual hardware measurements
python roofline_analysis.py --batch 4 --seq 512 --measure

# Override GPU type if auto-detection fails
python roofline_analysis.py --gpu H800 --batch 4 --seq 512
```

**What it tells you:**
- Is QKV projection compute-bound or memory-bound?
- What's the theoretical minimum time?
- How close is PyTorch to the limit?

### 2. Baseline Model Benchmark
**Run this second** to establish baseline latencies:
```bash
# Default configuration (Qwen3-7B-like)
python benchmark_qwen3_baseline.py --batch 1 --seq 512

# Custom configuration
python benchmark_qwen3_baseline.py --batch 4 --seq 1024 --hidden 3584

# Save results to JSON
python benchmark_qwen3_baseline.py --output results/baseline.json
```

**What it measures:**
- Prefill latency (processing full prompt)
- Per-token generation latency
- Component breakdown (QKV, attention, MLP)
- Different QKV projection approaches

## 🧪 Main Scripts

### Benchmarking
- **`roofline_analysis.py`** ⭐ - Theoretical performance limits analysis
- **`benchmark_qwen3_baseline.py`** ⭐ - Qwen3 baseline latency measurements
- **`benchmark_all_approaches.py`** - Compare all QKV fusion implementations
- **`run_phase3_benchmark.sh`** - Quick compile + benchmark script

### Profiling
- **`profile_kernels.py`** - Profile individual GEMM operations
- **`profile_lightweight.py`** - Component-by-component profiling

### Debugging/Analysis
- **`debug_gemm_overhead.py`** - Analyze GEMM wrapper overhead
- **`trace_exact_timing.py`** - CUDA event timing breakdown
- **`verify_timing_methods.py`** - Compare timing methodologies
- **`check_contiguous.py`** - Verify tensor memory layout
- **`test_bias_broadcast.py`** - Analyze bias addition performance

### Testing
- **`test_optimized.py`** - Correctness tests for optimized kernel
- **`test_qkv_fusion.py`** - Original baseline tests

## 🚀 Quick Start

### Step 1: Roofline Analysis (START HERE)
```bash
cd benchmarks_and_tests
python roofline_analysis.py --measure
```

### Step 2: Baseline Benchmark
```bash
python benchmark_qwen3_baseline.py --batch 1 --seq 512
```

### Step 3: Compare QKV Fusion Approaches
```bash
python benchmark_all_approaches.py
```

### Profile Components
```bash
python profile_lightweight.py
```

### Debug GEMM Performance
```bash
python debug_gemm_overhead.py
```

## 📊 Key Results

See `results/` directory for saved benchmark outputs:
- Baseline performance: 0.073 ms
- Phase 2 (custom CUDA): 0.111 ms
- Phase 3 (hybrid): 0.098-0.101 ms

## 📝 Understanding the Analysis Scripts

### Roofline Model (`roofline_analysis.py`)
The roofline model helps identify whether an operation is **compute-bound** or **memory-bound**:

- **Arithmetic Intensity (AI)** = FLOPs / Bytes transferred
- **Ridge Point** = Peak Compute / Peak Memory BW
- If AI < Ridge Point → **Memory Bound** (focus on reducing memory traffic)
- If AI > Ridge Point → **Compute Bound** (focus on utilizing tensor cores)

For QKV projection, AI is typically **much lower** than the ridge point, meaning it's memory-bound.

### Baseline Benchmark (`benchmark_qwen3_baseline.py`)
Measures three key metrics (per advisor feedback):

1. **Prefill Latency**: Time to process the entire input sequence
2. **Per-Token Latency**: Time to generate one token (with KV cache)
3. **Component Breakdown**: Where time is spent (QKV, attention, MLP)

## 📝 Notes

- All scripts should be run from the `benchmarks_and_tests` directory
- Requires PyTorch with CUDA support
- Some scripts require the qkv_fusion extension to be compiled first:
  ```bash
  cd .. && pip install -e . --no-build-isolation && cd benchmarks_and_tests
  ```

## 🔗 Related Documentation

- `../PROJECT_COMPLETION_ROADMAP.md` - Full project roadmap and TODO list
- `../progress_analysis/PROJECT_REPORT.md` - Detailed analysis and findings
- `../progress_analysis/OPTIMIZATION_ROADMAP.md` - Optimization strategies

