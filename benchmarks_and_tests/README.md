# Benchmarks and Tests

This directory contains all testing, benchmarking, and profiling scripts used during development.

## 📁 Directory Structure

```
benchmarks_and_tests/
├── README.md                          # This file
├── results/                           # Test outputs and logs
│   ├── *.txt                         # Benchmark results
│   └── *.log                         # Compilation/test logs
├── benchmark_all_approaches.py        # Main benchmark suite
├── profile_*.py                       # Profiling scripts
├── debug_*.py                         # Debugging/analysis scripts
├── test_*.py                          # Correctness tests
└── run_phase3_benchmark.sh           # Quick benchmark script
```

## 🧪 Main Scripts

### Benchmarking
- **`benchmark_all_approaches.py`** - Compare all implementations (Phase 2, Phase 3, baseline)
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

### Run Main Benchmark
```bash
cd /home/kchenbx/attention_optimization/qkv_fusion
python benchmarks_and_tests/benchmark_all_approaches.py
```

### Profile Components
```bash
python benchmarks_and_tests/profile_lightweight.py
```

### Debug GEMM Performance
```bash
python benchmarks_and_tests/debug_gemm_overhead.py
```

## 📊 Key Results

See `results/` directory for saved benchmark outputs:
- Baseline performance: 0.073 ms
- Phase 2 (custom CUDA): 0.111 ms
- Phase 3 (hybrid): 0.098-0.101 ms

## 📝 Notes

All scripts should be run from the project root directory to ensure proper imports.

