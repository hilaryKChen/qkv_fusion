#!/bin/bash
# Quick script to compile and benchmark Phase 3 lightweight implementation

set -e  # Exit on error

echo "================================================================================"
echo "Phase 3 Lightweight QKV Fusion - Compilation and Benchmark"
echo "================================================================================"

# Change to project directory
cd "$(dirname "$0")"

echo ""
echo "[1/2] Recompiling CUDA extension with lightweight kernel..."
echo "--------------------------------------------------------------------------------"
pip install -e . --force-reinstall --no-deps

echo ""
echo "[2/2] Running comprehensive benchmark..."
echo "--------------------------------------------------------------------------------"
python benchmark_all_approaches.py

echo ""
echo "================================================================================"
echo "Benchmark complete!"
echo "================================================================================"
echo ""
echo "Expected outcome:"
echo "  - Phase 3 Lightweight should be ~1.5x faster than Phase 2 Optimized"
echo "  - Should match or beat PyTorch baseline (3 nn.Linear)"
echo ""
echo "If Phase 3 is slower, check:"
echo "  1. CUDA compilation warnings/errors above"
echo "  2. GPU utilization (nvidia-smi)"
echo "  3. Memory bandwidth bottlenecks"
echo ""

