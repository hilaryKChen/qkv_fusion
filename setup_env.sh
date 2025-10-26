#!/bin/bash
# Setup environment for qkv_fusion CUDA extension
# Source this file before using the package: source setup_env.sh

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate attn-op

# Load CUDA toolkit
module purge
module load cuda12.2/toolkit/12.2.2

# Set up library paths for PyTorch and NVIDIA CUDA libraries
NVIDIA_LIBS=$(find ${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}"

echo "✓ Environment configured for qkv_fusion"
echo "  - Conda env: attn-op"
echo "  - CUDA: 12.2"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"

