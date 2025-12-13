#!/bin/bash
# Setup environment for qkv_fusion CUDA extension
# Source this file before using the package: source setup_env.sh


# Load required modules
module purge
module load Anaconda3/2023.09-0
module load gcc/13.1.0
module load cuda12.2/toolkit/12.2.2
module load nvhpc/23.11

# Activate conda environment
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate attn-op

# Set up library paths for PyTorch and NVIDIA CUDA libraries
NVIDIA_LIBS=$(find ${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}"

echo "✓ Environment configured for qkv_fusion"
echo "  - Conda env: attn-op"
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda if hasattr(torch.version, 'cuda') else 'unknown')")
echo "  - CUDA: $CUDA_VERSION"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"

