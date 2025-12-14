#!/bin/bash
# Setup script for new machine with PyTorch 2.5.1 + CUDA 12.4
# Usage: bash setup_new_machine.sh

set -e

echo "=================================================================="
echo "Setting up environment for Qwen3-30B-A3B-GPTQ-Int4"
echo "Machine: PyTorch 2.5.1 + CUDA 12.4"
echo "=================================================================="

# Create conda environment
echo "Creating conda environment..."
conda create -n attn-op python=3.10 -y
conda activate attn-op

# Install PyTorch 2.5.1 with CUDA 12.4
echo "Installing PyTorch 2.5.1 with CUDA 12.4..."
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Install auto-gptq WITH CUDA extensions (CRITICAL!)
echo "Installing auto-gptq with CUDA extensions..."
pip install auto-gptq --no-build-isolation

# Install core dependencies
echo "Installing core dependencies..."
pip install transformers>=4.57.1
pip install optimum>=2.0.0
pip install accelerate>=1.11.0

# Install other requirements
echo "Installing other packages..."
pip install -r requirements_new_machine.txt

# Optional: Install flash-attn if needed
# echo "Installing flash-attn (this may take 5-10 minutes)..."
# pip install flash-attn --no-build-isolation

# Install QKV Fusion package
echo "Installing QKV Fusion package..."
pip install git+https://github.com/hilaryKChen/qkv_fusion.git@e57029270e09637cd86159df9a8a2345a84c0680

echo ""
echo "=================================================================="
echo "Setup complete!"
echo "=================================================================="
echo "To verify GPTQ loading works:"
echo "  python -c \"from transformers import AutoModelForCausalLM; print('OK')\""
echo ""
echo "You should NOT see 'CUDA extension not installed' warnings"
echo "=================================================================="

