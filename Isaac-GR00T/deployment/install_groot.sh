#!/bin/bash
# GR00T Installation Script for CUDA 12.x Servers
#
# Usage:
#   ./deployment/install_groot.sh
#
# This script installs GR00T dependencies without requiring nvcc
# by using pre-built wheels and pip instead of uv sync.

set -e

echo "=============================================="
echo "GR00T Installation Script (CUDA 12.x)"
echo "=============================================="

# Check Python version
PYTHON_VERSION=$(python3.10 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
    echo "ERROR: Python 3.10 required, found: $PYTHON_VERSION"
    echo "Install Python 3.10 or use: python3.10 explicitly"
    exit 1
fi
echo "Python version: OK ($PYTHON_VERSION)"

# Check CUDA
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
    echo "CUDA toolkit: $NVCC_VERSION"
else
    echo "WARNING: nvcc not found in PATH"
    echo "flash-attn will be installed from pre-built wheel"

    # Try to find CUDA
    CUDA_PATHS=("/usr/local/cuda" "/usr/local/cuda-12.6" "/usr/local/cuda-12.4" "/usr/local/cuda-12.1")
    for path in "${CUDA_PATHS[@]}"; do
        if [[ -d "$path" ]]; then
            echo "Found CUDA at: $path"
            export CUDA_HOME=$path
            export PATH=$CUDA_HOME/bin:$PATH
            break
        fi
    done
fi

# Check nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "GPU: $GPU_NAME (Driver: $DRIVER_VERSION)"
else
    echo "ERROR: nvidia-smi not found. NVIDIA drivers not installed?"
    exit 1
fi

echo ""
echo "Creating virtual environment..."
python3.10 -m venv .venv
source .venv/bin/activate

echo ""
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

echo ""
echo "Installing PyTorch with CUDA 12.6..."
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126

# Verify PyTorch CUDA
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available in PyTorch!'
print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')
"

echo ""
echo "Installing core dependencies..."
pip install \
    transformers==4.51.3 \
    diffusers==0.35.1 \
    peft==0.17.1 \
    accelerate \
    einops==0.8.1 \
    omegaconf==2.3.0 \
    tyro==0.9.17 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    scipy==1.15.3 \
    matplotlib==3.10.1 \
    pyzmq==27.0.1 \
    msgpack==1.1.0 \
    msgpack-numpy==0.4.8 \
    albumentations==1.4.18 \
    av==15.0.0 \
    lmdb==1.7.5 \
    dm-tree==0.1.8 \
    termcolor==3.2.0 \
    click==8.1.8 \
    datasets==3.6.0 \
    gymnasium==1.2.2 \
    wandb==0.23.0 \
    torchcodec==0.4.0

echo ""
echo "Installing deepspeed..."
pip install deepspeed==0.17.6 || echo "WARNING: deepspeed install failed, continuing..."

echo ""
echo "Installing flash-attn..."
# Try pre-built wheel first
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

if pip install "$FLASH_ATTN_WHEEL" 2>/dev/null; then
    echo "flash-attn installed from pre-built wheel"
else
    echo "Pre-built wheel failed, trying pip install..."
    if pip install flash-attn==2.7.4.post1 --no-build-isolation 2>/dev/null; then
        echo "flash-attn installed via pip"
    else
        echo "WARNING: flash-attn installation failed"
        echo "The model will still work but may be slower"
    fi
fi

echo ""
echo "Installing gr00t package..."
pip install -e .

echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import transformers
print(f'Transformers: {transformers.__version__}')

try:
    import flash_attn
    print(f'Flash Attention: {flash_attn.__version__}')
except ImportError:
    print('Flash Attention: Not installed (OK - model will still work)')

import pyzmq
print(f'PyZMQ: {pyzmq.__version__}')

# Test gr00t imports
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyServer, PolicyClient
print('GR00T imports: OK')
"

echo ""
echo "=============================================="
echo "Installation complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To test the server setup:"
echo "  python deployment/test_server.py --task lego --skip-model"
echo ""
echo "To run the inference server:"
echo "  ./deployment/run_server_lego.sh"
echo ""
