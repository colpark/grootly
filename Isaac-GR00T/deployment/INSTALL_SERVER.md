# GR00T Server Installation Guide (CUDA 12.x)

This guide covers installing GR00T on a server with CUDA 12.x (tested with CUDA 12.6).

## Prerequisites

- Ubuntu 20.04/22.04
- NVIDIA GPU with CUDA 12.x drivers
- Python 3.10
- At least 24GB GPU VRAM (for inference)

## Step 1: Find Your CUDA Installation

First, locate where CUDA is installed:

```bash
# Check CUDA version from nvidia-smi
nvidia-smi

# Find nvcc location
which nvcc
find /usr -name nvcc 2>/dev/null
find /opt -name nvcc 2>/dev/null

# List CUDA installations
ls -la /usr/local/cuda* 2>/dev/null
```

Common CUDA locations:
- `/usr/local/cuda`
- `/usr/local/cuda-12.6`
- `/usr/local/cuda-12.1`
- `/opt/cuda`

## Step 2: Set CUDA Environment Variables

Once you find CUDA, set the environment variables. Add these to your `~/.bashrc`:

```bash
# Replace /usr/local/cuda-12.6 with your actual CUDA path
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify
source ~/.bashrc
nvcc --version
```

If CUDA toolkit is not installed (only drivers), install it:

```bash
# For Ubuntu 22.04 with CUDA 12.6
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-6
```

## Step 3: Install uv Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal
```

## Step 4: Clone and Setup GR00T

```bash
cd ~/grootly  # or your preferred directory
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```

## Step 5: Install Dependencies (Method A - With flash-attn)

If CUDA toolkit is properly set up:

```bash
# Ensure CUDA environment is set
export CUDA_HOME=/usr/local/cuda-12.6  # adjust path
export PATH=$CUDA_HOME/bin:$PATH

# Create environment
uv sync --python 3.10

# Install package
uv pip install -e .
```

## Step 5: Install Dependencies (Method B - Without flash-attn)

If you can't get flash-attn to build, install without it:

```bash
# Create virtual environment manually
python3.10 -m venv .venv
source .venv/bin/activate

# Install core dependencies without flash-attn
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126

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
    wandb==0.23.0

# Install flash-attn from pre-built wheel (recommended)
# Find compatible wheel at: https://github.com/Dao-AILab/flash-attention/releases
pip install flash-attn --no-build-isolation

# If flash-attn still fails, the model can run without it (slower)
# Install the gr00t package
pip install -e .
```

## Step 6: Install flash-attn from Pre-built Wheel

The easiest way to install flash-attn is using pre-built wheels:

```bash
# Check your PyTorch and CUDA versions
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Download appropriate wheel from:
# https://github.com/Dao-AILab/flash-attention/releases

# For torch 2.7.0 + CUDA 12.6 + Python 3.10:
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Or try installing with no-build-isolation
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## Step 7: Verify Installation

```bash
# Activate environment
source .venv/bin/activate

# Test imports
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

from transformers import AutoModel
print('Transformers OK')

try:
    import flash_attn
    print(f'Flash Attention: {flash_attn.__version__}')
except ImportError:
    print('Flash Attention: Not installed (model will still work)')

from gr00t.policy.gr00t_policy import Gr00tPolicy
print('GR00T imports OK')
"
```

## Step 8: Test Server Deployment

```bash
# Run server test
python deployment/test_server.py --task lego --skip-model

# If that passes, test with model loading
python deployment/test_server.py --task lego --full-test
```

## Troubleshooting

### Error: "No such file or directory: '/usr/local/cuda/bin/nvcc'"

CUDA toolkit not in expected location. Set CUDA_HOME:

```bash
# Find actual CUDA location
find /usr -name nvcc 2>/dev/null

# Set environment (add to ~/.bashrc)
export CUDA_HOME=/path/to/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

### Error: flash-attn build fails

Options:
1. Install from pre-built wheel (see Step 6)
2. Skip flash-attn entirely (model will work, just slower)
3. Install CUDA toolkit with nvcc

### Error: torch not compiled with CUDA

Install PyTorch with CUDA support:

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
```

### Error: Out of memory

- Use a GPU with more VRAM (24GB+ recommended)
- Use `--device cuda:1` if you have multiple GPUs
- Reduce batch size in inference

## Quick Install Script

Save this as `install_groot.sh`:

```bash
#!/bin/bash
set -e

echo "=== GR00T Installation Script ==="

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: nvcc not found. flash-attn may fail to build."
    echo "Set CUDA_HOME to your CUDA installation path."
fi

# Create virtual environment
echo "Creating virtual environment..."
python3.10 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA
echo "Installing PyTorch..."
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
echo "Installing dependencies..."
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
    deepspeed==0.17.6

# Try to install flash-attn
echo "Attempting flash-attn installation..."
pip install flash-attn --no-build-isolation || echo "flash-attn failed, continuing without it"

# Install gr00t package
echo "Installing gr00t..."
pip install -e .

echo "=== Installation complete ==="
echo "Activate with: source .venv/bin/activate"
```

Run with:
```bash
chmod +x install_groot.sh
./install_groot.sh
```

## Server Configuration Summary

| Parameter | Value |
|-----------|-------|
| Server | matrix (130.199.95.27) |
| User | dpark1 |
| Project Path | ~/grootly/Isaac-GR00T |
| Python | 3.10 |
| CUDA | 12.6 |
| Port | 5559 |
