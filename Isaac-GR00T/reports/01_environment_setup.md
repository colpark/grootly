# GR00T N1 Environment Setup Guide

**Date**: January 2026
**Repository**: Isaac-GR00T
**Model**: nvidia/GR00T-N1.6-3B

---

## Overview

This guide documents the complete setup process for running GR00T N1 Vision-Language-Action (VLA) model evaluations in simulation. The setup supports two main evaluation environments:

1. **RoboCasa Panda** - Single-arm manipulation tasks (CloseDrawer, etc.)
2. **GR-1 Tabletop** - Bimanual humanoid robot tasks (Pick and Place, etc.)

---

## System Requirements

### Hardware
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 16GB | 24GB+ |
| RAM | 32GB | 64GB |
| Disk Space | 20GB | 50GB |
| Display | Required for simulation rendering | - |

### Software
- Python 3.10
- CUDA 11.8+ (for GPU acceleration)
- Ubuntu 20.04/22.04 or compatible Linux

---

## 1. Base Environment Setup

### 1.1 Clone the Repository

```bash
git clone https://github.com/NVIDIA-Omniverse/Isaac-GR00T.git
cd Isaac-GR00T
```

### 1.2 Install UV Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal
```

### 1.3 Create Base Python Environment

```bash
# Using conda
conda create -n py310 python=3.10 -y
conda activate py310

# Install base dependencies
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
uv pip install transformers huggingface_hub tyro
```

---

## 2. RoboCasa Panda Environment Setup

This environment is for single-arm manipulation tasks with the Panda robot.

### 2.1 Run the Setup Script

```bash
# Install system dependencies (Ubuntu)
sudo apt install libegl1-mesa-dev libglu1-mesa

# Run the RoboCasa setup script
bash gr00t/eval/sim/robocasa/setup_RoboCasa.sh
```

### 2.2 Verify Installation

```bash
# Activate the RoboCasa virtual environment
source gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/activate

# Test import
python -c "import robocasa; import robosuite; print('RoboCasa OK')"
```

### 2.3 Environment Variables

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### 2.4 Quick Environment Alias

Add to `~/.bashrc`:

```bash
alias panda-env='conda activate py310 && cd /path/to/Isaac-GR00T && source gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/activate && export MUJOCO_GL=egl && export PYOPENGL_PLATFORM=egl'
```

---

## 3. GR-1 Tabletop Environment Setup

This environment is for bimanual humanoid robot tasks with the Fourier GR-1.

### 3.1 Clone the GR-1 Tabletop Tasks Repository

The setup script may fail due to git submodule issues. Manual clone is recommended:

```bash
# Create external dependencies directory
mkdir -p external_dependencies

# Clone manually
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks \
    external_dependencies/robocasa-gr1-tabletop-tasks
```

### 3.2 Run the Setup Script

```bash
bash gr00t/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh
```

### 3.3 Manual Dependency Installation (if needed)

If the setup script fails, install dependencies manually:

```bash
# Activate the GR-1 venv
source gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/activate

# Install specific versions (critical for compatibility)
uv pip install mujoco==3.2.6
uv pip install numpy==1.26.4
uv pip install robosuite==1.5.1 --no-deps

# Install remaining dependencies
uv pip install scipy numba h5py imageio gymnasium opencv-python
uv pip install transformers pyzmq msgpack av
```

### 3.4 Download Required Assets

```bash
# Download kitchen assets
python external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/scripts/download_kitchen_assets.py

# Download GR00T-specific assets
python external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/scripts/download_groot_assets.py
```

### 3.5 Verify Installation

```bash
source gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python -c "
import gymnasium as gym
import robocasa
from robocasa.utils.gym_utils import GrootRoboCasaEnv
print('GR-1 Environment OK')
"
```

### 3.6 Quick Environment Alias

Add to `~/.bashrc`:

```bash
alias gr1-env='conda activate py310 && cd /path/to/Isaac-GR00T && source gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/activate && export MUJOCO_GL=egl && export PYOPENGL_PLATFORM=egl'
```

---

## 4. Model Download

The GR00T N1.6 model is automatically downloaded from HuggingFace on first use:

```bash
# Pre-download the model (optional)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/GR00T-N1.6-3B')
print('Model downloaded successfully')
"
```

Model files are cached at: `~/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/`

---

## 5. Architecture Overview

### Client-Server Architecture

GR00T uses a client-server architecture for evaluation:

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐         ZeroMQ          ┌──────────────┐  │
│  │              │  ←───────────────────→  │              │  │
│  │   CLIENT     │     Port 5555           │   SERVER     │  │
│  │ (Evaluation  │                         │  (Policy     │  │
│  │   Script)    │     Observations →      │   Inference) │  │
│  │              │     ← Actions           │              │  │
│  └──────────────┘                         └──────────────┘  │
│         │                                        │          │
│         ▼                                        ▼          │
│  ┌──────────────┐                         ┌──────────────┐  │
│  │  Simulation  │                         │  GR00T N1.6  │  │
│  │ Environment  │                         │    Model     │  │
│  │  (MuJoCo)    │                         │   (GPU)      │  │
│  └──────────────┘                         └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| `run_gr00t_server.py` | Loads model, starts ZeroMQ server |
| `rollout_policy.py` | Official evaluation client |
| `transparent_evaluation.py` | Custom evaluation with video recording |
| `PolicyClient` | ZeroMQ client for sending observations |
| `Gr00tSimPolicyWrapper` | Adapts observations for model input |
| `MultiStepWrapper` | Handles action chunking (8-16 steps) |

---

## 6. Embodiment Configuration

Each robot type has a specific embodiment tag and modality configuration:

| Embodiment Tag | Robot | Video Keys | State Keys |
|----------------|-------|------------|------------|
| `ROBOCASA_PANDA_OMRON` | Panda Arm | `res256_image_side_0/1`, `res256_image_wrist_0` | `x,y,z,roll,pitch,yaw,gripper` |
| `GR1` | Fourier GR-1 | `ego_view_bg_crop_pad_res256_freq20` | `left_arm,right_arm,left_hand,right_hand,waist` |
| `UNITREE_G1` | Unitree G1 | `ego_view` | `left_leg,right_leg,waist,left_arm,right_arm,left_hand,right_hand` |

**Critical**: The server's `--embodiment-tag` must match the environment being evaluated!

---

## 7. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `MuJoCo version must be 3.2.6` | `uv pip install mujoco==3.2.6` |
| `numpy version must be...` | `uv pip install numpy==1.26.4` |
| `robosuite version must be 1.5.{0,1}` | `uv pip install robosuite==1.5.1 --no-deps` |
| `evdev compilation error` | Install robosuite with `--no-deps`, skip pynput |
| `Video key 'X' must be in observation` | Restart server with correct `--embodiment-tag` |
| `Object registry directory not found` | Run asset download scripts |
| `MUJOCO_GL` errors | `export MUJOCO_GL=egl` |

### Checking Model Configuration

```bash
# View model's expected observation keys for each embodiment
python -c "
import json
with open('~/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/snapshots/*/processor_config.json') as f:
    config = json.load(f)
for emb, cfg in config['processor_kwargs']['modality_configs'].items():
    print(f'{emb}: {cfg[\"video\"][\"modality_keys\"]}')
"
```

---

## 8. Directory Structure

```
Isaac-GR00T/
├── gr00t/
│   ├── eval/
│   │   ├── run_gr00t_server.py      # Policy server
│   │   ├── rollout_policy.py        # Official evaluation
│   │   └── sim/
│   │       ├── robocasa/            # Panda environment
│   │       │   ├── setup_RoboCasa.sh
│   │       │   └── robocasa_uv/.venv/
│   │       └── robocasa-gr1-tabletop-tasks/  # GR-1 environment
│   │           ├── setup_RoboCasaGR1TabletopTasks.sh
│   │           └── robocasa_uv/.venv/
│   ├── policy/
│   │   ├── gr00t_policy.py          # Policy implementation
│   │   └── server_client.py         # ZeroMQ communication
│   └── data/
│       └── embodiment_tags.py       # Embodiment definitions
├── scripts/
│   ├── transparent_evaluation.py    # Custom evaluation script
│   └── verify_gr1_install.py        # Installation verification
├── external_dependencies/
│   └── robocasa-gr1-tabletop-tasks/ # Cloned GR-1 tasks repo
└── reports/                         # This documentation
```

---

## 9. References

- [GR00T N1 Paper](https://arxiv.org/abs/2503.14734)
- [Isaac-GR00T Repository](https://github.com/NVIDIA-Omniverse/Isaac-GR00T)
- [RoboCasa Documentation](https://github.com/robocasa/robocasa)
- [HuggingFace Model](https://huggingface.co/nvidia/GR00T-N1.6-3B)
