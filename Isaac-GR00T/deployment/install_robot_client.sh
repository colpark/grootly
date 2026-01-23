#!/bin/bash
# ============================================
# GR00T Robot Client Installation Script
# For Trossen AI Mobile robot computer (Conda)
# ============================================

set -e

echo "============================================"
echo "GR00T Robot Client Installer"
echo "============================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Environment name
ENV_NAME="${ENV_NAME:-groot_client}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo ""
echo "Configuration:"
echo "  Environment: ${ENV_NAME}"
echo "  Python: ${PYTHON_VERSION}"
echo ""

# Step 1: Create conda environment
echo "[1/5] Creating conda environment..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Environment '${ENV_NAME}' already exists. Activating..."
else
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Step 2: Install Python dependencies
echo ""
echo "[2/5] Installing Python dependencies..."
pip install --upgrade pip
pip install numpy opencv-python pyzmq scipy

# Step 3: Install pyrealsense2
echo ""
echo "[3/5] Installing pyrealsense2..."
pip install pyrealsense2 || {
    echo "WARNING: pyrealsense2 pip install failed."
    echo "You may need to install the Intel RealSense SDK manually."
    echo "See: https://github.com/IntelRealSense/librealsense"
}

# Step 4: Install LeRobot from Interbotix
echo ""
echo "[4/5] Installing LeRobot (Trossen AI branch)..."
LEROBOT_DIR="${HOME}/lerobot_trossen"

if [[ -d "${LEROBOT_DIR}" ]]; then
    echo "  LeRobot directory exists at ${LEROBOT_DIR}"
    echo "  Updating..."
    cd ${LEROBOT_DIR}
    git fetch origin
    git checkout trossen-ai
    git pull origin trossen-ai
else
    echo "  Cloning LeRobot..."
    git clone -b trossen-ai https://github.com/Interbotix/lerobot.git ${LEROBOT_DIR}
    cd ${LEROBOT_DIR}
fi

# Install LeRobot with Trossen support
pip install -e ".[intelrealsense,trossen]" || {
    echo "  Trying minimal install..."
    pip install -e .
}

# Step 5: Return to original directory and show status
echo ""
echo "[5/5] Verifying installation..."
cd -

python -c "
import sys
print(f'Python: {sys.version}')

try:
    import numpy as np
    print(f'NumPy: {np.__version__}')
except ImportError as e:
    print(f'NumPy: FAILED - {e}')

try:
    import cv2
    print(f'OpenCV: {cv2.__version__}')
except ImportError as e:
    print(f'OpenCV: FAILED - {e}')

try:
    import zmq
    print(f'ZMQ: {zmq.__version__}')
except ImportError as e:
    print(f'ZMQ: FAILED - {e}')

try:
    import pyrealsense2 as rs
    print(f'RealSense: OK')
except ImportError as e:
    print(f'RealSense: FAILED - {e}')

try:
    from lerobot.common.robot_devices.robots.factory import make_robot
    print(f'LeRobot: OK')
except ImportError as e:
    print(f'LeRobot: FAILED - {e}')
"

echo ""
echo "============================================"
echo "Installation complete!"
echo "============================================"
echo ""
echo "To use the client:"
echo "  conda activate ${ENV_NAME}"
echo "  cd ~/Isaac-GR00T"
echo "  python deployment/trossen_client.py --help"
echo ""
echo "Quick test (mock mode, no robot required):"
echo "  python deployment/trossen_client.py --server-ip SERVER_IP --mock"
echo ""
