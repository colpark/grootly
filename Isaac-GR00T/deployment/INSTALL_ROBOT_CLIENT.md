# GR00T Robot Client Installation (Anaconda)

Installation guide for the Trossen AI Mobile robot computer that runs the client.

## Prerequisites

- Trossen AI Mobile robot with:
  - Left arm at IP 192.168.1.5
  - Right arm at IP 192.168.1.4
  - 3 Intel RealSense cameras
- Ubuntu 22.04 (or compatible)
- Anaconda or Miniconda installed
- Network access to the GPU inference server

## Step 1: Create Conda Environment

```bash
# Create a new conda environment with Python 3.11
conda create -n groot_client python=3.11 -y
conda activate groot_client
```

## Step 2: Install System Dependencies

```bash
# Intel RealSense SDK (for cameras)
sudo apt-get update
sudo apt-get install -y \
    libusb-1.0-0-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev

# Install Intel RealSense library
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
    sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get update
sudo apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev
```

## Step 3: Install LeRobot (Trossen AI Branch)

```bash
# Clone LeRobot from Interbotix with Trossen AI support
cd ~
git clone -b trossen-ai https://github.com/Interbotix/lerobot.git lerobot_trossen
cd lerobot_trossen

# Install LeRobot
pip install -e ".[intelrealsense,trossen]"
```

## Step 4: Install GR00T Client Dependencies

```bash
# Core dependencies
pip install numpy opencv-python pyzmq scipy

# GR00T policy client (from the Isaac-GR00T repo)
cd ~/Isaac-GR00T  # or wherever you have the repo
pip install -e .
```

## Step 5: Verify Camera Setup

```bash
# List connected RealSense cameras
rs-enumerate-devices

# Note down the serial numbers for each camera:
# - cam_high: overhead/top camera
# - cam_left_wrist: left arm wrist camera
# - cam_right_wrist: right arm wrist camera
```

## Step 6: Verify Robot Connection

```bash
# Ping the robot arms
ping -c 3 192.168.1.5  # Left arm
ping -c 3 192.168.1.4  # Right arm
```

## Usage

### Test Connection to Server (No Robot Required)

```bash
cd ~/Isaac-GR00T
conda activate groot_client

# Test server connection only
python deployment/trossen_client.py \
    --server-ip 130.199.95.27 \
    --server-port 5559 \
    --test-only
```

### Run with Mock Robot (No Hardware)

```bash
# Test the full pipeline without robot hardware
python deployment/trossen_client.py \
    --server-ip 130.199.95.27 \
    --server-port 5559 \
    --task "Transfer the ball" \
    --mock
```

### Run with Real Robot

```bash
# Run with real robot hardware
python deployment/trossen_client.py \
    --server-ip 130.199.95.27 \
    --server-port 5559 \
    --task "Transfer the ball" \
    --left-arm-ip 192.168.1.5 \
    --right-arm-ip 192.168.1.4 \
    --cam-high-serial YOUR_CAM_HIGH_SERIAL \
    --cam-left-wrist-serial YOUR_LEFT_WRIST_SERIAL \
    --cam-right-wrist-serial YOUR_RIGHT_WRIST_SERIAL
```

### Dry Run Mode (Robot Connected, No Commands Sent)

```bash
# Connect to robot but don't send motor commands
python deployment/trossen_client.py \
    --server-ip 130.199.95.27 \
    --server-port 5559 \
    --task "Transfer the ball" \
    --dry-run --verbose
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--server-ip` | 130.199.95.27 | GPU server IP address |
| `--server-port` | 5559 | GPU server port |
| `--task` | "Stack the lego blocks" | Task instruction |
| `--action-horizon` | 8 | Steps to execute per inference |
| `--control-freq` | 30.0 | Control loop frequency (Hz) |
| `--left-arm-ip` | 192.168.1.5 | Left arm IP address |
| `--right-arm-ip` | 192.168.1.4 | Right arm IP address |
| `--cam-high-serial` | 130322274102 | High camera serial |
| `--cam-left-wrist-serial` | 130322271087 | Left wrist camera serial |
| `--cam-right-wrist-serial` | 130322270184 | Right wrist camera serial |
| `--mock` | False | Use mock robot (no hardware) |
| `--dry-run` | False | Don't send commands to robot |
| `--verbose` | False | Enable verbose logging |
| `--test-only` | False | Only test server connection |

## Troubleshooting

### Camera Not Found

```bash
# Check if cameras are detected
rs-enumerate-devices

# If not detected, try reconnecting USB or:
sudo service udev restart
```

### Robot Arm Not Responding

```bash
# Check network connection
ping 192.168.1.5
ping 192.168.1.4

# Verify robot is powered on and in the correct mode
```

### Server Connection Failed

```bash
# Test network connectivity to server
nc -zv 130.199.95.27 5559

# Check if server is running on the GPU machine
```

### LeRobot Import Error

```bash
# Make sure you're in the correct conda environment
conda activate groot_client

# Reinstall LeRobot
cd ~/lerobot_trossen
pip install -e ".[intelrealsense,trossen]"
```

## Quick Reference: Find Camera Serial Numbers

```bash
# Run this to find all connected RealSense cameras:
python -c "
import pyrealsense2 as rs
ctx = rs.context()
for d in ctx.devices:
    print(f'Camera: {d.get_info(rs.camera_info.name)}')
    print(f'  Serial: {d.get_info(rs.camera_info.serial_number)}')
    print()
"
```

## Architecture

```
┌─────────────────────────────────────┐
│         Robot Computer              │
│                                     │
│  ┌─────────────────────────────┐    │
│  │     trossen_client.py       │    │
│  │  ┌─────────────────────┐    │    │
│  │  │  TrossenRobotInterface   │    │
│  │  │  (LeRobot integration)   │    │
│  │  └──────────┬──────────┘    │    │
│  │             │               │    │        ┌──────────────────┐
│  │  ┌──────────▼──────────┐    │    │        │   GPU Server     │
│  │  │  TrossenMobileAdapter│   │    │        │                  │
│  │  └──────────┬──────────┘    │    │  ZMQ   │  ┌────────────┐  │
│  │             │               │────┼────────┼──│ PolicyServer│  │
│  │  ┌──────────▼──────────┐    │    │  5559  │  │ (GR00T VLA)│  │
│  │  │    PolicyClient     │    │    │        │  └────────────┘  │
│  │  └─────────────────────┘    │    │        │                  │
│  └─────────────────────────────┘    │        └──────────────────┘
│              │                      │
│              ▼                      │
│  ┌─────────────────────────────┐    │
│  │    Trossen AI Mobile        │    │
│  │  ┌─────┐  ┌─────┐  ┌─────┐  │    │
│  │  │Left │  │Base │  │Right│  │    │
│  │  │Arm  │  │     │  │Arm  │  │    │
│  │  └─────┘  └─────┘  └─────┘  │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```
