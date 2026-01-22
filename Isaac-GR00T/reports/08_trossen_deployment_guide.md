# Trossen AI Mobile Deployment Guide

**Date**: January 2026
**Tasks**: LEGO Manipulation, Ball Transfer (Ball2), Plug Stacking
**Robot**: Trossen AI Mobile (Bimanual Mobile Robot)
**Model**: GR00T N1.6-3B Finetuned Models
**Architecture**: Server-Client via ZeroMQ

---

## 1. Executive Summary

This report documents the complete deployment architecture for running GR00T policy inference on a remote GPU server while the Trossen AI Mobile robot executes actions in real-time. This setup separates compute-intensive neural network inference from the robot's control computer, enabling deployment on robots with limited compute capabilities.

### 1.1 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────────────────┐              ┌───────────────────────┐         │
│   │   ROBOT COMPUTER      │              │     GPU SERVER        │         │
│   │   (130.199.x.x)       │              │   (130.199.95.27)     │         │
│   │                       │              │                       │         │
│   │  ┌─────────────────┐  │   ZeroMQ    │  ┌─────────────────┐  │         │
│   │  │ trossen_client  │──┼──(TCP)──────┼──│  PolicyServer   │  │         │
│   │  │                 │  │  Port 5559  │  │                 │  │         │
│   │  │  - Observation  │  │             │  │  - GR00T Model  │  │         │
│   │  │  - Adapter      │◄─┼─────────────┼──│  - Inference    │  │         │
│   │  │  - Control Loop │  │   Actions   │  │  - CUDA:0       │  │         │
│   │  └────────┬────────┘  │             │  └─────────────────┘  │         │
│   │           │           │              │                       │         │
│   │           ▼           │              │  Checkpoint:          │         │
│   │  ┌─────────────────┐  │              │  ./outputs/lego/      │         │
│   │  │ TROSSEN ROBOT   │  │              │    checkpoint-10000   │         │
│   │  │                 │  │              │                       │         │
│   │  │  - 3 Cameras    │  │              └───────────────────────┘         │
│   │  │  - 2 Arms       │  │                                                │
│   │  │  - Mobile Base  │  │                                                │
│   │  └─────────────────┘  │                                                │
│   └───────────────────────┘                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Server IP | 130.199.95.27 | GPU server address |
| Server Port | 5559 | ZeroMQ service port |
| Protocol | TCP (ZeroMQ REQ/REP) | Request-reply pattern |
| Checkpoint | checkpoint-10000 | LEGO manipulation model |
| Embodiment | NEW_EMBODIMENT | Trossen AI Mobile |
| Action Horizon | 16 steps | ~0.8s at 20Hz |

### 1.3 Supported Tasks

| Task | Server Script | Client Script | Checkpoint |
|------|---------------|---------------|------------|
| LEGO Manipulation | `run_server_lego.sh` | `run_client_lego.sh` | `./outputs/lego/checkpoint-10000` |
| Ball Transfer | `run_server_ball2.sh` | `run_client_ball2.sh` | `./outputs/ball2_groot/checkpoint-10000` |
| Plug Stacking | `run_server_plug_stacking.sh` | `run_client_plug_stacking.sh` | `./outputs/plug_stacking/checkpoint-10000` |

### 1.4 Files Created

| File | Location | Purpose |
|------|----------|---------|
| **Server Scripts** | | |
| `run_server_lego.sh` | deployment/ | LEGO task server |
| `run_server_ball2.sh` | deployment/ | Ball transfer task server |
| `run_server_plug_stacking.sh` | deployment/ | Plug stacking task server |
| **Client Scripts** | | |
| `run_client_lego.sh` | deployment/ | LEGO task client |
| `run_client_ball2.sh` | deployment/ | Ball transfer task client |
| `run_client_plug_stacking.sh` | deployment/ | Plug stacking task client |
| **Core Components** | | |
| `trossen_adapter.py` | deployment/ | Robot-model adapter |
| `trossen_client.py` | deployment/ | Robot control client |
| **Testing Utilities** | | |
| `test_server.py` | deployment/ | Server environment test |
| `test_connection.py` | deployment/ | Client connection test |

---

## 2. System Requirements

### 2.1 GPU Server Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| GPU | NVIDIA GPU with 24GB+ VRAM | RTX 3090, A5000, A6000, H100 |
| CUDA | 11.8+ | Must match PyTorch build |
| RAM | 32GB+ | For model loading |
| Network | Stable connection to robot | Low latency preferred |
| Storage | 20GB+ | Model checkpoints |

**Required Software:**
```bash
# Python environment
conda activate gr00t  # or your environment

# Required packages
pip install torch torchvision  # CUDA enabled
pip install transformers
pip install pyzmq  # ZeroMQ bindings
pip install msgpack  # Serialization
```

### 2.2 Robot Computer Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| CPU | Modern multi-core | For sensor processing |
| RAM | 8GB+ | Camera buffers |
| Network | Stable connection to server | Ethernet preferred |
| Python | 3.10+ | For client code |

**Required Software:**
```bash
# Minimal dependencies for robot client
pip install numpy
pip install pyzmq
pip install msgpack
```

### 2.3 Network Requirements

| Requirement | Value | Notes |
|-------------|-------|-------|
| Latency | < 50ms round-trip | Higher latency reduces control quality |
| Bandwidth | 10 Mbps+ | For camera image transmission |
| Firewall | Port 5559 open | TCP traffic |
| Protocol | TCP | Reliable ordered delivery |

---

## 3. Task-Specific Quick Start

### 3.1 LEGO Manipulation Task

**On GPU Server (130.199.95.27):**
```bash
cd /path/to/Isaac-GR00T
conda activate gr00t
./deployment/run_server_lego.sh
```

**On Robot Computer:**
```bash
./deployment/run_client_lego.sh

# Or test connection first:
./deployment/run_client_lego.sh --test

# Or dry run without robot:
./deployment/run_client_lego.sh --dry-run
```

### 3.2 Ball Transfer Task (Ball2)

**On GPU Server (130.199.95.27):**
```bash
cd /path/to/Isaac-GR00T
conda activate gr00t
./deployment/run_server_ball2.sh
```

**On Robot Computer:**
```bash
./deployment/run_client_ball2.sh

# Or test connection first:
./deployment/run_client_ball2.sh --test
```

### 3.3 Plug Stacking Task

**On GPU Server (130.199.95.27):**
```bash
cd /path/to/Isaac-GR00T
conda activate gr00t
./deployment/run_server_plug_stacking.sh
```

**On Robot Computer:**
```bash
./deployment/run_client_plug_stacking.sh

# Or test connection first:
./deployment/run_client_plug_stacking.sh --test
```

### 3.4 Task Configuration Summary

| Task | Server Script | Client Script | Default Instruction |
|------|---------------|---------------|---------------------|
| LEGO | `run_server_lego.sh` | `run_client_lego.sh` | "Stack the lego blocks" |
| Ball2 | `run_server_ball2.sh` | `run_client_ball2.sh` | "Transfer the ball from one arm to the other" |
| Plug Stacking | `run_server_plug_stacking.sh` | `run_client_plug_stacking.sh` | "Stack the plugs" |

### 3.5 Custom Task Instructions

Override the default task instruction using environment variables:

```bash
# Custom LEGO instruction
TASK_INSTRUCTION="Build a tower with lego blocks" ./deployment/run_client_lego.sh

# Custom ball transfer instruction
TASK_INSTRUCTION="Pass the ball between hands" ./deployment/run_client_ball2.sh
```

---

## 4. Server Setup Details

### 4.1 Server Launch Scripts

The server runs on the GPU machine and hosts the GR00T policy model. Task-specific scripts are available:

| Script | Checkpoint Path | Default Port |
|--------|-----------------|--------------|
| `run_server_lego.sh` | `./outputs/lego/checkpoint-10000` | 5559 |
| `run_server_ball2.sh` | `./outputs/ball2_groot/checkpoint-10000` | 5559 |
| `run_server_plug_stacking.sh` | `./outputs/plug_stacking/checkpoint-10000` | 5559 |

### 4.2 Server Startup Procedure

**Step 1: Navigate to project directory**
```bash
cd /path/to/Isaac-GR00T
```

**Step 2: Activate environment**
```bash
conda activate gr00t
```

**Step 3: Verify checkpoint exists**
```bash
# For LEGO task:
ls -la ./outputs/lego/checkpoint-10000/

# For Ball2 task:
ls -la ./outputs/ball2_groot/checkpoint-10000/

# For Plug Stacking task:
ls -la ./outputs/plug_stacking/checkpoint-10000/
```

**Step 4: Start the server**
```bash
# Choose one based on task:
./deployment/run_server_lego.sh
./deployment/run_server_ball2.sh
./deployment/run_server_plug_stacking.sh
```

**Expected output (example for LEGO task):**
```
==============================================
GR00T Server: LEGO Manipulation Task
==============================================
Task: lego
Checkpoint: ./outputs/lego/checkpoint-10000
Host: 0.0.0.0
Port: 5559
Device: cuda:0
==============================================

Registered Trossen AI Mobile modality config for finetuning (as NEW_EMBODIMENT)
Starting GR00T inference server...
  Embodiment tag: EmbodimentTag.NEW_EMBODIMENT
  Model path: ./outputs/lego/checkpoint-10000
  Device: cuda:0
  Host: 0.0.0.0
  Port: 5559
Server is ready and listening on tcp://0.0.0.0:5559
```

### 4.3 Server Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `CHECKPOINT` | Task-specific | Model checkpoint path |
| `HOST` | 0.0.0.0 | Bind address |
| `PORT` | 5559 | Service port |
| `DEVICE` | cuda:0 | GPU device |
| `MODALITY_CONFIG` | ./finetuning/trossen_modality_config.py | Robot config |

**Example with custom settings:**
```bash
# Custom checkpoint for LEGO task
CHECKPOINT=./outputs/lego/checkpoint-5000 \
PORT=5560 \
DEVICE=cuda:1 \
./deployment/run_server_lego.sh

# Custom checkpoint for Ball2 task
CHECKPOINT=./outputs/ball2_groot/checkpoint-5000 \
./deployment/run_server_ball2.sh
```

---

## 5. Robot Client Setup

### 5.1 TrossenMobileAdapter Class

The adapter handles conversion between robot sensor data and GR00T model format.

**File:** `deployment/trossen_adapter.py`

**Key responsibilities:**
1. Package camera images as `obs["video"]`
2. Build state vector as `obs["state"]`
3. Add language instruction
4. Add batch/time dimensions (B=1, T=1)
5. Decode action chunks to motor commands

### 5.2 Observation Format

**Input to adapter:**

| Component | Format | Shape |
|-----------|--------|-------|
| `cameras["cam_high"]` | RGB image | (H, W, 3) |
| `cameras["cam_left_wrist"]` | RGB image | (H, W, 3) |
| `cameras["cam_right_wrist"]` | RGB image | (H, W, 3) |
| `state` | Float array | (19,) |
| `language` | String | - |

**State vector breakdown (19 DOF):**

| Index | Component | DOF | Description |
|-------|-----------|-----|-------------|
| 0-2 | base_odom | 3 | odom_x, odom_y, odom_theta |
| 3-4 | base_vel | 2 | linear_vel, angular_vel |
| 5-11 | left_arm | 7 | waist, shoulder, elbow, forearm_roll, wrist_pitch, wrist_roll, gripper |
| 12-18 | right_arm | 7 | waist, shoulder, elbow, forearm_roll, wrist_pitch, wrist_roll, gripper |

### 5.3 Action Format

**Output from adapter:**

| Component | Shape | Representation |
|-----------|-------|----------------|
| `base_vel` | (2,) | ABSOLUTE velocity |
| `left_arm` | (7,) | RELATIVE joint delta |
| `right_arm` | (7,) | RELATIVE joint delta |

### 5.4 Client Usage Example

```python
from gr00t.policy.server_client import PolicyClient
from deployment.trossen_adapter import TrossenMobileAdapter

# Connect to server
client = PolicyClient(
    host="130.199.95.27",
    port=5559,
    timeout_ms=15000,
)

# Create adapter
adapter = TrossenMobileAdapter(client)

# Build observation
cameras = {
    "cam_high": get_camera_image("high"),
    "cam_left_wrist": get_camera_image("left_wrist"),
    "cam_right_wrist": get_camera_image("right_wrist"),
}
state = get_robot_state()  # 19-DOF vector
language = "Stack the lego blocks"

# Get action sequence
actions = adapter.get_action(cameras, state, language)

# Execute actions on robot
for action in actions[:8]:  # Use first 8 of 16 steps
    send_to_robot(action)
    time.sleep(1/30)  # 30 Hz control rate
```

---

## 6. Communication Protocol

### 5.1 ZeroMQ Architecture

The server-client communication uses ZeroMQ's REQ-REP pattern:

```
┌──────────────┐                         ┌──────────────┐
│    Client    │                         │    Server    │
│   (REQ)      │                         │    (REP)     │
├──────────────┤                         ├──────────────┤
│              │    1. Request           │              │
│   Serialize  │ ─────────────────────►  │   Receive    │
│   (msgpack)  │    (observation)        │              │
│              │                         │   Process    │
│              │    2. Response          │   (GPU)      │
│   Receive    │ ◄─────────────────────  │              │
│              │    (action)             │   Serialize  │
│   Decode     │                         │   (msgpack)  │
└──────────────┘                         └──────────────┘
```

### 5.2 Message Format

**Request (Client → Server):**
```python
{
    "endpoint": "get_action",
    "data": {
        "observation": {
            "video": {...},      # Camera images
            "state": {...},      # Robot state
            "language": {...},   # Task instruction
        },
        "options": None,
    },
}
```

**Response (Server → Client):**
```python
(
    {
        "base_vel": np.ndarray,   # (B, T, 2)
        "left_arm": np.ndarray,   # (B, T, 7)
        "right_arm": np.ndarray,  # (B, T, 7)
    },
    {
        "info": {...},  # Additional metadata
    }
)
```

### 5.3 Available Endpoints

| Endpoint | Input | Output | Description |
|----------|-------|--------|-------------|
| `ping` | None | `{"status": "ok"}` | Health check |
| `get_action` | Observation dict | Action tuple | Main inference |
| `reset` | Options dict | Reset info | Reset policy state |
| `get_modality_config` | None | Config dict | Get robot config |
| `kill` | None | None | Stop server |

---

## 7. Testing and Validation

### 7.1 Server Environment Test (Before Deployment)

**File:** `deployment/test_server.py`

Run this on the GPU server **before starting the inference server** to validate the environment:

```bash
# Basic test for LEGO task
python deployment/test_server.py --task lego

# Full test with dummy inference
python deployment/test_server.py --task lego --full-test

# Test specific checkpoint
python deployment/test_server.py --task ball2 --checkpoint ./outputs/ball2_groot/checkpoint-5000

# Test server port binding
python deployment/test_server.py --task lego --test-server --port 5559

# Quick test (skip model loading)
python deployment/test_server.py --task lego --skip-model
```

**What it tests:**
1. Python environment and version
2. Required dependencies (torch, transformers, pyzmq, msgpack)
3. CUDA availability and GPU memory
4. Modality config registration
5. Checkpoint file structure
6. Model loading (optional)
7. Dummy inference timing (with `--full-test`)
8. Server port binding (with `--test-server`)

**Expected output:**
```
============================================================
  GR00T SERVER ENVIRONMENT TEST
============================================================
  Task: lego
  Device: cuda:0
  Full test: True
============================================================

============================================================
  1. Python Environment
============================================================
  Python version >= 3.10                   [✓ PASS]
    → 3.10.12

============================================================
  2. Dependencies
============================================================
  torch                                    [✓ PASS]
    → v2.1.0
  transformers                             [✓ PASS]
    → v4.36.0
  ...

============================================================
  6. Model Loading
============================================================
  Loading model from: ./outputs/lego/checkpoint-10000
  Model loaded                             [✓ PASS]
    → 12.3s
  GPU memory used                          [✓ PASS]
    → 8.45GB

============================================================
  TEST SUMMARY
============================================================
  ✓ ALL TESTS PASSED - Server is ready for deployment!
============================================================
```

### 7.2 Connection Test Utility (Client Side)

**File:** `deployment/test_connection.py`

Run this from the robot computer **after the server is running** to verify connectivity:

```bash
# From robot computer
python deployment/test_connection.py \
    --server-ip 130.199.95.27 \
    --server-port 5559
```

**Expected output:**
```
============================================================
GR00T Policy Server Connection Test
============================================================
Target: tcp://130.199.95.27:5559
Timeout: 15000ms
============================================================

Connecting to server...
Client created successfully

[1/4] Testing ping...
      ✓ Ping successful (latency: 12.3ms)

[2/4] Testing modality config retrieval...
      ✓ Modality config retrieved (latency: 8.1ms)
      Available modalities: ['video', 'state', 'action', 'language']

[3/4] Testing inference (5 trials)...
      Action keys: ['base_vel', 'left_arm', 'right_arm']
        base_vel: shape=(1, 16, 2)
        left_arm: shape=(1, 16, 7)
        right_arm: shape=(1, 16, 7)
      ✓ All inferences successful
      Latency stats: avg=45.2ms, min=42.1ms, max=51.3ms

[4/4] Testing policy reset...
      ✓ Reset successful (latency: 5.2ms)

============================================================
TEST SUMMARY
============================================================
  ping                : ✓ PASS
  modality_config     : ✓ PASS
  inference           : ✓ PASS
  reset               : ✓ PASS
============================================================

✓ All tests passed! Server is ready for robot deployment.
```

### 7.3 Dry Run Mode

Test the client without robot hardware:

```bash
python deployment/trossen_client.py \
    --server-ip 130.199.95.27 \
    --server-port 5559 \
    --dry-run \
    --verbose
```

### 7.4 Performance Benchmarks

Expected latency breakdown:

| Component | Typical Latency | Notes |
|-----------|-----------------|-------|
| Network RTT | 5-20ms | Depends on network |
| Serialization | 2-5ms | msgpack encoding |
| GPU Inference | 30-50ms | Model forward pass |
| Deserialization | 2-5ms | msgpack decoding |
| **Total** | **40-80ms** | Per inference |

---

## 8. Deployment Checklist

### 7.1 Server Setup Checklist

```
□ 1. GPU server accessible at 130.199.95.27
□ 2. Port 5559 open in firewall
□ 3. Isaac-GR00T repository cloned
□ 4. Conda environment activated
□ 5. Checkpoint available at ./outputs/lego/checkpoint-10000
□ 6. Test server startup locally
□ 7. Server running and listening
```

### 7.2 Robot Client Checklist

```
□ 1. Network connection to 130.199.95.27:5559
□ 2. Python environment with dependencies
□ 3. deployment/ folder copied to robot
□ 4. test_connection.py passes all tests
□ 5. Camera feeds working
□ 6. Robot state reading working
□ 7. Robot command sending working
```

### 7.3 Pre-Deployment Verification

```bash
# On server
./deployment/run_trossen_server.sh

# On robot (in another terminal)
python deployment/test_connection.py \
    --server-ip 130.199.95.27 \
    --server-port 5559 \
    --inference-trials 10
```

---

## 9. Control Loop Details

### 8.1 Main Loop Architecture

```python
while running:
    # 1. Read sensors (5-10ms)
    cameras = read_cameras()
    state = read_robot_state()

    # 2. Query policy server (40-80ms)
    actions = adapter.get_action(cameras, state, task)

    # 3. Execute action chunk (8 steps × 33ms = 264ms)
    for action in actions[:8]:
        robot.send_action(action)
        time.sleep(1/30)  # 30 Hz

    # Total loop time: ~320-350ms per inference
```

### 8.2 Action Horizon Strategy

The model outputs 16 action steps, but executing all 16 before re-inferencing may cause drift. Recommended strategies:

| Strategy | Execute | Re-infer | Use Case |
|----------|---------|----------|----------|
| Conservative | 4 steps | Every 4 | High precision tasks |
| Balanced | 8 steps | Every 8 | General manipulation |
| Aggressive | 16 steps | Every 16 | Fast, open-loop tasks |

### 8.3 Error Handling

```python
try:
    actions = adapter.get_action(cameras, state, task)
except zmq.error.Again:
    # Timeout - server not responding
    logger.warning("Server timeout, retrying...")
    continue
except Exception as e:
    # Other error
    logger.error(f"Inference error: {e}")
    robot.safe_stop()
    break
```

---

## 10. Troubleshooting

### 9.1 Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Connection refused | `ConnectionRefusedError` | Check server is running, port open |
| Timeout | `zmq.error.Again` | Increase timeout, check network |
| CUDA OOM | `OutOfMemoryError` | Use smaller batch, check GPU memory |
| Shape mismatch | `RuntimeError: shape` | Verify observation dimensions |
| Modality not found | `KeyError: NEW_EMBODIMENT` | Import modality config before server |

### 9.2 Network Debugging

```bash
# Test basic connectivity
ping 130.199.95.27

# Test port accessibility
nc -zv 130.199.95.27 5559

# Check firewall rules (on server)
sudo iptables -L -n | grep 5559
```

### 9.3 Server Debugging

```bash
# Check GPU status
nvidia-smi

# Check if port is in use
lsof -i :5559

# Check server logs
./deployment/run_trossen_server.sh 2>&1 | tee server.log
```

### 9.4 Client Debugging

```bash
# Verbose mode
python deployment/trossen_client.py --verbose --dry-run

# Test with specific timeout
python deployment/test_connection.py --timeout 30000
```

---

## 11. Security Considerations

### 10.1 Network Security

| Risk | Mitigation |
|------|------------|
| Unauthorized access | Use API token authentication |
| Data interception | Use VPN or private network |
| DoS attacks | Firewall rate limiting |

### 10.2 API Token Authentication

Enable token authentication on server:

```python
# Server side
server = PolicyServer(
    policy=policy,
    host="0.0.0.0",
    port=5559,
    api_token="your-secret-token",  # Set token
)

# Client side
client = PolicyClient(
    host="130.199.95.27",
    port=5559,
    api_token="your-secret-token",  # Must match
)
```

### 10.3 Best Practices

1. **Isolate network**: Use dedicated VLAN for robot-server communication
2. **Use VPN**: For remote deployments across internet
3. **Rotate tokens**: Change API tokens periodically
4. **Monitor traffic**: Log and audit server requests
5. **Limit exposure**: Bind to specific IPs, not 0.0.0.0 in production

---

## 12. Performance Optimization

### 11.1 Reducing Latency

| Optimization | Impact | Implementation |
|--------------|--------|----------------|
| Use wired network | -10-30ms | Ethernet instead of WiFi |
| Compress images | -5-10ms | JPEG compression before send |
| Batch cameras | -2-5ms | Send all cameras in one message |
| GPU warm-up | -20-50ms first call | Run dummy inference on startup |

### 11.2 Improving Throughput

```python
# Pre-allocate arrays
obs_buffer = {
    "video": {k: np.zeros((1,1,256,256,3)) for k in cameras},
    "state": {k: np.zeros((1,1,d)) for k,d in state_dims},
}

# Reuse buffer instead of creating new arrays
def update_observation(raw_obs):
    np.copyto(obs_buffer["video"]["cam_high"], raw_obs["cam_high"])
    # ... etc
```

### 11.3 Multi-Threading

```python
import threading
from queue import Queue

# Separate threads for observation and action
obs_queue = Queue(maxsize=1)
action_queue = Queue(maxsize=1)

def sensor_thread():
    while running:
        obs = read_sensors()
        obs_queue.put(obs)

def inference_thread():
    while running:
        obs = obs_queue.get()
        action = adapter.get_action(obs)
        action_queue.put(action)

def control_thread():
    while running:
        action = action_queue.get()
        robot.send_action(action)
```

---

## 13. Quick Reference

### 13.1 Server Commands

```bash
# Start task-specific server (recommended)
./deployment/run_server_lego.sh           # LEGO manipulation
./deployment/run_server_ball2.sh          # Ball transfer
./deployment/run_server_plug_stacking.sh  # Plug stacking

# Start with custom checkpoint
CHECKPOINT=./outputs/lego/checkpoint-5000 ./deployment/run_server_lego.sh

# Start on different port
PORT=5560 ./deployment/run_server_lego.sh

# Start on specific GPU
DEVICE=cuda:1 ./deployment/run_server_lego.sh
```

### 13.2 Client Commands

```bash
# Test connection
python deployment/test_connection.py --server-ip 130.199.95.27 --server-port 5559

# Task-specific client launchers (recommended)
./deployment/run_client_lego.sh
./deployment/run_client_ball2.sh
./deployment/run_client_plug_stacking.sh

# Dry run (no robot)
./deployment/run_client_lego.sh --dry-run

# Test connection only
./deployment/run_client_lego.sh --test

# Direct client with custom task
python deployment/trossen_client.py \
    --server-ip 130.199.95.27 \
    --server-port 5559 \
    --task "Custom task instruction"
```

### 13.3 Task Configuration Summary

```yaml
LEGO Task:
  Server: ./deployment/run_server_lego.sh
  Client: ./deployment/run_client_lego.sh
  Checkpoint: ./outputs/lego/checkpoint-10000
  Instruction: "Stack the lego blocks"

Ball2 Task:
  Server: ./deployment/run_server_ball2.sh
  Client: ./deployment/run_client_ball2.sh
  Checkpoint: ./outputs/ball2_groot/checkpoint-10000
  Instruction: "Transfer the ball from one arm to the other"

Plug Stacking Task:
  Server: ./deployment/run_server_plug_stacking.sh
  Client: ./deployment/run_client_plug_stacking.sh
  Checkpoint: ./outputs/plug_stacking/checkpoint-10000
  Instruction: "Stack the plugs"

Common Settings:
  Server IP: 130.199.95.27
  Port: 5559
  Device: cuda:0
  Timeout: 15000ms
  Action Horizon: 8 (of 16)
  Control Frequency: 30 Hz

Robot:
  Cameras: cam_high, cam_left_wrist, cam_right_wrist
  State DOF: 19
  Action DOF: 16
```

---

## 14. Appendix

### A. File Structure

```
deployment/
├── run_server_lego.sh           # LEGO task server
├── run_server_ball2.sh          # Ball2 task server
├── run_server_plug_stacking.sh  # Plug stacking task server
├── run_client_lego.sh           # LEGO task client launcher
├── run_client_ball2.sh          # Ball2 task client launcher
├── run_client_plug_stacking.sh  # Plug stacking task client launcher
├── trossen_adapter.py           # Robot-model adapter
├── trossen_client.py            # Robot control client (core)
├── test_server.py               # Server environment test
└── test_connection.py           # Client connection test

outputs/
├── lego/
│   └── checkpoint-10000/        # LEGO finetuned model
├── ball2_groot/
│   └── checkpoint-10000/        # Ball2 finetuned model
└── plug_stacking/
    └── checkpoint-10000/        # Plug stacking finetuned model

finetuning/
└── trossen_modality_config.py   # Robot configuration
```

### B. Dependencies

**Server:**
```
torch>=2.0
transformers
pyzmq
msgpack
numpy
```

**Client:**
```
pyzmq
msgpack
numpy
```

### C. Related Documentation

| Document | Description |
|----------|-------------|
| `reports/05_ball2_groot_inhouse_pipeline.md` | Finetuning pipeline |
| `reports/07_trossen_lego_pipeline.md` | LEGO task pipeline |
| `getting_started/policy.md` | GR00T policy documentation |
| `gr00t/policy/server_client.py` | Server-client implementation |

---

*This deployment guide provides the complete setup for running GR00T inference on a remote server with the Trossen AI Mobile robot. For questions or issues, refer to the troubleshooting section or the related documentation.*
