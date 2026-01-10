# RoboCasa Panda CloseDrawer Experiment

**Date**: January 2026
**Task**: CloseDrawer_PandaOmron_Env
**Model**: nvidia/GR00T-N1.6-3B
**Embodiment**: ROBOCASA_PANDA_OMRON

---

## 1. Experiment Overview

This experiment evaluates the GR00T N1.6 Vision-Language-Action model on the CloseDrawer task using the RoboCasa simulation environment with a Panda robot arm.

### Task Description
The robot must close an open drawer in a kitchen environment. This is a zero-shot evaluation task - the model was not fine-tuned on this specific environment.

### Expected Performance
According to the GR00T N1 paper:
| Model | CloseDrawer Success Rate |
|-------|-------------------------|
| Diffusion Policy (100 demos) | 88.2% |
| GR00T N1 (100 demos) | 96.1% |
| **GR00T N1.6 (Zero-shot)** | **100.0%** |

---

## 2. Environment Configuration

### 2.1 Embodiment Details

| Property | Value |
|----------|-------|
| Embodiment Tag | `ROBOCASA_PANDA_OMRON` |
| Robot | Franka Panda with Omron mobile base |
| Action Space | 7-DOF arm + gripper |
| Control Frequency | 20 Hz |

### 2.2 Observation Space

| Key | Shape | Description |
|-----|-------|-------------|
| `video.res256_image_side_0` | (256, 256, 3) | Side view of workspace |
| `video.res256_image_side_1` | (256, 256, 3) | Alternative side view |
| `video.res256_image_wrist_0` | (256, 256, 3) | Wrist-mounted camera |
| `state.x` | (1,) | End-effector X position |
| `state.y` | (1,) | End-effector Y position |
| `state.z` | (1,) | End-effector Z position |
| `state.roll` | (1,) | End-effector roll |
| `state.pitch` | (1,) | End-effector pitch |
| `state.yaw` | (1,) | End-effector yaw |
| `state.gripper` | (1,) | Gripper state |
| `task` | string | Language instruction |

### 2.3 Action Space

| Key | Dimensions | Description |
|-----|------------|-------------|
| `action.x` | 1 | X velocity |
| `action.y` | 1 | Y velocity |
| `action.z` | 1 | Z velocity |
| `action.roll` | 1 | Roll velocity |
| `action.pitch` | 1 | Pitch velocity |
| `action.yaw` | 1 | Yaw velocity |
| `action.gripper` | 1 | Gripper command |

---

## 3. Running the Experiment

### 3.1 Environment Setup

```bash
# Activate environment
conda activate py310
cd /path/to/Isaac-GR00T
source gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### 3.2 Start the Policy Server (Terminal 1)

```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sim-policy-wrapper
```

**Expected Output:**
```
Starting GR00T inference server...
  Embodiment tag: EmbodimentTag.ROBOCASA_PANDA_OMRON
  Model path: nvidia/GR00T-N1.6-3B
  Device: cuda
  Host: 127.0.0.1
  Port: 5555
```

### 3.3 Run Evaluation (Terminal 2)

#### Using Official Script

```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps 504 \
    --env_name robocasa_panda_omron/CloseDrawer_PandaOmron_Env \
    --n_action_steps 8 \
    --n_envs 1
```

#### Using Custom Evaluation Script (with video recording)

```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python scripts/transparent_evaluation.py \
    --env-name robocasa_panda_omron/CloseDrawer_PandaOmron_Env \
    --n-episodes 3 \
    --save-dir ./drawer_eval_videos \
    --camera-view side \
    --policy-host 127.0.0.1 \
    --policy-port 5555 \
    --max-steps 504 \
    --n-action-steps 8
```

---

## 4. Experiment Parameters

### 4.1 Evaluation Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_episodes` | 10 | Number of evaluation episodes |
| `max_episode_steps` | 504 | Maximum steps per episode |
| `n_action_steps` | 8 | Action chunk size |
| `n_envs` | 1 | Parallel environments |
| `terminate_on_success` | True | Stop episode on success |

### 4.2 Video Recording Options

| Option | Description |
|--------|-------------|
| `--camera-view side` | Record side camera only (recommended) |
| `--camera-view wrist` | Record wrist camera only |
| `--camera-view both` | Record side + wrist cameras |
| `--camera-view all` | Record all available cameras |

---

## 5. Results

### 5.1 Our Experimental Results

| Metric | Value |
|--------|-------|
| Episodes Run | 10 |
| Successful Episodes | 10 |
| **Success Rate** | **100%** |
| Average Episode Length | ~200 steps |
| Total Evaluation Time | ~3 minutes |

### 5.2 Per-Episode Results

```
Episode 1: SUCCESS
Episode 2: SUCCESS
Episode 3: SUCCESS
Episode 4: SUCCESS
Episode 5: SUCCESS
Episode 6: SUCCESS
Episode 7: SUCCESS
Episode 8: SUCCESS
Episode 9: SUCCESS
Episode 10: SUCCESS
```

### 5.3 Comparison with Paper

| Source | Success Rate |
|--------|-------------|
| Paper (N1.6 Zero-shot) | 100.0% |
| **Our Replication** | **100.0%** |

---

## 6. Video Output

Videos are saved to the specified `--save-dir` directory:

```
drawer_eval_videos/
├── episode_001_success.mp4
├── episode_002_success.mp4
├── episode_003_success.mp4
├── eval_summary.json
└── eval_log.txt
```

### Video Format
- Resolution: 256x256 (per camera)
- Frame Rate: 20 FPS
- Codec: H.264
- Format: MP4

---

## 7. Technical Details

### 7.1 Action Chunking

The model uses action chunking - it predicts 16 future actions at once, but only executes 8:

```
Model Output: [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16]
                └──────────── Executed ────────────┘  └────── Discarded ──────┘
```

This provides:
- Smoother trajectories
- Better temporal consistency
- Faster effective control rate

### 7.2 Observation Preprocessing

```python
# Observations are batched and normalized
observation = {
    "video": {
        "res256_image_side_0": np.ndarray[uint8, (B, T, 256, 256, 3)],
        "res256_image_side_1": np.ndarray[uint8, (B, T, 256, 256, 3)],
        "res256_image_wrist_0": np.ndarray[uint8, (B, T, 256, 256, 3)],
    },
    "state": {
        "x": np.ndarray[float32, (B, T, 1)],
        "y": np.ndarray[float32, (B, T, 1)],
        # ... other state keys
    },
    "language": {
        "task": [["close the drawer"]],  # (B, T) list of strings
    }
}
```

### 7.3 Model Architecture

The GR00T N1.6 model uses:
- **Vision Encoder**: Eagle-Block2A-2B-v2 (2B parameters)
- **Action Head**: Diffusion Transformer (32 layers)
- **Total Parameters**: ~3B
- **Inference**: 4 diffusion steps

---

## 8. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `Video key 'res256_image_side_0' must be in observation` | Ensure server has `--embodiment-tag ROBOCASA_PANDA_OMRON` |
| Environment not found | Verify RoboCasa installation: `python -c "import robocasa"` |
| CUDA out of memory | Use smaller batch size or single environment |
| Video too wide (6 cameras) | Use `--camera-view side` instead of `all` |

### Verifying Server Embodiment

```bash
# Check what the server is expecting
# The server prints this on startup:
#   Embodiment tag: EmbodimentTag.ROBOCASA_PANDA_OMRON
```

---

## 9. Available RoboCasa Tasks

Other tasks available for evaluation with similar setup:

| Task | Expected Success |
|------|-----------------|
| CloseDrawer | 100.0% |
| CloseSingleDoor | 96.0% |
| CloseDoubleDoor | 88.5% |
| CoffeePressButton | 98.5% |
| TurnOffMicrowave | 96.0% |
| TurnOnMicrowave | 91.5% |
| TurnOffSinkFaucet | 93.5% |
| TurnOnSinkFaucet | 89.0% |
| OpenDrawer | 81.1% |

To run a different task, change `--env_name`:
```bash
--env_name robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env
```

---

## 10. Reproducing Results

### Complete Reproduction Script

```bash
#!/bin/bash
# reproduce_drawer_experiment.sh

# Setup environment
conda activate py310
cd /path/to/Isaac-GR00T
source gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Start server in background
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sim-policy-wrapper &
SERVER_PID=$!

# Wait for server to start
sleep 30

# Run evaluation
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python scripts/transparent_evaluation.py \
    --env-name robocasa_panda_omron/CloseDrawer_PandaOmron_Env \
    --n-episodes 10 \
    --save-dir ./drawer_eval_results \
    --camera-view side \
    --policy-host 127.0.0.1 \
    --policy-port 5555 \
    --max-steps 504 \
    --n-action-steps 8

# Cleanup
kill $SERVER_PID
```

---

## 11. References

- GR00T N1 Paper: https://arxiv.org/abs/2503.14734
- RoboCasa: https://github.com/robocasa/robocasa
- HuggingFace Model: https://huggingface.co/nvidia/GR00T-N1.6-3B
