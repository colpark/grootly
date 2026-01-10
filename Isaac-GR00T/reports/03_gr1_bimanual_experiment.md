# GR-1 Bimanual Tabletop Experiment

**Date**: January 2026
**Task**: PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
**Model**: nvidia/GR00T-N1.6-3B
**Embodiment**: GR1

---

## 1. Experiment Overview

This experiment evaluates the GR00T N1.6 Vision-Language-Action model on a bimanual pick-and-place task using the Fourier GR-1 humanoid robot in the RoboCasa simulation environment.

### Task Description
The GR-1 humanoid robot uses both arms to pick an object from one plate and place it on another plate. This demonstrates the model's capability for bimanual coordination.

### Expected Performance
According to the GR00T N1 paper (GR-1 Tabletop Tasks):

| Model | Plate to Plate Success Rate |
|-------|---------------------------|
| Diffusion Policy (100 demos) | ~50% |
| GR00T N1 (100 demos) | ~65% |
| **GR00T N1.6 (Zero-shot)** | **78.7%** |

---

## 2. Environment Configuration

### 2.1 Embodiment Details

| Property | Value |
|----------|-------|
| Embodiment Tag | `GR1` |
| Robot | Fourier GR-1 Humanoid |
| Configuration | Arms + Waist + Fourier Hands |
| Control Type | Bimanual manipulation |
| Control Frequency | 20 Hz |

### 2.2 Observation Space

| Key | Shape | Description |
|-----|-------|-------------|
| `video.ego_view_bg_crop_pad_res256_freq20` | (256, 256, 3) | Ego-centric view (background cropped) |
| `video.ego_view_pad_res256_freq20` | (256, 256, 3) | Ego-centric view (padded) |
| `state.left_arm` | (7,) | Left arm joint positions |
| `state.right_arm` | (7,) | Right arm joint positions |
| `state.left_hand` | (6,) | Left hand joint positions |
| `state.right_hand` | (6,) | Right hand joint positions |
| `state.waist` | (3,) | Waist joint positions |
| `annotation.human.coarse_action` | string | Language instruction |

### 2.3 Action Space

| Key | Dimensions | Description |
|-----|------------|-------------|
| `action.left_arm` | 7 | Left arm joint velocities |
| `action.right_arm` | 7 | Right arm joint velocities |
| `action.left_hand` | 6 | Left hand joint commands |
| `action.right_hand` | 6 | Right hand joint commands |
| `action.waist` | 3 | Waist joint velocities |

**Total Action Dimensions**: 29

---

## 3. Environment Setup

### 3.1 Prerequisites

The GR-1 environment requires additional setup beyond the base RoboCasa installation.

```bash
# Create directory for external dependencies
mkdir -p external_dependencies

# Clone GR-1 tabletop tasks repository
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks \
    external_dependencies/robocasa-gr1-tabletop-tasks
```

### 3.2 Run Setup Script

```bash
bash gr00t/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh
```

### 3.3 Manual Dependency Installation (if needed)

```bash
source gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/activate

# Critical version requirements
uv pip install mujoco==3.2.6
uv pip install numpy==1.26.4
uv pip install robosuite==1.5.1 --no-deps

# Additional dependencies
uv pip install scipy numba h5py imageio gymnasium opencv-python
uv pip install transformers pyzmq msgpack av
```

### 3.4 Download Assets

```bash
# Kitchen environment assets
python external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/scripts/download_kitchen_assets.py

# GR00T-specific assets
python external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/scripts/download_groot_assets.py
```

### 3.5 Environment Activation

```bash
# Activate GR-1 environment
conda activate py310
cd /path/to/Isaac-GR00T
source gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

---

## 4. Running the Experiment

### 4.1 Start the Policy Server (Terminal 1)

**CRITICAL**: Use `--embodiment-tag GR1` for GR-1 tasks!

```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag GR1 \
    --use-sim-policy-wrapper
```

**Expected Output:**
```
Starting GR00T inference server...
  Embodiment tag: EmbodimentTag.GR1
  Model path: nvidia/GR00T-N1.6-3B
  Device: cuda
  Host: 127.0.0.1
  Port: 5555
```

### 4.2 Run Evaluation (Terminal 2)

#### Using Official Script

```bash
gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps 720 \
    --env_name gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env \
    --n_action_steps 8 \
    --n_envs 1
```

#### Using Custom Evaluation Script (with video recording)

```bash
gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python scripts/transparent_evaluation.py \
    --env-name gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env \
    --n-episodes 3 \
    --save-dir ./gr1_eval_videos \
    --camera-view side \
    --policy-host 127.0.0.1 \
    --policy-port 5555 \
    --max-steps 720 \
    --n-action-steps 8
```

---

## 5. Experiment Parameters

### 5.1 Evaluation Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_episodes` | 10 | Number of evaluation episodes |
| `max_episode_steps` | 720 | Maximum steps per episode |
| `n_action_steps` | 8 | Action chunk size |
| `n_envs` | 1 | Parallel environments |
| `terminate_on_success` | True | Stop episode on success |

### 5.2 Key Differences from Panda Experiment

| Aspect | RoboCasa Panda | GR-1 Bimanual |
|--------|----------------|---------------|
| Robot Type | Single arm | Dual arms + waist |
| Action Dims | 7 | 29 |
| Video Key | `res256_image_side_*` | `ego_view_bg_crop_*` |
| Max Steps | 504 | 720 |
| Embodiment | ROBOCASA_PANDA_OMRON | GR1 |
| venv Path | `robocasa/robocasa_uv/` | `robocasa-gr1-tabletop-tasks/robocasa_uv/` |

---

## 6. Results

### 6.1 Our Experimental Results

| Metric | Value |
|--------|-------|
| Episodes Run | 10 |
| Successful Episodes | 9 |
| **Success Rate** | **90%** |
| Average Episode Length | ~400 steps |
| Total Evaluation Time | ~5 minutes |

### 6.2 Per-Episode Results

```
Episode 1: SUCCESS (True)
Episode 2: SUCCESS (True)
Episode 3: SUCCESS (True)
Episode 4: SUCCESS (True)
Episode 5: SUCCESS (True)
Episode 6: SUCCESS (True)
Episode 7: SUCCESS (True)
Episode 8: FAILURE (False)
Episode 9: SUCCESS (True)
Episode 10: SUCCESS (True)

Results: [True, True, True, True, True, True, True, False, True, True]
Success Rate: 0.9
```

### 6.3 Comparison with Paper

| Source | Success Rate |
|--------|-------------|
| Paper (N1.6 Zero-shot, Plate to Plate) | 78.7% |
| **Our Replication** | **90.0%** |

Our results exceed the paper's reported success rate, which may be due to:
- Randomness in episode sampling
- Slightly different environment initialization
- Small sample size (10 episodes vs paper's larger evaluation)

---

## 7. Technical Details

### 7.1 Embodiment Tag Mismatch Issue

A common error when running GR-1 evaluation:

```
RuntimeError: Server error: Video key 'video.res256_image_side_0' must be in observation
```

**Cause**: Server started with wrong embodiment tag (e.g., `ROBOCASA_PANDA_OMRON` instead of `GR1`)

**Solution**: Restart server with correct tag:
```bash
# Kill existing server
fuser -k 5555/tcp

# Restart with GR1 tag
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag GR1 \
    --use-sim-policy-wrapper
```

### 7.2 Model Modality Configuration

The model's `processor_config.json` defines what observation keys each embodiment expects:

**GR1 Configuration:**
```json
{
  "gr1": {
    "video": {
      "delta_indices": [0],
      "modality_keys": ["ego_view_bg_crop_pad_res256_freq20"]
    },
    "state": {
      "delta_indices": [0],
      "modality_keys": ["left_arm", "right_arm", "left_hand", "right_hand", "waist"],
      "sin_cos_embedding_keys": ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
    },
    "action": {
      "delta_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
      "modality_keys": ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
    },
    "language": {
      "delta_indices": [0],
      "modality_keys": ["task"]
    }
  }
}
```

### 7.3 Bimanual Coordination

The GR-1 task requires coordinating:
- **Left arm** (7 DOF): Reach and grasp object
- **Right arm** (7 DOF): Support or secondary manipulation
- **Left hand** (6 DOF): Grip control
- **Right hand** (6 DOF): Grip control
- **Waist** (3 DOF): Body positioning

The model learns to coordinate these through end-to-end training on demonstration data.

---

## 8. Available GR-1 Tasks

Other GR-1 tabletop tasks available for evaluation:

| Task | Expected Success |
|------|-----------------|
| Plate to Plate | 78.7% |
| Tray to Plate | 71.0% |
| Cutting Board to Pan | 68.5% |
| Cutting Board to Pot | 65.0% |
| Tray to Pot | 64.5% |
| Placemat to Plate | 63.0% |
| Cutting Board to Basket | 58.0% |
| Placemat to Basket | 58.5% |
| Placemat to Bowl | 57.5% |

To run a different task, change `--env_name`:
```bash
--env_name gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
```

---

## 9. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `Video key 'res256_image_side_0' must be in observation` | Restart server with `--embodiment-tag GR1` |
| `ModuleNotFoundError: robocasa` | Use correct venv: `robocasa-gr1-tabletop-tasks/robocasa_uv/.venv` |
| `MuJoCo version must be 3.2.6` | `uv pip install mujoco==3.2.6` |
| `Object registry directory not found` | Run `download_kitchen_assets.py` and `download_groot_assets.py` |
| Controller warnings (head/base/legs) | Safe to ignore - GR1 tabletop uses upper body only |

### Verifying Environment Observation Keys

```bash
gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python -c "
import os
os.environ['MUJOCO_GL'] = 'egl'
import gymnasium as gym
import robocasa
from robocasa.utils.gym_utils import GrootRoboCasaEnv

env = gym.make('gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env', enable_render=True)
obs, _ = env.reset()
print('Observation keys:')
for k in sorted(obs.keys()):
    val = obs[k]
    shape = val.shape if hasattr(val, 'shape') else type(val)
    print(f'  {k}: {shape}')
env.close()
"
```

**Expected Output:**
```
Observation keys:
  annotation.human.coarse_action: <class 'str'>
  state.left_arm: (7,)
  state.left_hand: (6,)
  state.right_arm: (7,)
  state.right_hand: (6,)
  state.waist: (3,)
  video.ego_view_bg_crop_pad_res256_freq20: (256, 256, 3)
  video.ego_view_pad_res256_freq20: (256, 256, 3)
```

---

## 10. Reproducing Results

### Complete Reproduction Script

```bash
#!/bin/bash
# reproduce_gr1_experiment.sh

# Setup environment
conda activate py310
cd /path/to/Isaac-GR00T
source gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Kill any existing server on port 5555
fuser -k 5555/tcp 2>/dev/null || true

# Start server in background with GR1 embodiment
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag GR1 \
    --use-sim-policy-wrapper &
SERVER_PID=$!

# Wait for server to start
sleep 30

# Run evaluation
gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps 720 \
    --env_name gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env \
    --n_action_steps 8 \
    --n_envs 1

# Cleanup
kill $SERVER_PID
```

---

## 11. Key Takeaways

1. **Embodiment Tag is Critical**: The server must be started with `--embodiment-tag GR1` for GR-1 tasks. Using the wrong tag causes observation key mismatches.

2. **Different Virtual Environments**: GR-1 and Panda tasks use different virtual environments with different dependencies.

3. **Bimanual Coordination**: The model successfully coordinates 29 action dimensions across two arms, two hands, and the waist.

4. **Zero-Shot Performance**: The model achieves 90% success rate without any fine-tuning on the specific task, demonstrating strong generalization.

5. **Longer Episodes**: GR-1 tasks require more steps (720 vs 504) due to the complexity of bimanual manipulation.

---

## 12. References

- GR00T N1 Paper: https://arxiv.org/abs/2503.14734
- Fourier GR-1 Robot: https://www.fftai.com/products/gr1
- RoboCasa GR-1 Tasks: https://github.com/robocasa/robocasa-gr1-tabletop-tasks
- HuggingFace Model: https://huggingface.co/nvidia/GR00T-N1.6-3B
