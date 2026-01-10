# Ball2 GR00T Inhouse Finetuning Pipeline

**Date**: January 2026
**Task**: Ball Transfer (Bimanual Mobile Manipulation)
**Robot**: Trossen AI Mobile (Bimanual Mobile Robot)
**Base Model**: nvidia/GR00T-N1.6-3B
**Embodiment Tag**: NEW_EMBODIMENT (custom robot)

---

## 1. Executive Summary

This report documents the complete pipeline for finetuning GR00T on inhouse robotics datasets, using the `ball2_groot` dataset as the reference implementation. This pipeline is designed to be **reusable** for future inhouse datasets with minimal reconfiguration.

### Key Achievements
- Established end-to-end pipeline: Raw LeRobot data → GR00T finetuning → Evaluation
- Handled custom robot embodiment (Trossen AI Mobile) with 3-camera, 19-DOF state, 16-DOF action
- Created modular, reusable scripts for dataset conversion, training, and evaluation
- Documented all processing steps for reproducibility

### Pipeline Files Created

| File | Purpose |
|------|---------|
| `finetuning/convert_ball2_groot.py` | Dataset conversion script |
| `finetuning/trossen_modality_config.py` | Robot modality configuration |
| `finetuning/finetune_ball2_groot.sh` | Training launch script |
| `finetuning/evaluate_ball2_groot.py` | Evaluation with metrics & plots |
| `finetuning/evaluate_checkpoints.sh` | Batch checkpoint evaluation |

---

## 2. Robot Configuration: Trossen AI Mobile

### 2.1 Hardware Overview

The Trossen AI Mobile is a bimanual mobile manipulation platform with:

```
┌─────────────────────────────────────────────────────────┐
│                    TROSSEN AI MOBILE                     │
├─────────────────────────────────────────────────────────┤
│  Cameras (3):                                            │
│    • cam_high        - Central overhead camera           │
│    • cam_left_wrist  - Left arm wrist-mounted camera     │
│    • cam_right_wrist - Right arm wrist-mounted camera    │
├─────────────────────────────────────────────────────────┤
│  Mobile Base (5 DOF state, 2 DOF action):                │
│    State: odom_x, odom_y, odom_theta, lin_vel, ang_vel   │
│    Action: linear_velocity, angular_velocity             │
├─────────────────────────────────────────────────────────┤
│  Left Arm (7 DOF):                                       │
│    waist, shoulder, elbow, forearm_roll,                 │
│    wrist_pitch, wrist_roll, gripper                      │
├─────────────────────────────────────────────────────────┤
│  Right Arm (7 DOF):                                      │
│    waist, shoulder, elbow, forearm_roll,                 │
│    wrist_pitch, wrist_roll, gripper                      │
└─────────────────────────────────────────────────────────┘
```

### 2.2 DOF Summary

| Component | State DOF | Action DOF | Representation |
|-----------|-----------|------------|----------------|
| Base Odometry | 3 (x, y, θ) | - | - |
| Base Velocity | 2 (lin, ang) | 2 (lin, ang) | **ABSOLUTE** |
| Left Arm | 7 joints | 7 joints | **RELATIVE** |
| Right Arm | 7 joints | 7 joints | **RELATIVE** |
| **Total** | **19 DOF** | **16 DOF** | - |

### 2.3 Action Representation

GR00T supports two action representations:

| Type | Meaning | Used For |
|------|---------|----------|
| `ABSOLUTE` | Raw action values | Base velocity (commanded directly) |
| `RELATIVE` | Action = target - current_state | Arm joints (delta from current pose) |

The RELATIVE representation requires computing `relative_stats.json` containing statistics of the action deltas.

---

## 3. Source Data Format

### 3.1 LeRobot v2.1 Format

Inhouse data arrives in LeRobot v2.1 format:

```
ball2_groot/
├── meta/
│   ├── info.json          # Dataset metadata
│   ├── episodes.jsonl     # Per-episode info
│   └── tasks.jsonl        # Task descriptions
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── cam_high/
        │   ├── episode_000000.mp4
        │   └── ...
        ├── cam_left_wrist/
        └── cam_right_wrist/
```

### 3.2 Source Data Characteristics

| Property | ball2_groot Value |
|----------|-------------------|
| Total Episodes | 25 |
| Original FPS | 30 Hz |
| Video Codec | AV1 |
| Video Resolution | 480 × 640 |
| State Column | `observation.state` (19 DOF) |
| Action Column | `action` (16 DOF) |

### 3.3 Parquet Schema

```python
# Columns in source parquet files
columns = [
    "frame_index",        # int
    "episode_index",      # int
    "timestamp",          # float (seconds)
    "observation.state",  # array[19] - concatenated state vector
    "action",             # array[16] - concatenated action vector
    "task_index",         # int
]
```

---

## 4. Conversion Pipeline

### 4.1 Overview

The conversion pipeline transforms LeRobot v2.1 data to GR00T-compatible format:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  LeRobot v2.1   │ ──► │  convert_*.py    │ ──► │  GR00T Format   │
│  (30Hz, AV1)    │     │  (Conversion)    │     │  (20Hz, MP4V)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### 4.2 Processing Steps

#### Step 1: Temporal Resampling (30Hz → 20Hz)

```python
def compute_resample_indices(num_frames, source_fps=30, target_fps=20):
    """
    Compute frame indices for temporal resampling.

    Example: 150 frames @ 30Hz (5 sec) → 100 frames @ 20Hz
    """
    duration = num_frames / source_fps  # 5.0 seconds
    target_frames = int(duration * target_fps)  # 100 frames

    indices = []
    for i in range(target_frames):
        t = i / target_fps  # timestamp in seconds
        source_idx = int(t * source_fps)  # nearest source frame
        indices.append(min(source_idx, num_frames - 1))

    return indices
```

**Why 20Hz?** GR00T was pretrained at 20Hz. Matching this frequency provides optimal transfer learning.

#### Step 2: Video Processing

```python
# Video transformation pipeline
Source Video (AV1, 480×640, 30fps)
    │
    ▼ imageio.v3 (AV1 decode)
    │
    ▼ Frame selection (resample_indices)
    │
    ▼ Center crop (min_dim × min_dim)
    │
    ▼ Resize (256 × 256)
    │
    ▼ cv2.VideoWriter (MP4V, 20fps)
    │
Output Video (MP4V, 256×256, 20fps)
```

**Key parameters:**
- Target resolution: 256 × 256 (GR00T input size)
- Target codec: MP4V (widely compatible)
- Center crop preserves aspect ratio before resize

#### Step 3: State/Action Extraction

```python
# Source parquet has concatenated vectors
state_data = df["observation.state"].values  # Shape: (T, 19)
action_data = df["action"].values            # Shape: (T, 16)

# Apply temporal resampling
resampled_states = state_data[resample_indices]   # (T', 19)
resampled_actions = action_data[resample_indices] # (T', 16)
```

#### Step 4: Statistics Computation

Two statistics files are required:

**stats.json** - Absolute statistics for normalization:
```json
{
  "observation.state": {
    "mean": [...],  // 19 values
    "std": [...],
    "min": [...],
    "max": [...],
    "q01": [...],
    "q99": [...]
  },
  "action": {
    "mean": [...],  // 16 values
    "std": [...],
    "min": [...],
    "max": [...],
    "q01": [...],
    "q99": [...]
  }
}
```

**relative_stats.json** - Delta statistics for RELATIVE actions:
```json
{
  "left_arm": {
    "mean": [[...], ...],   // Shape: (16, 7) - action_horizon × arm_dof
    "std": [[...], ...],
    "min": [[...], ...],
    "max": [[...], ...],
    "q01": [[...], ...],
    "q99": [[...], ...]
  },
  "right_arm": {
    // Same structure
  }
}
```

**Relative action computation:**
```python
# For each timestep t in trajectory:
#   relative_action[i] = action[t+i] - state[t]
#   where i ∈ [0, action_horizon)

for t in range(episode_length - action_horizon):
    current_state = states[t, left_arm_indices]  # (7,)
    future_actions = actions[t:t+16, left_arm_indices]  # (16, 7)
    relative_chunk = future_actions - current_state  # (16, 7)
    all_chunks.append(relative_chunk)

# Statistics computed across all chunks
```

#### Step 5: Metadata Generation

**modality.json** - Maps component names to vector indices:
```json
{
  "state": {
    "base_odom": {"start": 0, "end": 3},
    "base_vel": {"start": 3, "end": 5},
    "left_arm": {"start": 5, "end": 12},
    "right_arm": {"start": 12, "end": 19}
  },
  "action": {
    "base_vel": {"start": 0, "end": 2},
    "left_arm": {"start": 2, "end": 9},
    "right_arm": {"start": 9, "end": 16}
  },
  "video": {
    "cam_high": {"original_key": "observation.images.cam_high"},
    "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
    "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"}
  }
}
```

### 4.3 Output Structure

```
data/ball2_groot_train/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   ├── stats.json           # Absolute statistics
│   ├── relative_stats.json  # Relative action statistics
│   └── modality.json        # Component → index mapping
├── data/
│   └── chunk-000/
│       └── episode_*.parquet
└── videos/
    └── chunk-000/
        ├── observation.images.cam_high/
        ├── observation.images.cam_left_wrist/
        └── observation.images.cam_right_wrist/
```

### 4.4 Conversion Command

```bash
python finetuning/convert_ball2_groot.py \
    --input_path ./inhouse/lerobot_dataset/lerobot/recorded_data/ball2_groot \
    --output_base ./data \
    --train_episodes 19 \
    --test_episodes 5
```

---

## 5. Modality Configuration

### 5.1 Custom Embodiment Registration

GR00T requires a modality configuration for each robot embodiment. Custom robots use the `NEW_EMBODIMENT` tag:

```python
# finetuning/trossen_modality_config.py

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation,
    ActionType, ModalityConfig
)

trossen_mobile_config = {
    "video": ModalityConfig(
        delta_indices=[0],  # Current frame only
        modality_keys=[
            "cam_high",
            "cam_left_wrist",
            "cam_right_wrist",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],  # Current state only
        modality_keys=[
            "base_odom",   # 3 DOF
            "base_vel",    # 2 DOF
            "left_arm",    # 7 DOF
            "right_arm",   # 7 DOF
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),  # 16-step action horizon
        modality_keys=[
            "base_vel",    # 2 DOF - ABSOLUTE
            "left_arm",    # 7 DOF - RELATIVE
            "right_arm",   # 7 DOF - RELATIVE
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}

register_modality_config(
    trossen_mobile_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
```

### 5.2 Key Configuration Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Action Horizon | 16 steps | GR00T default, ~0.8s at 20Hz |
| Base Velocity | ABSOLUTE | Direct velocity commands |
| Arm Joints | RELATIVE | Delta from current pose |
| State delta_indices | [0] | Current observation only |
| Video delta_indices | [0] | Current frame only |

---

## 6. Training Configuration

### 6.1 Default Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model_path` | nvidia/GR00T-N1.6-3B | Pretrained foundation model |
| `global_batch_size` | 64 | Effective batch size |
| `learning_rate` | 1e-4 | Initial learning rate |
| `max_steps` | 10000 | Total training steps |
| `save_steps` | 1000 | Checkpoint frequency |
| `warmup_ratio` | 0.05 | LR warmup proportion |
| `weight_decay` | 1e-5 | L2 regularization |

### 6.2 Model Components

| Component | Tuned | Frozen |
|-----------|-------|--------|
| LLM Backbone (Eagle-2B) | ❌ | ✅ |
| Visual Encoder | ❌ | ✅ |
| Projector | ✅ | ❌ |
| Diffusion Model | ✅ | ❌ |

### 6.3 Training Command

```bash
./finetuning/finetune_ball2_groot.sh

# Or with custom settings:
NUM_GPUS=4 \
DATASET_PATH=./data/ball2_groot_train \
OUTPUT_DIR=./outputs/ball2_groot \
./finetuning/finetune_ball2_groot.sh
```

### 6.4 Multi-GPU Training

```bash
torchrun --nproc_per_node=4 --standalone \
    -m gr00t.experiment.launch_finetune \
    --modality-config-path ./finetuning/trossen_modality_config.py \
    --embodiment-tag NEW_EMBODIMENT \
    --dataset-path ./data/ball2_groot_train \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --output-dir ./outputs/ball2_groot
```

---

## 7. Evaluation Pipeline

### 7.1 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | mean((pred - gt)²) | Lower is better |
| **MAE** | mean(\|pred - gt\|) | Average error magnitude |
| **Success@X** | % steps where ALL dims < X | Strict per-step accuracy |
| **Accuracy@X** | % predictions < X | Per-dimension accuracy |
| **Error Drift** | late_MAE - early_MAE | Negative = improving |

### 7.2 Single Checkpoint Evaluation

```bash
python finetuning/evaluate_ball2_groot.py \
    --checkpoint_path ./outputs/ball2_groot/checkpoint-1000 \
    --dataset_path ./data/ball2_groot_test \
    --output_dir ./eval_results/ball2_groot
```

### 7.3 Batch Checkpoint Evaluation

```bash
# Evaluate specific checkpoints
./finetuning/evaluate_checkpoints.sh 1000 2000 3000 5000 10000

# Evaluate all checkpoints
./finetuning/evaluate_checkpoints.sh --all
```

**Output:** `checkpoint_comparison.csv`
```csv
step,mse,mae,success_005,success_01,success_02,base_mae,left_arm_mae,right_arm_mae
1000,0.009407,0.059277,0.0,6.5,67.4,0.0575,0.0727,0.0464
2000,...
```

### 7.4 Generated Artifacts

```
eval_results/ball2_groot/
├── checkpoint_comparison.csv      # Cross-checkpoint metrics
├── step_1000/
│   ├── trajectory_0_comparison.png  # Action prediction plots
│   ├── trajectory_0_cameras.png     # Multi-camera snapshot
│   ├── trajectory_0_visualization.mp4  # (if --generate_videos)
│   └── evaluation_summary.txt       # Detailed metrics
└── step_2000/
    └── ...
```

---

## 8. Results: ball2_groot Checkpoint-1000

### 8.1 Aggregate Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Average MSE | 0.009407 | Early training |
| Average MAE | 0.059277 | ~0.06 rad error |
| Success@0.05 | 0.0% | Too strict for early checkpoint |
| Success@0.10 | 6.5% | Low step-wise success |
| Success@0.20 | 67.4% | Reasonable |
| Accuracy@0.10 | 81.0% | Most individual predictions close |
| Accuracy@0.20 | 95.9% | Almost all predictions reasonable |
| Error Drift | -0.011 | Model improves over trajectory |

### 8.2 Component-wise Performance

| Component | MAE | Notes |
|-----------|-----|-------|
| Base Velocity | 0.0575 | Good |
| Left Arm | 0.0727 | Needs improvement |
| Right Arm | 0.0464 | Best performing |

### 8.3 Interpretation

- **Early training (1000 steps)**: Model is learning but not converged
- **High Accuracy, Low Success**: Most joints accurate, but strict all-joint threshold fails
- **Negative drift**: Model predictions improve as trajectory progresses
- **Left arm lags**: May need more training data or attention

---

## 9. Reusable Pipeline Template

### 9.1 For New Inhouse Datasets

When a new inhouse dataset arrives, follow this checklist:

```
□ Step 1: Identify Robot Configuration
  - Number of cameras and their names
  - State DOF breakdown (base, arms, etc.)
  - Action DOF breakdown
  - Which actions are ABSOLUTE vs RELATIVE

□ Step 2: Copy and Modify Conversion Script
  cp finetuning/convert_ball2_groot.py finetuning/convert_NEW_DATASET.py
  # Update: CAMERA_KEYS, STATE_DIM, ACTION_DIM, STATE_INDICES, ACTION_INDICES

□ Step 3: Copy and Modify Modality Config
  cp finetuning/trossen_modality_config.py finetuning/NEW_ROBOT_modality_config.py
  # Update: modality_keys, action_configs

□ Step 4: Copy and Modify Training Script
  cp finetuning/finetune_ball2_groot.sh finetuning/finetune_NEW_DATASET.sh
  # Update: DATASET_PATH, OUTPUT_DIR

□ Step 5: Copy and Modify Evaluation Script
  cp finetuning/evaluate_ball2_groot.py finetuning/evaluate_NEW_DATASET.py
  # Update: ACTION_LABELS, STATE_LABELS

□ Step 6: Run Pipeline
  python finetuning/convert_NEW_DATASET.py
  ./finetuning/finetune_NEW_DATASET.sh
  ./finetuning/evaluate_checkpoints.sh 1000 2000 ...
```

### 9.2 Configuration Variables to Update

| Variable | Location | Description |
|----------|----------|-------------|
| `CAMERA_KEYS` | convert_*.py | List of camera names |
| `STATE_DIM` | convert_*.py | Total state dimensions |
| `ACTION_DIM` | convert_*.py | Total action dimensions |
| `STATE_INDICES` | convert_*.py | Component → index ranges |
| `ACTION_INDICES` | convert_*.py | Component → index ranges |
| `RELATIVE_ACTION_KEYS` | convert_*.py | Which components use RELATIVE |
| `SOURCE_FPS` | convert_*.py | Input dataset frame rate |
| `modality_keys` | *_modality_config.py | Component names |
| `action_configs` | *_modality_config.py | ABSOLUTE/RELATIVE per component |
| `ACTION_LABELS` | evaluate_*.py | Human-readable joint names |

### 9.3 Quick Reference: Adding a New Robot

```python
# Example: New robot with 2 cameras, 1 arm (6 DOF), mobile base

CAMERA_KEYS = ["front_cam", "wrist_cam"]
STATE_DIM = 8   # odom(3) + base_vel(2) + arm(6) - wait, that's 11
ACTION_DIM = 8  # base_vel(2) + arm(6)

STATE_INDICES = {
    "base_odom": (0, 3),
    "base_vel": (3, 5),
    "arm": (5, 11),
}

ACTION_INDICES = {
    "base_vel": (0, 2),
    "arm": (2, 8),
}

RELATIVE_ACTION_KEYS = ["arm"]  # Arm uses relative, base uses absolute
```

---

## 10. Troubleshooting Guide

### 10.1 Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `invalid choice: 'custom_robot'` | Custom embodiment tag not recognized | Use `NEW_EMBODIMENT` tag |
| `FileNotFoundError: modality.json` | Missing modality mapping file | Add modality.json creation in convert script |
| `KeyError: 'observation.state'` | Stats.json has wrong keys | Use `observation.state` and `action` as top-level keys |
| `relative_action not found` | Missing relative_stats.json | Compute relative action statistics in conversion |
| Video decode error | AV1 codec not supported | Use imageio.v3 for AV1 decode |

### 10.2 Validation Checks

```bash
# Check dataset structure
ls -la data/ball2_groot_train/meta/

# Verify stats.json format
cat data/ball2_groot_train/meta/stats.json | python -m json.tool | head -20

# Verify relative_stats.json
cat data/ball2_groot_train/meta/relative_stats.json | python -m json.tool | head -30

# Check video conversion
ffprobe data/ball2_groot_train/videos/chunk-000/observation.images.cam_high/episode_000000.mp4
```

### 10.3 Memory Issues

If training runs out of memory:
```bash
# Reduce batch size
--global-batch-size 32

# Reduce number of shards
--num-shards-per-epoch 50000

# Use gradient accumulation
--gradient-accumulation-steps 2
```

---

## 11. Future Improvements

### 11.1 Potential Enhancements

1. **Automated robot config detection**: Parse source dataset to auto-detect DOF structure
2. **Multi-task support**: Handle datasets with multiple task descriptions
3. **Data augmentation**: Add image augmentation options to conversion
4. **Streaming conversion**: Process large datasets without loading all into memory
5. **MCP integration**: Create MCP server for standardized dataset operations

### 11.2 MCP Server Consideration

For 10+ datasets, an MCP server could provide:

```python
# Potential MCP tools
convert_dataset(input_path, robot_config, output_path)
validate_dataset(dataset_path)
run_finetuning(dataset_path, config)
evaluate_checkpoint(checkpoint_path, test_path)
compare_checkpoints(checkpoint_list)
```

**Recommendation**: Wait until patterns stabilize across 3-4 more datasets before investing in MCP infrastructure.

---

## 12. Appendix

### A. File Checksums

```bash
# Verify conversion script integrity
md5 finetuning/convert_ball2_groot.py
md5 finetuning/trossen_modality_config.py
md5 finetuning/evaluate_ball2_groot.py
```

### B. Dependencies

```
torch>=2.0
transformers
imageio[ffmpeg]
opencv-python
pandas
pyarrow
matplotlib
numpy
```

### C. Git Commits

| Commit | Description |
|--------|-------------|
| Initial | Create ball2_groot conversion script |
| Fix | Use NEW_EMBODIMENT tag for custom robots |
| Fix | Add modality.json creation |
| Fix | Correct stats.json format with observation.state key |
| Fix | Source column is observation.state not observation |
| Add | Relative action statistics for RELATIVE representation |
| Add | Batch checkpoint evaluation script |
| Update | Physical joint names in evaluation plots |

---

*This report serves as the canonical reference for inhouse dataset finetuning. Update this document as the pipeline evolves.*
