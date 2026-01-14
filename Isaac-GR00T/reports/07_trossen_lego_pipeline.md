# Trossen LEGO Manipulation Finetuning Pipeline

**Date**: January 2026
**Task**: LEGO Manipulation (Bimanual Mobile Manipulation)
**Robot**: Trossen AI Mobile (Bimanual Mobile Robot)
**Base Model**: nvidia/GR00T-N1.6-3B
**Embodiment Tag**: NEW_EMBODIMENT (custom robot)
**Source**: [HuggingFace - TrossenRoboticsCommunity/trossen_ai_mobile_lego](https://huggingface.co/datasets/TrossenRoboticsCommunity/trossen_ai_mobile_lego)

---

## 1. Executive Summary

This report documents the finetuning pipeline for the `trossen_lego` dataset - a LEGO manipulation task performed by the Trossen AI Mobile robot. This is the **third dataset** in our Trossen robot finetuning series, following ball2_groot and plug_stacking.

### Key Information

| Property | Value |
|----------|-------|
| **Source** | Hugging Face Hub |
| **Repo ID** | `TrossenRoboticsCommunity/trossen_ai_mobile_lego` |
| **Episodes** | 52 |
| **Total Frames** | 29,641 |
| **Source FPS** | 30 Hz |
| **Target FPS** | 20 Hz (GR00T standard) |
| **State DOF** | 19 |
| **Action DOF** | 16 |
| **Cameras** | 3 (cam_high, cam_left_wrist, cam_right_wrist) |

### Pipeline Files

| File | Purpose |
|------|---------|
| `finetuning/convert_lego.py` | Dataset conversion script |
| `finetuning/trossen_modality_config.py` | Robot modality configuration (shared) |
| `finetuning/finetune_lego.sh` | Training launch script |
| `finetuning/evaluate_lego.py` | Evaluation with metrics & plots |
| `finetuning/evaluate_lego_checkpoints.sh` | Batch checkpoint evaluation |

### Compatibility Note

This dataset uses the **identical robot configuration** as ball2_groot and plug_stacking:
- Same Trossen AI Mobile platform
- Same camera setup (3 cameras)
- Same state/action dimensions (19/16 DOF)
- Same modality config (`trossen_modality_config.py`)

---

## 2. Dataset Details

### 2.1 Source Dataset Structure

```
trossen_ai_mobile_lego/
├── meta/
│   ├── info.json           # Dataset metadata
│   ├── episodes.jsonl      # Per-episode info
│   └── tasks.jsonl         # Task descriptions
├── data/
│   └── chunk-000/
│       └── episode_*.parquet   # 52 episodes
└── videos/
    └── chunk-000/
        ├── observation.images.cam_high/
        ├── observation.images.cam_left_wrist/
        └── observation.images.cam_right_wrist/
```

### 2.2 State Space (19 DOF)

| Index | Component | Description |
|-------|-----------|-------------|
| 0 | odom_x | Base X position (m) |
| 1 | odom_y | Base Y position (m) |
| 2 | odom_theta | Base orientation (rad) |
| 3 | linear_vel | Base linear velocity (m/s) |
| 4 | angular_vel | Base angular velocity (rad/s) |
| 5-11 | left_joint_0-6 | Left arm joints (rad) |
| 12-18 | right_joint_0-6 | Right arm joints (rad) |

### 2.3 Action Space (16 DOF)

| Index | Component | Representation |
|-------|-----------|----------------|
| 0 | linear_vel | ABSOLUTE |
| 1 | angular_vel | ABSOLUTE |
| 2-8 | left_joint_0-6 | RELATIVE |
| 9-15 | right_joint_0-6 | RELATIVE |

### 2.4 Camera Configuration

| Camera | Resolution | Codec | Description |
|--------|------------|-------|-------------|
| cam_high | 480×640 | AV1 | Central overhead view |
| cam_left_wrist | 480×640 | AV1 | Left arm wrist camera |
| cam_right_wrist | 480×640 | AV1 | Right arm wrist camera |

---

## 3. Pipeline Steps

### 3.1 Step 1: Download Dataset

```bash
# Download from Hugging Face
python scripts/download_hf_dataset.py TrossenRoboticsCommunity/trossen_ai_mobile_lego --name trossen_lego

# Dataset will be saved to:
# ./inhouse/lerobot_dataset/lerobot/recorded_data/trossen_lego/
```

### 3.2 Step 2: Convert to GR00T Format

```bash
# Convert with 80/20 train/test split
python finetuning/convert_lego.py \
    --input_path ./inhouse/lerobot_dataset/lerobot/recorded_data/trossen_lego \
    --output_base ./data \
    --train_episodes 42 \
    --test_episodes 10
```

**Conversion Operations:**
- Temporal resampling: 30 Hz → 20 Hz
- Image resizing: 480×640 → 256×256
- Video re-encoding: AV1 → H.264
- Statistics computation: stats.json, relative_stats.json
- Modality config generation: modality.json

**Output Structure:**
```
./data/
├── lego_train/
│   ├── meta/
│   │   ├── info.json
│   │   ├── episodes.jsonl
│   │   ├── tasks.jsonl
│   │   ├── stats.json
│   │   ├── relative_stats.json
│   │   └── modality.json
│   ├── data/
│   │   └── chunk-000/
│   │       └── episode_*.parquet
│   └── videos/
│       └── chunk-000/
│           └── observation.images.*/
└── lego_test/
    └── (same structure)
```

### 3.3 Step 3: Validate Conversion

```bash
# Check required files exist
ls ./data/lego_train/meta/

# Verify stats.json format
cat ./data/lego_train/meta/stats.json | python -c "
import json, sys
stats = json.load(sys.stdin)
print('State keys:', list(stats['observation.state'].keys()))
print('Action keys:', list(stats['action'].keys()))
"

# Check video files
ls ./data/lego_train/videos/chunk-000/
```

### 3.4 Step 4: Run Finetuning

```bash
# Single GPU
./finetuning/finetune_lego.sh

# Multi-GPU (4 GPUs)
./finetuning/finetune_lego.sh 4

# Custom configuration
DATASET_PATH=./data/lego_train \
OUTPUT_DIR=./outputs/lego \
./finetuning/finetune_lego.sh 4
```

**Default Training Configuration:**
| Parameter | Value |
|-----------|-------|
| Base Model | nvidia/GR00T-N1.6-3B |
| Learning Rate | 1e-4 |
| Global Batch Size | 64 |
| Max Steps | 10,000 |
| Save Steps | 1,000 |
| Warmup Ratio | 0.05 |
| Weight Decay | 1e-5 |

### 3.5 Step 5: Evaluate Checkpoints

```bash
# Evaluate single checkpoint
python finetuning/evaluate_lego.py \
    --checkpoint_path ./outputs/lego/checkpoint-5000 \
    --dataset_path ./data/lego_test \
    --output_dir ./eval_results/lego

# Batch evaluate all checkpoints
./finetuning/evaluate_lego_checkpoints.sh --all

# Evaluate specific checkpoints
./finetuning/evaluate_lego_checkpoints.sh 1000 2000 5000 10000
```

### 3.6 Step 6: Generate Training Report

```bash
python finetuning/plot_checkpoint_results.py \
    --eval_dir ./eval_results/lego \
    --checkpoint_dir ./outputs/lego

# Opens: ./eval_results/lego/training_report.html
```

---

## 4. Comparison with Other Trossen Datasets

| Property | ball2_groot | plug_stacking | trossen_lego |
|----------|-------------|---------------|--------------|
| **Task** | Ball Transfer | Plug Stacking | LEGO Manipulation |
| **Source** | Inhouse | Inhouse | Hugging Face |
| **Episodes** | ~100 | 56 | 52 |
| **Frames** | ~20K | ~11K | ~30K |
| **FPS** | 30 Hz | 30 Hz | 30 Hz |
| **State DOF** | 19 | 19 | 19 |
| **Action DOF** | 16 | 16 | 16 |
| **Cameras** | 3 | 3 | 3 |
| **Modality Config** | trossen_modality_config.py | Same | Same |

---

## 5. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `huggingface_hub not found` | Missing dependency | `pip install huggingface_hub` |
| `Video decode error` | AV1 codec issues | Use `imageio.v3` for reading |
| `Shape mismatch` | DOF mismatch | Verify 19 state / 16 action dims |
| `relative_stats.json missing` | Conversion incomplete | Re-run conversion script |
| `CUDA OOM` | Batch size too large | Reduce global_batch_size |

### Verification Commands

```bash
# Check dataset info
cat ./data/lego_train/meta/info.json | python -m json.tool

# Check state/action dimensions
python -c "
import pandas as pd
df = pd.read_parquet('./data/lego_train/data/chunk-000/episode_000000.parquet')
print('State dim:', len(df['observation.state'].iloc[0]))
print('Action dim:', len(df['action'].iloc[0]))
"

# Check video resolution
ffprobe ./data/lego_train/videos/chunk-000/observation.images.cam_high/episode_000000.mp4 2>&1 | grep Stream
```

---

## 6. Training Progress

### 6.1 Training Status

| Stage | Status | Date | Notes |
|-------|--------|------|-------|
| Dataset Download | ⏳ Pending | - | - |
| Dataset Conversion | ⏳ Pending | - | - |
| Training | ⏳ Pending | - | - |
| Evaluation | ⏳ Pending | - | - |

### 6.2 Results

*(To be updated after training)*

| Checkpoint | MSE | MAE | Success@0.10 |
|------------|-----|-----|--------------|
| - | - | - | - |

### 6.3 Best Checkpoint

*(To be updated after evaluation)*

---

## 7. Changelog

| Date | Changes |
|------|---------|
| 2026-01-14 | Initial pipeline creation |

---

*Last Updated: January 2026*
*Reference Implementation: ball2_groot, plug_stacking (Trossen AI Mobile)*
