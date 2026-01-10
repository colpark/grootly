# GR00T Finetuning Guide: ALOHA to GR00T Pipeline

**Complete guide from dataset download to finetuning the GR00T N1.6 model**

---

## Overview

This guide covers the entire pipeline for finetuning GR00T N1.6 on the ALOHA Transfer Cube dataset (bimanual handover task). The ALOHA dataset was NOT used in GR00T pretraining, making it ideal for testing transfer learning capabilities.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FINETUNING PIPELINE OVERVIEW                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │  STEP 1  │    │  STEP 2  │    │  STEP 3  │    │  STEP 4  │             │
│   │ Download │ →  │ Convert  │ →  │ Register │ →  │ Finetune │             │
│   │  ALOHA   │    │ to GR00T │    │Embodiment│    │  Model   │             │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│        │               │               │               │                    │
│        ▼               ▼               ▼               ▼                    │
│   HuggingFace     LeRobot v2     Modality       nvidia/GR00T               │
│   Datasets        Format         Config         -N1.6-3B                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Download ALOHA Dataset

### 1.1 Dataset Information

| Property | Value |
|----------|-------|
| **Dataset ID** | `lerobot/aloha_sim_transfer_cube_human` |
| **Task** | Cube handover between two robot arms |
| **Robot** | ALOHA (2x ViperX arms) |
| **Episodes** | 50 |
| **Format** | LeRobot v2 |
| **License** | Apache 2.0 |

### 1.2 Download Commands

```bash
# Option 1: Using HuggingFace CLI
pip install huggingface_hub

# Download dataset
huggingface-cli download lerobot/aloha_sim_transfer_cube_human \
    --repo-type dataset \
    --local-dir ./data/aloha_transfer_cube

# Option 2: Using Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='lerobot/aloha_sim_transfer_cube_human',
    repo_type='dataset',
    local_dir='./data/aloha_transfer_cube'
)
"
```

### 1.3 Verify Download

```bash
# Check directory structure
ls -la ./data/aloha_transfer_cube/

# Expected structure:
# aloha_transfer_cube/
# ├── data/
# │   └── train-00000-of-00001.parquet
# ├── videos/
# │   └── observation.images.top/
# │       └── episode_000000.mp4 ... episode_000049.mp4
# └── meta/
#     └── info.json
```

---

## Step 2: Convert ALOHA to GR00T Format

### 2.1 Why Conversion is Needed

| Property | ALOHA (Source) | GR00T GR1 (Target) | Conversion |
|----------|---------------|-------------------|------------|
| **Control FPS** | 50 Hz | 20 Hz | Downsample 2.5× |
| **State DOF** | 14 | 44 | Map + pad zeros |
| **Action DOF** | 14 | 44 | Map + pad zeros |
| **Gripper** | 1 DOF | 6 DOF hand | Expand |
| **Image Size** | 480×640 | 256×256 | Crop + resize |
| **Video Codec** | AV1 | H.264 | Re-encode |
| **Camera Key** | `observation.images.top` | `ego_view` | Rename |

### 2.2 Dimension Mapping

```
ALOHA (14 DOF)                    GR00T GR1 (44 DOF)
──────────────                    ─────────────────
Left Arm [0:6]      ────────→     left_arm [0:7] (pad 1 zero)
Left Gripper [6]    ────────→     left_hand [7:13] (expand to 6)
                                  left_leg [13:19] (zeros)
                                  neck [19:22] (zeros)
Right Arm [7:13]    ────────→     right_arm [22:29] (pad 1 zero)
Right Gripper [13]  ────────→     right_hand [29:35] (expand to 6)
                                  right_leg [35:41] (zeros)
                                  waist [41:44] (zeros)
```

### 2.3 Run Conversion

```bash
cd Isaac-GR00T

# Run conversion script
python finetuning/convert_aloha_to_groot.py \
    --input-dir ./data/aloha_transfer_cube \
    --output-dir ./data/aloha_groot_format \
    --target-fps 20 \
    --target-resolution 256
```

### 2.4 Conversion Details

The conversion script performs these transformations:

```python
# 1. Temporal Resampling (50 Hz → 20 Hz)
target_indices = np.linspace(0, num_frames-1, int(num_frames * 20/50)).astype(int)
resampled_data = original_data[target_indices]

# 2. State/Action Dimension Mapping (14 → 44)
groot_state = np.zeros(44, dtype=np.float32)
groot_state[0:6] = aloha_state[0:6]      # left arm
groot_state[7:13] = expand_gripper(aloha_state[6])  # left hand
groot_state[22:28] = aloha_state[7:13]   # right arm
groot_state[29:35] = expand_gripper(aloha_state[13])  # right hand

# 3. Gripper Expansion (1 DOF → 6 DOF)
def expand_gripper(gripper_value):
    return np.full(6, gripper_value, dtype=np.float32)

# 4. Image Transformation (480×640 → 256×256)
# Center crop to square, then resize
min_dim = min(height, width)
cropped = img[center_h:center_h+min_dim, center_w:center_w+min_dim]
resized = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_AREA)

# 5. Video Re-encoding (AV1 → H.264)
# Using imageio/ffmpeg to re-encode video streams
```

### 2.5 Verify Conversion

```bash
# Check output structure
ls -la ./data/aloha_groot_format/

# Expected structure:
# aloha_groot_format/
# ├── data/
# │   └── chunk-000/
# │       └── episode_000000.parquet ... episode_000049.parquet
# ├── videos/
# │   └── chunk-000/
# │       └── ego_view/
# │           └── episode_000000.mp4 ... episode_000049.mp4
# └── meta/
#     ├── info.json
#     ├── modality.json
#     ├── episodes.jsonl
#     ├── tasks.jsonl
#     └── stats.json

# Verify parquet schema
python -c "
import pyarrow.parquet as pq
table = pq.read_table('./data/aloha_groot_format/data/chunk-000/episode_000000.parquet')
print('Columns:', table.column_names)
print('Rows:', table.num_rows)
"
```

---

## Step 3: Register Embodiment Configuration

### 3.1 Understanding Embodiment Tags

GR00T uses embodiment tags to identify robot configurations and their modality requirements:

| Tag | Robot | Description |
|-----|-------|-------------|
| `GR1` | Fourier GR-1 | Pretrained bimanual humanoid |
| `ROBOCASA_PANDA_OMRON` | Panda | Pretrained single arm |
| `NEW_EMBODIMENT` | Custom | For new robots |

### 3.2 Create Modality Configuration

Create a file `finetuning/aloha_modality_config.py`:

```python
"""Modality configuration for ALOHA Transfer Cube dataset mapped to GR1 format."""

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.configs.data.embodiment_configs import register_modality_config

# Define ALOHA modality config (mapped to GR1-compatible format)
ALOHA_TRANSFER_CUBE_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],  # Renamed from observation.images.top
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",    # 7 DOF (6 from ALOHA + 1 padded)
            "left_hand",   # 6 DOF (expanded from 1 gripper)
            "right_arm",   # 7 DOF (6 from ALOHA + 1 padded)
            "right_hand",  # 6 DOF (expanded from 1 gripper)
            "waist",       # 3 DOF (zeros, unused)
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),  # 16 action prediction steps
        modality_keys=[
            "left_arm",
            "left_hand",
            "right_arm",
            "right_hand",
            "waist",
        ],
        action_configs=[
            # left_arm - relative joint control
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_hand - absolute gripper control
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_arm - relative joint control
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_hand - absolute gripper control
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # waist - absolute (zeros, unused)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
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

# Register the configuration
register_modality_config(ALOHA_TRANSFER_CUBE_CONFIG, EmbodimentTag.NEW_EMBODIMENT)

print("Registered ALOHA Transfer Cube modality config as NEW_EMBODIMENT")
```

### 3.3 Alternative: Use Existing GR1 Tag

If you want to leverage GR1 pretrained weights more directly, you can map ALOHA data to the exact GR1 format without registering a new embodiment:

```python
# Use GR1 embodiment tag directly
# The converted data already matches GR1's expected format
embodiment_tag = EmbodimentTag.GR1
```

**Pros**: Can leverage GR1 pretrained weights directly
**Cons**: Unused dimensions (legs, neck) will be zeros

---

## Step 4: Finetune the Model

### 4.1 Finetuning Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GR00T N1.6 MODEL ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────┐                                                        │
│   │  Vision Encoder │  ← tune_visual=False (frozen by default)              │
│   │ Eagle-Block2A-2B│                                                        │
│   └────────┬────────┘                                                        │
│            │                                                                 │
│            ▼                                                                 │
│   ┌─────────────────┐                                                        │
│   │   Projector     │  ← tune_projector=True (finetuned)                    │
│   │   Layers        │                                                        │
│   └────────┬────────┘                                                        │
│            │                                                                 │
│            ▼                                                                 │
│   ┌─────────────────┐                                                        │
│   │  Language Model │  ← tune_llm=False (frozen by default)                 │
│   │     (LLM)       │                                                        │
│   └────────┬────────┘                                                        │
│            │                                                                 │
│            ▼                                                                 │
│   ┌─────────────────┐                                                        │
│   │ Diffusion Head  │  ← tune_diffusion_model=True (finetuned)              │
│   │ (Action Decoder)│                                                        │
│   └─────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Finetuning Configuration

Key parameters in `FinetuneConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model_path` | - | Path to pretrained model (nvidia/GR00T-N1.6-3B) |
| `dataset_path` | - | Path to converted dataset |
| `embodiment_tag` | - | NEW_EMBODIMENT or GR1 |
| `tune_llm` | False | Freeze LLM backbone |
| `tune_visual` | False | Freeze vision encoder |
| `tune_projector` | True | Finetune projector layers |
| `tune_diffusion_model` | True | Finetune action decoder |
| `learning_rate` | 1e-4 | Initial learning rate |
| `global_batch_size` | 64 | Total batch size |
| `max_steps` | 10000 | Training steps |
| `save_steps` | 1000 | Checkpoint frequency |

### 4.3 Launch Finetuning

```bash
cd Isaac-GR00T

# Option 1: Using GR1 embodiment (recommended for transfer learning)
python -m gr00t.experiment.launch_finetune \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path ./data/aloha_groot_format \
    --embodiment-tag GR1 \
    --output-dir ./outputs/aloha_finetune \
    --learning-rate 1e-4 \
    --global-batch-size 32 \
    --max-steps 5000 \
    --save-steps 500 \
    --num-gpus 1

# Option 2: Using custom embodiment config
python -m gr00t.experiment.launch_finetune \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path ./data/aloha_groot_format \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path ./finetuning/aloha_modality_config.py \
    --output-dir ./outputs/aloha_finetune \
    --learning-rate 1e-4 \
    --global-batch-size 32 \
    --max-steps 5000 \
    --save-steps 500 \
    --num-gpus 1
```

### 4.4 Multi-GPU Training

```bash
# For multi-GPU training, adjust num_gpus and batch size
python -m gr00t.experiment.launch_finetune \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path ./data/aloha_groot_format \
    --embodiment-tag GR1 \
    --output-dir ./outputs/aloha_finetune \
    --learning-rate 1e-4 \
    --global-batch-size 64 \
    --gradient-accumulation-steps 2 \
    --max-steps 10000 \
    --num-gpus 4
```

### 4.5 Monitor Training

```bash
# Option 1: Watch output directory
watch -n 10 ls -la ./outputs/aloha_finetune/

# Option 2: Enable Weights & Biases
python -m gr00t.experiment.launch_finetune \
    ... \
    --use-wandb True

# Then view at: https://wandb.ai/<your-project>/finetune-gr00t-n1d6
```

---

## Step 5: Evaluate Finetuned Model

### 5.1 Start Inference Server

```bash
# Use your finetuned checkpoint
python gr00t/eval/run_gr00t_server.py \
    --model-path ./outputs/aloha_finetune/checkpoint-5000 \
    --embodiment-tag GR1 \
    --use-sim-policy-wrapper
```

### 5.2 Run Evaluation

For ALOHA evaluation, you'll need to set up the ALOHA simulation environment (separate from RoboCasa). The finetuned model can also be evaluated on GR1 tasks to test transfer:

```bash
# Test on GR1 Plate-to-Plate (transfer evaluation)
gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python \
    gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --env_name gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env \
    --max_episode_steps 720 \
    --n_action_steps 8
```

---

## Complete Pipeline Script

```bash
#!/bin/bash
# complete_finetuning_pipeline.sh

set -e  # Exit on error

echo "=== Step 1: Download ALOHA Dataset ==="
pip install huggingface_hub
huggingface-cli download lerobot/aloha_sim_transfer_cube_human \
    --repo-type dataset \
    --local-dir ./data/aloha_transfer_cube

echo "=== Step 2: Convert to GR00T Format ==="
python finetuning/convert_aloha_to_groot.py \
    --input-dir ./data/aloha_transfer_cube \
    --output-dir ./data/aloha_groot_format \
    --target-fps 20 \
    --target-resolution 256

echo "=== Step 3: Verify Conversion ==="
python -c "
import pyarrow.parquet as pq
import os

data_dir = './data/aloha_groot_format/data/chunk-000'
files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
print(f'Found {len(files)} episode files')

table = pq.read_table(os.path.join(data_dir, files[0]))
print(f'Columns: {table.column_names}')
print('Conversion verified!')
"

echo "=== Step 4: Launch Finetuning ==="
python -m gr00t.experiment.launch_finetune \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path ./data/aloha_groot_format \
    --embodiment-tag GR1 \
    --output-dir ./outputs/aloha_finetune \
    --learning-rate 1e-4 \
    --global-batch-size 32 \
    --max-steps 5000 \
    --save-steps 500 \
    --num-gpus 1

echo "=== Finetuning Complete ==="
echo "Checkpoints saved to: ./outputs/aloha_finetune/"
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `global_batch_size` or use `gradient_accumulation_steps` |
| `Embodiment tag not found` | Ensure modality config is loaded before training |
| `Dataset loading error` | Verify parquet files have correct schema |
| `Video decoding error` | Re-encode videos with H.264 codec |
| `Shape mismatch` | Check state/action dimensions match 44 DOF |

### Resource Requirements

| Configuration | GPU Memory | Training Time (5K steps) |
|--------------|------------|--------------------------|
| 1x A100 (80GB) | ~40GB | ~2 hours |
| 1x RTX 4090 (24GB) | ~20GB (batch 16) | ~4 hours |
| 4x A100 (80GB) | ~30GB each | ~30 minutes |

---

## Summary

1. **Download**: Get ALOHA dataset from HuggingFace
2. **Convert**: Transform to GR00T format (FPS, dimensions, images)
3. **Register**: Create modality config for the embodiment
4. **Finetune**: Run `launch_finetune.py` with appropriate settings
5. **Evaluate**: Test finetuned model on target tasks

The key insight is that ALOHA's bimanual handover task transfers well to GR1's bimanual manipulation because both involve coordinated two-arm movements, even though the specific robots and joint configurations differ.
