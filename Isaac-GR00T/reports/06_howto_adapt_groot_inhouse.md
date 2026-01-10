# How-To: Adapt GR00T for New Inhouse Dataset

**Purpose**: Quick reference guide for adapting GR00T finetuning pipeline to new inhouse robotics datasets.

**Time Required**: ~30 minutes for adaptation, hours for training

**Reference Implementation**: `ball2_groot` (Trossen AI Mobile)

---

## Quick Start Checklist

```
□ 1. Gather robot info (cameras, DOFs, action types)
□ 2. Create conversion script (copy & modify)
□ 3. Create modality config (copy & modify)
□ 4. Create training script (copy & modify)
□ 5. Create evaluation script (copy & modify)
□ 6. Run conversion
□ 7. Validate converted data
□ 8. Run finetuning
□ 9. Evaluate checkpoints
```

---

## Step 1: Gather Robot Information

Before writing any code, fill out this table:

```
Dataset Name: _______________
Robot Name: _______________
Source Path: ./inhouse/.../_______________
Source FPS: ___ Hz
Total Episodes: ___

CAMERAS:
  □ Camera 1: _______________
  □ Camera 2: _______________
  □ Camera 3: _______________
  (add more as needed)

STATE (observation.state):
  Component      | DOF | Index Range
  ---------------|-----|------------
  ______________ | ___ | ___:___
  ______________ | ___ | ___:___
  ______________ | ___ | ___:___
  ______________ | ___ | ___:___
  TOTAL          | ___ |

ACTION:
  Component      | DOF | Index Range | ABSOLUTE/RELATIVE
  ---------------|-----|-------------|------------------
  ______________ | ___ | ___:___     | _______________
  ______________ | ___ | ___:___     | _______________
  ______________ | ___ | ___:___     | _______________
  TOTAL          | ___ |

Train/Test Split: ___/___ episodes
```

### How to Find This Info

```bash
# Check source dataset structure
ls -la ./inhouse/.../YOUR_DATASET/

# Read dataset info
cat ./inhouse/.../YOUR_DATASET/meta/info.json | python -m json.tool

# Check episode count
cat ./inhouse/.../YOUR_DATASET/meta/info.json | grep total_episodes

# Check FPS
cat ./inhouse/.../YOUR_DATASET/meta/info.json | grep fps

# Check feature dimensions
cat ./inhouse/.../YOUR_DATASET/meta/info.json | grep -A5 features

# List cameras (look at video folders)
ls ./inhouse/.../YOUR_DATASET/videos/chunk-000/

# Check parquet columns
python -c "import pandas as pd; df = pd.read_parquet('./inhouse/.../YOUR_DATASET/data/chunk-000/episode_000000.parquet'); print(df.columns.tolist())"

# Check state/action dimensions
python -c "import pandas as pd; df = pd.read_parquet('./inhouse/.../YOUR_DATASET/data/chunk-000/episode_000000.parquet'); print('State dim:', len(df['observation.state'].iloc[0])); print('Action dim:', len(df['action'].iloc[0]))"
```

---

## Step 2: Create Conversion Script

### 2.1 Copy Template

```bash
cp finetuning/convert_ball2_groot.py finetuning/convert_YOUR_DATASET.py
```

### 2.2 Modify These Variables

Open `finetuning/convert_YOUR_DATASET.py` and update:

```python
# Line ~37-39: Frame rates
SOURCE_FPS = 30  # <-- Your source FPS
TARGET_FPS = 20  # Keep at 20 (GR00T standard)
TARGET_IMAGE_SIZE = 256  # Keep at 256

# Line ~42: Camera keys
CAMERA_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]  # <-- Your cameras

# Line ~45-46: DOF configuration
STATE_DIM = 19  # <-- Your total state dimensions
ACTION_DIM = 16  # <-- Your total action dimensions

# Line ~48: Action horizon (usually keep at 16)
ACTION_HORIZON = 16

# Line ~51-57: State index mapping
STATE_INDICES = {
    "component_1": (0, 3),    # <-- Your components
    "component_2": (3, 5),
    "left_arm": (5, 12),
    "right_arm": (12, 19),
}

# Line ~59-63: Action index mapping
ACTION_INDICES = {
    "component_1": (0, 2),    # <-- Your components
    "left_arm": (2, 9),
    "right_arm": (9, 16),
}

# Line ~66: Which actions use RELATIVE representation
RELATIVE_ACTION_KEYS = ["left_arm", "right_arm"]  # <-- Components with RELATIVE actions
```

### 2.3 Update Default Paths

```python
# In main() argparse section (~line 340):
parser.add_argument(
    "--input_path",
    type=str,
    default="./inhouse/.../YOUR_DATASET",  # <-- Your source path
)
```

---

## Step 3: Create Modality Config

### 3.1 Copy Template

```bash
cp finetuning/trossen_modality_config.py finetuning/YOUR_ROBOT_modality_config.py
```

### 3.2 Modify Configuration

```python
# Update the config dictionary:

your_robot_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "cam_1",      # <-- Your camera names (must match CAMERA_KEYS)
            "cam_2",
            "cam_3",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "component_1",  # <-- Your state component names
            "component_2",  # Must match STATE_INDICES keys
            "left_arm",
            "right_arm",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),  # Action horizon
        modality_keys=[
            "component_1",  # <-- Your action component names
            "left_arm",     # Must match ACTION_INDICES keys
            "right_arm",
        ],
        action_configs=[
            # One ActionConfig per modality_key, in same order
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,  # or RELATIVE
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

# Register with NEW_EMBODIMENT tag
register_modality_config(your_robot_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

# Export for evaluation
YOUR_ROBOT_CONFIG = your_robot_config
```

### 3.3 Action Representation Rules

| Action Type | Use ABSOLUTE | Use RELATIVE |
|-------------|--------------|--------------|
| Velocity commands | ✅ | ❌ |
| Position targets | ❌ | ✅ |
| Joint angles | ❌ | ✅ |
| Gripper | Either | Either |

**RELATIVE requires `relative_stats.json`** - the conversion script computes this automatically for components listed in `RELATIVE_ACTION_KEYS`.

---

## Step 4: Create Training Script

### 4.1 Copy Template

```bash
cp finetuning/finetune_ball2_groot.sh finetuning/finetune_YOUR_DATASET.sh
chmod +x finetuning/finetune_YOUR_DATASET.sh
```

### 4.2 Modify Paths

```bash
# Update these lines:
DATASET_PATH="${DATASET_PATH:-./data/YOUR_DATASET_train}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/YOUR_DATASET}"

# Update modality config path in torchrun command:
--modality-config-path ./finetuning/YOUR_ROBOT_modality_config.py
```

---

## Step 5: Create Evaluation Script

### 5.1 Copy Template

```bash
cp finetuning/evaluate_ball2_groot.py finetuning/evaluate_YOUR_DATASET.py
```

### 5.2 Update Labels

```python
# Update ACTION_LABELS with your joint names:
ACTION_LABELS = [
    "component_1_dim_0 (unit)",
    "component_1_dim_1 (unit)",
    "left_joint_0 (rad)",
    "left_joint_1 (rad)",
    # ... etc
]

# Update STATE_LABELS similarly
STATE_LABELS = [
    # ...
]
```

### 5.3 Update Import

```python
# Change this line:
from finetuning.YOUR_ROBOT_modality_config import YOUR_ROBOT_CONFIG  # noqa: F401
```

### 5.4 Update Default Paths

```python
# In argparse:
parser.add_argument("--dataset_path", default="./data/YOUR_DATASET_test")
parser.add_argument("--output_dir", default="./eval_results/YOUR_DATASET")

# In find_latest_checkpoint:
checkpoint_path = find_latest_checkpoint("./outputs/YOUR_DATASET")
```

---

## Step 6: Run Conversion

```bash
# Run conversion
python finetuning/convert_YOUR_DATASET.py \
    --input_path ./inhouse/.../YOUR_DATASET \
    --output_base ./data \
    --train_episodes N \
    --test_episodes M

# Check output
ls -la ./data/YOUR_DATASET_train/meta/
```

---

## Step 7: Validate Converted Data

### 7.1 Check All Required Files Exist

```bash
# Must have all these files:
ls ./data/YOUR_DATASET_train/meta/
# Expected: info.json, episodes.jsonl, tasks.jsonl, stats.json, relative_stats.json, modality.json
```

### 7.2 Validate stats.json Format

```bash
cat ./data/YOUR_DATASET_train/meta/stats.json | python -c "
import json, sys
stats = json.load(sys.stdin)
assert 'observation.state' in stats, 'Missing observation.state'
assert 'action' in stats, 'Missing action'
print('stats.json: OK')
print('  State keys:', list(stats['observation.state'].keys()))
print('  Action keys:', list(stats['action'].keys()))
"
```

### 7.3 Validate relative_stats.json Format

```bash
cat ./data/YOUR_DATASET_train/meta/relative_stats.json | python -c "
import json, sys
stats = json.load(sys.stdin)
for key in stats:
    shape = (len(stats[key]['mean']), len(stats[key]['mean'][0]))
    print(f'{key}: shape {shape}')
print('relative_stats.json: OK')
"
```

### 7.4 Check Video Files

```bash
# List video directories
ls ./data/YOUR_DATASET_train/videos/chunk-000/

# Check a video file
ffprobe ./data/YOUR_DATASET_train/videos/chunk-000/observation.images.YOUR_CAM/episode_000000.mp4 2>&1 | grep -E "Duration|Stream"
```

### 7.5 Check Parquet Data

```bash
python -c "
import pandas as pd
df = pd.read_parquet('./data/YOUR_DATASET_train/data/chunk-000/episode_000000.parquet')
print('Columns:', df.columns.tolist())
print('Rows:', len(df))
print('State shape:', len(df['observation.state'].iloc[0]))
print('Action shape:', len(df['action'].iloc[0]))
"
```

---

## Step 8: Run Finetuning

```bash
# Single GPU
python -m gr00t.experiment.launch_finetune \
    --modality-config-path ./finetuning/YOUR_ROBOT_modality_config.py \
    --embodiment-tag NEW_EMBODIMENT \
    --dataset-path ./data/YOUR_DATASET_train \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --output-dir ./outputs/YOUR_DATASET

# Multi-GPU (recommended)
./finetuning/finetune_YOUR_DATASET.sh
```

### Monitor Training

```bash
# Watch output directory for checkpoints
watch -n 60 "ls -la ./outputs/YOUR_DATASET/"

# If using wandb
# Check https://wandb.ai/YOUR_PROJECT
```

---

## Step 9: Evaluate Checkpoints

```bash
# Single checkpoint
python finetuning/evaluate_YOUR_DATASET.py \
    --checkpoint_path ./outputs/YOUR_DATASET/checkpoint-1000

# Multiple checkpoints
./finetuning/evaluate_checkpoints.sh 1000 2000 3000 5000 10000

# All checkpoints
CHECKPOINT_DIR=./outputs/YOUR_DATASET \
DATASET_PATH=./data/YOUR_DATASET_test \
OUTPUT_DIR=./eval_results/YOUR_DATASET \
./finetuning/evaluate_checkpoints.sh --all
```

---

## Quick Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `invalid choice` for embodiment | Wrong tag | Use `NEW_EMBODIMENT` |
| `FileNotFoundError: modality.json` | Missing file | Check conversion script creates it |
| `KeyError: 'observation.state'` | Wrong stats.json format | Top-level keys must be `observation.state` and `action` |
| `relative_action not found` | Missing relative_stats.json | Check `RELATIVE_ACTION_KEYS` in conversion |
| Video decode error | AV1 codec | Use `imageio.v3` for reading |
| Shape mismatch | DOF mismatch | Verify STATE_DIM, ACTION_DIM match actual data |
| Empty stats.json `{}` | Wrong column name | Source uses `observation.state` not `observation` |

---

## Template: Robot Info Card

Copy this for each new dataset:

```markdown
## Dataset: [NAME]

**Robot**: [Robot Name]
**Task**: [Task Description]
**Source**: `./inhouse/.../[PATH]`
**Date Converted**: [Date]

### Configuration

| Property | Value |
|----------|-------|
| Source FPS | XX Hz |
| Episodes | XX total (XX train / XX test) |
| Cameras | cam_1, cam_2, ... |
| State DOF | XX |
| Action DOF | XX |

### State Breakdown
| Component | DOF | Indices |
|-----------|-----|---------|
| ... | ... | ...:... |

### Action Breakdown
| Component | DOF | Indices | Representation |
|-----------|-----|---------|----------------|
| ... | ... | ...:... | ABSOLUTE/RELATIVE |

### Files Created
- `finetuning/convert_[NAME].py`
- `finetuning/[ROBOT]_modality_config.py`
- `finetuning/finetune_[NAME].sh`
- `finetuning/evaluate_[NAME].py`

### Results
| Checkpoint | MSE | MAE | Success@0.1 |
|------------|-----|-----|-------------|
| 1000 | ... | ... | ... |
| ... | ... | ... | ... |
```

---

## Files Reference

| File | Purpose | Key Variables to Modify |
|------|---------|------------------------|
| `convert_*.py` | Data conversion | `CAMERA_KEYS`, `STATE_DIM`, `ACTION_DIM`, `*_INDICES`, `RELATIVE_ACTION_KEYS`, `SOURCE_FPS` |
| `*_modality_config.py` | Robot config | `modality_keys`, `action_configs` |
| `finetune_*.sh` | Training | `DATASET_PATH`, `OUTPUT_DIR`, `--modality-config-path` |
| `evaluate_*.py` | Evaluation | `ACTION_LABELS`, `STATE_LABELS`, imports, default paths |

---

## Command Summary

```bash
# 1. Investigate source data
cat ./inhouse/.../DATASET/meta/info.json | python -m json.tool

# 2. Create scripts (copy from ball2_groot)
cp finetuning/convert_ball2_groot.py finetuning/convert_NEW.py
cp finetuning/trossen_modality_config.py finetuning/NEW_modality_config.py
cp finetuning/finetune_ball2_groot.sh finetuning/finetune_NEW.sh
cp finetuning/evaluate_ball2_groot.py finetuning/evaluate_NEW.py

# 3. Edit scripts with your robot config
# (see sections above for what to change)

# 4. Convert
python finetuning/convert_NEW.py

# 5. Validate
ls ./data/NEW_train/meta/
cat ./data/NEW_train/meta/stats.json | head -20

# 6. Train
./finetuning/finetune_NEW.sh

# 7. Evaluate
./finetuning/evaluate_checkpoints.sh 1000 2000 5000 10000
```

---

*Last Updated: January 2026*
*Reference Implementation: ball2_groot (Trossen AI Mobile)*
