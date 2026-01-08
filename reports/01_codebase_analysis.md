# Isaac GR00T N1.6 Codebase Analysis Report

**Date:** January 8, 2026
**Author:** Claude Code Analysis
**Version:** 1.0

---

## Executive Summary

Isaac GR00T N1.6 is NVIDIA's open vision-language-action (VLA) foundation model for generalist humanoid robots. This report documents the complete data processing pipeline, model architecture, and input/output specifications.

**Key Characteristics:**
- 3B-parameter multimodal model
- Inputs: Images + Proprioceptive State + Language Instructions
- Outputs: Action sequences (robot motor commands)
- Architecture: Eagle VLM backbone + 32-layer Diffusion Transformer
- Data Format: LeRobot v2 with GR00T extensions

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Data Pipeline Overview](#2-data-pipeline-overview)
3. [Input Specifications](#3-input-specifications)
4. [Output Specifications](#4-output-specifications)
5. [Model Architecture](#5-model-architecture)
6. [Data Processing Stages](#6-data-processing-stages)
7. [Configuration System](#7-configuration-system)
8. [Training Pipeline](#8-training-pipeline)
9. [Inference Pipeline](#9-inference-pipeline)
10. [Key Files Reference](#10-key-files-reference)

---

## 1. Project Structure

```
Isaac-GR00T/
├── gr00t/                          # Main package
│   ├── configs/                    # Configuration management
│   │   ├── base_config.py         # Base config definitions
│   │   ├── model/                  # Model configs (gr00t_n1d6.py)
│   │   ├── training/               # Training configs
│   │   └── data/                   # Data configs with embodiment definitions
│   ├── data/                       # Data loading and preprocessing
│   │   ├── dataset/                # Dataset implementations
│   │   ├── state_action/           # State and action processing
│   │   ├── collator/               # Batch assembly
│   │   └── types.py               # Core data types
│   ├── model/                      # Model implementations
│   │   ├── gr00t_n1d6/            # GR00T N1.6 specific
│   │   └── modules/               # Model components (Eagle, DiT)
│   ├── policy/                     # Inference interface
│   ├── experiment/                # Training and evaluation
│   └── eval/                       # Evaluation scripts
├── examples/                       # Robot-specific examples
├── scripts/                        # Deployment and conversion tools
├── getting_started/               # Documentation
└── docker/                        # Container setup
```

---

## 2. Data Pipeline Overview

The data flows through 8 distinct stages from raw files to model predictions:

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATA INPUT                           │
│  (Videos, Proprioception, Actions, Language Annotations)    │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
        ┌──────────────────────────────────┐
        │  Stage 1: LeRobotEpisodeLoader   │
        │  - Parse metadata files          │
        │  - Decode MP4 videos             │
        │  - Load parquet state/action     │
        └──────────────┬───────────────────┘
                       ↓
        ┌──────────────────────────────────┐
        │  Stage 2: ShardedSingleStepDataset│
        │  - Extract step-level data       │
        │  - Apply temporal sampling       │
        │  - Create VLAStepData objects    │
        └──────────────┬───────────────────┘
                       ↓
        ┌──────────────────────────────────┐
        │  Stage 3: ShardedMixtureDataset  │
        │  - Mix multiple datasets         │
        │  - Apply sampling weights        │
        └──────────────┬───────────────────┘
                       ↓
        ┌──────────────────────────────────┐
        │  Stage 4: Gr00tN1d6Processor     │
        │  - Normalize state/action        │
        │  - Augment images                │
        │  - Process vision (Eagle VLM)    │
        │  - Tokenize language             │
        └──────────────┬───────────────────┘
                       ↓
        ┌──────────────────────────────────┐
        │  Stage 5: Gr00tN1d6DataCollator  │
        │  - Batch samples                 │
        │  - Stack tensors                 │
        │  - Pad sequences                 │
        └──────────────┬───────────────────┘
                       ↓
        ┌──────────────────────────────────┐
        │  Stage 6: Gr00tN1d6ActionHead    │
        │  - State Encoder (MLP)           │
        │  - DiT (32 layers)               │
        │  - Action Decoder (MLP)          │
        └──────────────┬───────────────────┘
                       ↓
        ┌──────────────────────────────────┐
        │   ACTION PREDICTION OUTPUT       │
        │   (unnormalized physical units)  │
        └──────────────────────────────────┘
```

---

## 3. Input Specifications

### 3.1 Raw Dataset Format (LeRobot v2 + GR00T Extensions)

```
dataset_root/
├── meta/
│   ├── info.json              # Dataset metadata (fps, features, splits)
│   ├── episodes.jsonl         # Episode-level information
│   ├── tasks.jsonl            # Language task annotations
│   ├── modality.json          # KEY: Defines state/action/video structure
│   ├── stats.json             # Normalization statistics (min/max/mean/std)
│   └── relative_stats.json    # Relative action statistics
├── videos/chunk-000/
│   └── observation.images.{camera_name}/
│       ├── episode_000000.mp4
│       └── episode_000001.mp4
└── data/chunk-000/
    ├── episode_000000.parquet
    └── episode_000001.parquet
```

### 3.2 Parquet File Structure

Each parquet file contains per-timestep data:

| Column | Type | Description |
|--------|------|-------------|
| `observation.state` | float32[] | Concatenated proprioceptive state |
| `action` | float32[] | Concatenated action array |
| `timestamp` | float64 | Timestep time |
| `annotation.human.action.task_description` | int | Task index |
| `task_index` | int | Task index reference |
| `episode_index` | int | Episode identifier |
| `index` | int | Global step index |

### 3.3 modality.json Structure

Defines how to parse concatenated arrays:

```json
{
  "observation.state": {
    "joint_position": {"start": 0, "end": 7},
    "gripper_position": {"start": 7, "end": 8}
  },
  "action": {
    "joint_action": {"start": 0, "end": 7},
    "gripper_action": {"start": 7, "end": 8}
  },
  "observation.images": {
    "ego_view": {"original_key": "observation.images.ego_view"}
  }
}
```

### 3.4 VLAStepData (Intermediate Data Structure)

After loading, each timestep is represented as:

```python
@dataclass
class VLAStepData:
    images: dict[str, list[np.ndarray]]     # {camera_name: [frame1, frame2, ...]}
                                            # Shape per frame: (H, W, 3), dtype: uint8

    states: dict[str, np.ndarray]           # {state_name: array}
                                            # Shape: (dim,), dtype: float32

    actions: dict[str, np.ndarray]          # {action_name: array}
                                            # Shape: (horizon, dim), dtype: float32

    text: str | None                        # Task description string

    embodiment: EmbodimentTag               # Robot type identifier

    metadata: dict[str, Any]                # Episode/step metadata
```

### 3.5 Model Input Format (After Processing)

The processed input to the model:

```python
{
    # Vision-Language content for Eagle backbone
    'vlm_content': {
        'text': str,                        # Task instruction
        'images': List[PIL.Image],          # Processed images (224x224)
        'conversation': [{'role': 'user', 'content': [...]}]
    },

    # Proprioceptive state
    'state': np.ndarray,                    # Shape: (state_dim,), normalized to [-1, 1]
    'state_mask': np.ndarray,               # Shape: (state_dim,), valid state indicators

    # Action targets (training only)
    'action': np.ndarray,                   # Shape: (horizon, action_dim), normalized
    'action_mask': np.ndarray               # Shape: (horizon, action_dim), valid action indicators
}
```

### 3.6 Batched Input (After Collation)

```python
BatchFeature({
    'input_ids': torch.Tensor,              # Shape: (B, seq_len), tokenized text
    'attention_mask': torch.Tensor,         # Shape: (B, seq_len), text attention mask
    'pixel_values': torch.Tensor,           # Shape: (B, N_imgs, 3, 224, 224), images
    'state': torch.Tensor,                  # Shape: (B, state_dim), normalized states
    'action': torch.Tensor,                 # Shape: (B, horizon, action_dim), actions
    'state_mask': torch.Tensor,             # Shape: (B, state_dim)
    'action_mask': torch.Tensor             # Shape: (B, horizon, action_dim)
})
```

---

## 4. Output Specifications

### 4.1 Model Output (Raw)

```python
{
    'action': torch.Tensor,                 # Shape: (B, horizon, action_dim)
                                            # Normalized to [-1, 1] range

    'loss': torch.Tensor,                   # Scalar, diffusion loss (training only)

    'hidden_states': torch.Tensor           # Optional: intermediate representations
}
```

### 4.2 Policy Output (After Denormalization)

```python
# From policy.get_action():
action_dict = {
    'joint_action': np.ndarray,             # Shape: (B, horizon, joint_dim)
                                            # Physical units (radians for joints)

    'gripper_action': np.ndarray,           # Shape: (B, horizon, 1)
                                            # Range: [0, 1] for gripper open/close
}

info_dict = {
    'embodiment': str,                      # Robot embodiment tag
    'action_horizon': int,                  # Number of future steps predicted
    'inference_time': float                 # Time taken for inference
}
```

### 4.3 Action Horizon

The model predicts a sequence of future actions:

| Embodiment | Typical Horizon | Action Dim |
|------------|-----------------|------------|
| Unitree G1 | 30 steps | 29 |
| Franka Panda | 16 steps | 8 |
| Google Robot | 8 steps | 7 |

---

## 5. Model Architecture

### 5.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     GR00T N1.6 Model                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Eagle VLM Backbone (2B)                 │   │
│   │  - Vision Encoder (SigLIP)                          │   │
│   │  - Language Model (Qwen2)                           │   │
│   │  - Vision-Language Projector                        │   │
│   │  Output: 2048-dim VL embeddings                     │   │
│   └──────────────────────┬──────────────────────────────┘   │
│                          ↓                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │           State Encoder (Embodiment MLP)             │   │
│   │  - CategorySpecificMLP (per robot type)             │   │
│   │  - Input: raw state (e.g., 29-dim for G1)           │   │
│   │  - Output: 1536-dim embedding                       │   │
│   └──────────────────────┬──────────────────────────────┘   │
│                          ↓                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │         Diffusion Transformer (DiT, 32 layers)       │   │
│   │  - AlternateVLDiT architecture                      │   │
│   │  - Cross-attention with VL embeddings               │   │
│   │  - Flow-matching diffusion                          │   │
│   │  - 4 denoising steps (inference)                    │   │
│   └──────────────────────┬──────────────────────────────┘   │
│                          ↓                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │          Action Decoder (Embodiment MLP)             │   │
│   │  - Projects latents back to action space            │   │
│   │  - Output: (B, horizon, action_dim)                 │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 Key Components

| Component | Parameters | Purpose |
|-----------|------------|---------|
| Eagle VLM | ~2B | Vision-language understanding |
| State Encoder | ~10M | Embodiment-specific state embedding |
| DiT | ~800M | Action generation via diffusion |
| Action Decoder | ~10M | Latent to action projection |

### 5.3 Diffusion Process

- **Training:** Add noise to ground truth actions, predict denoising
- **Inference:** Start from noise, denoise in 4 steps
- **Method:** Flow-matching (continuous-time diffusion)

---

## 6. Data Processing Stages

### Stage 1: LeRobotEpisodeLoader

**File:** `gr00t/data/dataset/lerobot_episode_loader.py`

**Process:**
1. Load `meta/info.json` for dataset metadata
2. Parse `meta/modality.json` for array structure
3. Load episode list from `meta/episodes.jsonl`
4. Load task descriptions from `meta/tasks.jsonl`
5. Load normalization stats from `meta/stats.json`

### Stage 2: ShardedSingleStepDataset

**File:** `gr00t/data/dataset/sharded_single_step_dataset.py`

**Key Function:** `extract_step_data()`

```python
def extract_step_data(df, step_idx, modality_config):
    """
    Extracts data for a single timestep with temporal context.

    Uses delta_indices to gather history:
    - delta_indices=[0] → current frame only
    - delta_indices=[-2, -1, 0] → 3 frames (2 past + current)
    """
```

### Stage 3: StateActionProcessor

**File:** `gr00t/data/state_action/state_action_processor.py`

**Normalization Methods:**
- **Min-Max:** `x_norm = 2 * (x - min) / (max - min) - 1`
- **Mean-Std:** `x_norm = (x - mean) / std`
- **Sin-Cos Encoding:** For rotational states (joints)

**Action Representations:**
- `ABSOLUTE`: Direct action values
- `RELATIVE`: Delta from current state
- `DELTA`: Change from previous action

### Stage 4: Gr00tN1d6Processor

**File:** `gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py`

**Image Processing:**
1. Crop to 244×244 (center crop)
2. Resize to 224×224
3. Apply augmentations:
   - Color jitter (brightness, contrast, saturation, hue)
   - Random crop
   - Albumentations transforms

### Stage 5: Gr00tN1d6DataCollator

**File:** `gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py`

**Process:**
1. Stack individual samples into batches
2. Pad text sequences (padding_side="left" for Flash Attention)
3. Process images through Eagle's vision processor
4. Create attention masks

---

## 7. Configuration System

### 7.1 ModalityConfig

Defines how to sample and load each modality:

```python
@dataclass
class ModalityConfig:
    delta_indices: list[int]           # Temporal sampling [-2, -1, 0]
    modality_keys: list[str]           # Which fields to load
    sin_cos_embedding_keys: list[str]  # Apply sin/cos encoding
    mean_std_embedding_keys: list[str] # Normalization type
    action_configs: list[ActionConfig] # Action-specific configs
```

### 7.2 ActionConfig

Specifies action handling:

```python
@dataclass
class ActionConfig:
    rep: ActionRepresentation    # RELATIVE | DELTA | ABSOLUTE
    type: ActionType             # EEF (end-effector) | NON_EEF
    format: ActionFormat         # DEFAULT | XYZ_ROT6D | XYZ_ROTVEC
    state_key: str | None        # Reference state for relative actions
```

### 7.3 EmbodimentTag

Predefined robot configurations:

| Tag | Robot | Use Case |
|-----|-------|----------|
| `GR1` | NVIDIA GR1 | Pretraining |
| `UNITREE_G1` | Unitree G1 | Posttraining |
| `LIBERO_PANDA` | Franka Panda | Simulation |
| `OXE_WIDOWX` | WidowX | Manipulation |
| `NEW_EMBODIMENT` | Custom | Finetuning |

---

## 8. Training Pipeline

### 8.1 Entry Point

**File:** `gr00t/experiment/launch_finetune.py`

```bash
python gr00t/experiment/launch_finetune.py \
    --dataset-path /path/to/lerobot/dataset \
    --embodiment-tag NEW_EMBODIMENT \
    --output-dir ./checkpoints \
    --num-train-steps 10000
```

### 8.2 Training Configuration

**Key Parameters:**
- `tune_projector=True` - Train embodiment-specific MLPs
- `tune_diffusion_model=True` - Train DiT layers
- `tune_llm=False` - Freeze VLM (default)
- `use_relative_action=True` - Relative action representation

### 8.3 Loss Function

Diffusion loss (flow-matching):
```
L = E[||v_θ(x_t, t) - (x_1 - x_0)||²]
```
Where:
- `x_0`: noise
- `x_1`: ground truth action
- `x_t`: interpolation at time t
- `v_θ`: predicted velocity

---

## 9. Inference Pipeline

### 9.1 Policy API

**File:** `gr00t/policy/gr00t_policy.py`

```python
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

# Initialize
policy = Gr00tPolicy(
    model_path="nvidia/GR00T-N1.6-3B",
    embodiment_tag=EmbodimentTag.GR1,
    device="cuda:0"
)

# Format observation
observation = {
    "video": {
        "ego_view": np.ndarray  # Shape: (B, T, H, W, 3), uint8
    },
    "state": {
        "joint_state": np.ndarray  # Shape: (B, T, D), float32
    },
    "language": {
        "task": [["pick up the apple"]]  # List of task strings
    }
}

# Get action
action, info = policy.get_action(observation)
# action: {"joint_action": np.ndarray(B, horizon, 7)}
```

### 9.2 Server-Client Architecture

**Server:**
```python
python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --port 5555
```

**Client:**
```python
from gr00t.policy.server_client import PolicyClient

policy = PolicyClient(host="localhost", port=5555)
action, info = policy.get_action(observation)
```

### 9.3 Inference Performance

| Hardware | Speed | Denoising Steps |
|----------|-------|-----------------|
| RTX 5090 | 27.3 Hz | 4 |
| RTX 4090 | ~20 Hz | 4 |
| A100 | ~18 Hz | 4 |

---

## 10. Key Files Reference

| File | Purpose |
|------|---------|
| `gr00t/data/types.py` | Core data structures (VLAStepData, ModalityConfig) |
| `gr00t/data/dataset/lerobot_episode_loader.py` | Load LeRobot format datasets |
| `gr00t/data/dataset/sharded_single_step_dataset.py` | Step-level dataset extraction |
| `gr00t/data/state_action/state_action_processor.py` | State/action normalization |
| `gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py` | Full input processing pipeline |
| `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` | Model architecture |
| `gr00t/model/modules/dit.py` | Diffusion Transformer implementation |
| `gr00t/model/modules/eagle_backbone.py` | Vision-Language backbone |
| `gr00t/policy/gr00t_policy.py` | Inference policy interface |
| `gr00t/experiment/launch_finetune.py` | Training entry point |
| `gr00t/configs/data/embodiment_configs.py` | Embodiment definitions |

---

## Appendix A: Example modality.json

```json
{
  "observation.state": {
    "left_arm": {"start": 0, "end": 7, "dtype": "float32"},
    "right_arm": {"start": 7, "end": 14, "dtype": "float32"},
    "left_gripper": {"start": 14, "end": 15, "dtype": "float32"},
    "right_gripper": {"start": 15, "end": 16, "dtype": "float32"}
  },
  "action": {
    "left_arm": {"start": 0, "end": 7, "dtype": "float32"},
    "right_arm": {"start": 7, "end": 14, "dtype": "float32"},
    "left_gripper": {"start": 14, "end": 15, "dtype": "float32"},
    "right_gripper": {"start": 15, "end": 16, "dtype": "float32"}
  },
  "observation.images": {
    "ego_view": {"original_key": "observation.images.ego_view"},
    "wrist_view": {"original_key": "observation.images.wrist_view"}
  }
}
```

---

## Appendix B: Tensor Shape Summary

| Stage | Data | Shape |
|-------|------|-------|
| Raw Image | video frame | (H, W, 3), uint8 |
| Processed Image | pixel_values | (B, N, 3, 224, 224), float32 |
| Raw State | joint_position | (D,), float32 |
| Processed State | state | (B, state_dim), float32, normalized |
| Raw Action | action | (horizon, D), float32 |
| Processed Action | action | (B, horizon, action_dim), float32, normalized |
| VLM Output | embeddings | (B, seq_len, 2048), float32 |
| DiT Output | latents | (B, horizon, latent_dim), float32 |
| Final Output | action | (B, horizon, action_dim), float32, physical units |

---

*Report generated by Claude Code analysis of Isaac-GR00T codebase*
