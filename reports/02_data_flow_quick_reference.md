# Data Flow Quick Reference

**Purpose:** Quick reference for understanding data transformations in Isaac GR00T N1.6

---

## Data Flow Diagram

```
RAW FILES                    INTERMEDIATE               MODEL INPUT               MODEL OUTPUT
─────────────────────────────────────────────────────────────────────────────────────────────

videos/                      VLAStepData                BatchFeature              Action Dict
├── episode_*.mp4   ────►    ├── images     ────►      ├── pixel_values  ────►   ├── joint_action
                             │   dict[str,              │   (B,N,3,224,224)       │   (B,H,7)
data/                        │    list[ndarray]]        │                         │
├── episode_*.parquet ───►   ├── states     ────►      ├── state         ────►   ├── gripper_action
    - observation.state      │   dict[str,ndarray]      │   (B,state_dim)         │   (B,H,1)
    - action                 │                          │                         │
                             ├── actions    ────►      ├── action        ────►   └── ...
meta/                        │   dict[str,ndarray]      │   (B,H,action_dim)
├── modality.json            │                          │
├── stats.json               ├── text       ────►      ├── input_ids
└── tasks.jsonl              │   str                    │   (B,seq_len)
                             │                          │
                             └── embodiment            └── attention_mask
                                 EmbodimentTag              (B,seq_len)
```

---

## Shape Transformations

### Images
```
Raw:        (H, W, 3) uint8        # From MP4 video
Cropped:    (244, 244, 3) uint8    # Center crop
Resized:    (224, 224, 3) uint8    # Model input size
Processed:  (3, 224, 224) float32  # CHW format, normalized
Batched:    (B, N, 3, 224, 224)    # N = number of views
```

### State
```
Raw:        (total_dim,) float32   # Concatenated from parquet
Split:      {                      # By modality.json
              "joint_pos": (7,),
              "gripper": (1,),
            }
Normalized: (state_dim,) float32  # Scaled to [-1, 1]
Batched:    (B, state_dim)
```

### Actions
```
Raw:        (total_dim,) float32   # Per timestep
Chunked:    (horizon, action_dim)  # Future action sequence
Normalized: (horizon, action_dim)  # Scaled to [-1, 1]
Batched:    (B, horizon, action_dim)
Output:     (B, horizon, action_dim) # Denormalized to physical units
```

---

## Key Processing Steps

### 1. Episode Loading
```python
# LeRobotEpisodeLoader
loader = LeRobotEpisodeLoader(dataset_path)
episode_df = loader.load_episode(episode_idx)
# Returns: DataFrame with all timesteps for episode
```

### 2. Step Extraction
```python
# ShardedSingleStepDataset.extract_step_data()
step_data = extract_step_data(
    df=episode_df,
    step_idx=100,
    delta_indices=[-2, -1, 0]  # 3 frames of history
)
# Returns: VLAStepData
```

### 3. State Normalization
```python
# StateActionProcessor.normalize_state()
normalized = (raw - stats['min']) / (stats['max'] - stats['min'])
normalized = 2 * normalized - 1  # Scale to [-1, 1]
```

### 4. Image Processing
```python
# Gr00tN1d6Processor
image = crop_center(image, 244, 244)
image = resize(image, 224, 224)
image = augment(image)  # Color jitter, etc.
image = normalize(image, mean, std)
```

### 5. Action Denormalization (Inference)
```python
# Inverse of normalization
raw = (normalized + 1) / 2  # Scale to [0, 1]
raw = raw * (stats['max'] - stats['min']) + stats['min']
```

---

## Normalization Statistics

Located in `meta/stats.json`:

```json
{
  "observation.state": {
    "min": [-3.14, -1.57, ...],
    "max": [3.14, 1.57, ...],
    "mean": [0.0, 0.5, ...],
    "std": [1.0, 0.8, ...]
  },
  "action": {
    "min": [...],
    "max": [...],
    "mean": [...],
    "std": [...]
  }
}
```

---

## Action Representations

| Type | Formula | Use Case |
|------|---------|----------|
| ABSOLUTE | `a` | Direct motor commands |
| RELATIVE | `a - s_current` | Delta from current state |
| DELTA | `a_t - a_{t-1}` | Change from previous action |

---

## Temporal Sampling (delta_indices)

```python
# delta_indices controls history window
delta_indices = [0]           # Current frame only
delta_indices = [-1, 0]       # Previous + current
delta_indices = [-2, -1, 0]   # 3 frames of history

# Action horizon (future predictions)
delta_indices = list(range(30))  # Predict 30 future steps
```

---

## Common Tensor Dimensions

| Variable | Typical Value | Description |
|----------|---------------|-------------|
| B | 8-64 | Batch size |
| H | 8-40 | Action horizon |
| state_dim | 8-29 | Robot state dimension |
| action_dim | 7-29 | Action dimension |
| seq_len | 50-200 | Token sequence length |
| N | 1-3 | Number of camera views |

---

## Quick Code Snippets

### Load and inspect dataset
```python
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

loader = LeRobotEpisodeLoader("/path/to/dataset")
print(f"Episodes: {loader.num_episodes}")
print(f"Modalities: {loader.modality_config}")
```

### Create training dataset
```python
from gr00t.data.dataset.factory import DatasetFactory

factory = DatasetFactory(
    data_config=data_config,
    embodiment_tag="NEW_EMBODIMENT"
)
train_dataset = factory.create_train_dataset()
```

### Run inference
```python
from gr00t.policy.gr00t_policy import Gr00tPolicy

policy = Gr00tPolicy(
    model_path="nvidia/GR00T-N1.6-3B",
    embodiment_tag=EmbodimentTag.GR1
)

obs = {
    "video": {"ego_view": images},
    "state": {"joint_state": states},
    "language": {"task": [["pick up cube"]]}
}

action, info = policy.get_action(obs)
```

---

## File Locations

| Data Type | Location |
|-----------|----------|
| Dataset metadata | `meta/info.json` |
| Modality structure | `meta/modality.json` |
| Normalization stats | `meta/stats.json` |
| Task descriptions | `meta/tasks.jsonl` |
| Episode info | `meta/episodes.jsonl` |
| Videos | `videos/chunk-*/observation.images.*/*.mp4` |
| State/Action data | `data/chunk-*/*.parquet` |

---

*Quick reference for Isaac GR00T N1.6 data flow*
