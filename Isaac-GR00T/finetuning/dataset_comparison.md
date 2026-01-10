# Dataset Comparison: ALOHA vs GR00T Bimanual Data

## Overview

This document compares the ALOHA Transfer Cube dataset with GR00T's pretrained bimanual data format to identify conversion requirements.

---

## Side-by-Side Comparison

| Property | ALOHA Transfer Cube | GR00T GR1 Bimanual |
|----------|--------------------|--------------------|
| **Source** | `lerobot/aloha_sim_transfer_cube_human` | `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` |
| **Format** | LeRobot v2 | LeRobot v2 (GR00T variant) |
| **Robot** | ALOHA (2x ViperX arms) | Fourier GR-1 Humanoid |
| **Task** | Cube handover between arms | Pick-and-place tabletop |

---

## Sampling Rate Comparison

| Property | ALOHA | GR00T GR1 | Conversion Required |
|----------|-------|-----------|---------------------|
| **Control FPS** | 50 Hz | 20 Hz | **YES - Downsample 2.5x** |
| **Video FPS** | 50 Hz | 20 Hz | **YES - Downsample 2.5x** |
| **Episode Length** | 400 frames (~8 sec) | ~420 frames (~21 sec) | No |

### Downsampling Strategy
```python
# ALOHA: 50 Hz → GR00T: 20 Hz
# Keep every 2.5th frame (or interpolate)
# Option 1: Skip frames (indices 0, 2, 5, 7, 10, ...)
# Option 2: Resample with interpolation
target_indices = np.linspace(0, num_frames-1, int(num_frames * 20/50)).astype(int)
```

---

## State/Action Dimension Comparison

### ALOHA (14 DOF)
```
observation.state: [14]
├── Left Arm (7 DOF)
│   ├── waist
│   ├── shoulder
│   ├── elbow
│   ├── forearm_roll
│   ├── wrist_angle
│   ├── wrist_rotate
│   └── gripper
└── Right Arm (7 DOF)
    └── (same as left)
```

### GR00T GR1 (44 DOF total, 29 DOF active for arms+hands+waist)
```
observation.state: [44]
├── left_arm: [0:7]      # 7 DOF
├── left_hand: [7:13]    # 6 DOF
├── left_leg: [13:19]    # 6 DOF (unused for tabletop)
├── neck: [19:22]        # 3 DOF (unused for tabletop)
├── right_arm: [22:29]   # 7 DOF
├── right_hand: [29:35]  # 6 DOF
├── right_leg: [35:41]   # 6 DOF (unused for tabletop)
└── waist: [41:44]       # 3 DOF
```

### Active DOF for Bimanual Tasks
| Body Part | GR00T Indices | Dimensions | ALOHA Equivalent |
|-----------|---------------|------------|------------------|
| left_arm | [0:7] | 7 | left_arm [0:6] + gripper mapping |
| left_hand | [7:13] | 6 | left_gripper [6] (expand to 6) |
| right_arm | [22:29] | 7 | right_arm [7:13] + gripper mapping |
| right_hand | [29:35] | 6 | right_gripper [13] (expand to 6) |
| waist | [41:44] | 3 | N/A (set to zeros) |

### Dimension Mapping Strategy
```python
# ALOHA [14] → GR00T [44]
def map_aloha_to_groot(aloha_state):
    groot_state = np.zeros(44, dtype=np.float32)

    # Left arm: ALOHA [0:6] → GR00T [0:7]
    groot_state[0:6] = aloha_state[0:6]  # Direct mapping for 6 joints
    groot_state[6] = 0.0                  # 7th joint (pad or interpolate)

    # Left hand: ALOHA gripper [6] → GR00T [7:13]
    gripper_value = aloha_state[6]
    groot_state[7:13] = expand_gripper_to_hand(gripper_value)

    # Right arm: ALOHA [7:13] → GR00T [22:29]
    groot_state[22:28] = aloha_state[7:13]
    groot_state[28] = 0.0

    # Right hand: ALOHA gripper [13] → GR00T [29:35]
    gripper_value = aloha_state[13]
    groot_state[29:35] = expand_gripper_to_hand(gripper_value)

    # Waist, legs, neck: zeros (not used in ALOHA)
    # groot_state[13:19] = 0  # left_leg
    # groot_state[19:22] = 0  # neck
    # groot_state[35:41] = 0  # right_leg
    # groot_state[41:44] = 0  # waist

    return groot_state
```

---

## Video/Image Comparison

| Property | ALOHA | GR00T GR1 | Conversion Required |
|----------|-------|-----------|---------------------|
| **Camera Key** | `observation.images.top` | `observation.images.ego_view` | **YES - Rename** |
| **Resolution** | 480 × 640 × 3 | 256 × 256 × 3 | **YES - Resize & Crop** |
| **Codec** | AV1 | H.264 | **YES - Re-encode** |
| **Color Format** | RGB | RGB | No |

### Image Transformation Pipeline
```python
# ALOHA: 480x640 → GR00T: 256x256
def transform_image(img):
    # 1. Center crop to square (480x480)
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    img_cropped = img[start_h:start_h+min_dim, start_w:start_w+min_dim]

    # 2. Resize to 256x256
    img_resized = cv2.resize(img_cropped, (256, 256), interpolation=cv2.INTER_AREA)

    return img_resized
```

---

## Language/Task Description

| Property | ALOHA | GR00T GR1 |
|----------|-------|-----------|
| **Key** | N/A (implicit task) | `annotation.human.action.task_description` |
| **Format** | None | Task index → Text lookup |

### Task Description Mapping
```python
# Add task description for ALOHA transfer cube
task_description = "pick up the cube with one hand and transfer it to the other hand"
```

---

## File Structure Comparison

### ALOHA (LeRobot)
```
aloha_sim_transfer_cube_human/
├── data/
│   └── train-00000-of-00001.parquet
├── videos/
│   └── observation.images.top/
│       └── episode_000000.mp4
└── meta/
    └── info.json
```

### GR00T (LeRobot v2)
```
gr1.PickNPlace/
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
├── videos/
│   └── chunk-000/
│       └── ego_view_bg_crop_pad_res256_freq20/
│           └── episode_000000.mp4
└── meta/
    ├── info.json
    ├── modality.json
    ├── episodes.jsonl
    ├── tasks.jsonl
    └── stats.json
```

---

## Summary of Required Conversions

| Aspect | Conversion | Difficulty |
|--------|------------|------------|
| **Sampling Rate** | 50 Hz → 20 Hz | Medium |
| **State Dimensions** | 14 → 44 (with zeros) | Easy |
| **Action Dimensions** | 14 → 44 (with zeros) | Easy |
| **Gripper → Hand** | 1 DOF → 6 DOF expansion | Medium |
| **Image Resolution** | 480×640 → 256×256 | Easy |
| **Image Key** | `top` → `ego_view_bg_crop_pad_res256_freq20` | Easy |
| **Video Codec** | AV1 → H.264 | Easy |
| **Add Modality Config** | Create modality.json | Easy |
| **Add Task Description** | Add language annotation | Easy |
| **Statistics** | Compute mean/std | Easy |

---

## Embodiment Configuration for ALOHA

Since ALOHA is a new embodiment not in GR00T's pretraining, we need to register it:

```python
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig

# Option 1: Map to existing GR1 config (recommended for transfer learning)
# Use GR1 but only populate arm+hand dimensions

# Option 2: Register new embodiment
ALOHA_MODALITY_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view_bg_crop_pad_res256_freq20"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"],
        sin_cos_embedding_keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}
```

---

## Recommended Approach

1. **Use GR1 Embodiment Tag**: Map ALOHA data to GR1 format
   - Pros: Can leverage pretrained GR1 weights
   - Cons: Unused dimensions (legs, neck) will be zeros

2. **Downsample to 20 Hz**: Match GR00T's control frequency
   - Use frame interpolation for smoother trajectories

3. **Expand Gripper to Hand**: Convert 1-DOF gripper to 6-DOF hand
   - Option A: Replicate gripper value across all hand joints
   - Option B: Use learned mapping (requires additional data)

4. **Maintain LeRobot v2 Format**: Keep compatibility with GR00T data loading
