# Input/Output Specifications

**Purpose:** Detailed specifications for model inputs and outputs with practical examples

---

## Overview

| Direction | Format | Description |
|-----------|--------|-------------|
| **Input** | Multi-modal dict | Images + State + Language |
| **Output** | Action dict | Robot motor commands |

---

## Model Inputs

### 1. Image Input

**Required Format:**
```python
{
    "video": {
        "camera_name": np.ndarray  # Shape: (B, T, H, W, 3)
    }
}
```

**Specifications:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Dtype | `uint8` | RGB values 0-255 |
| Channel order | RGB | Not BGR |
| Shape | (B, T, H, W, 3) | Batch, Time, Height, Width, Channels |
| Min resolution | 224×224 | Will be cropped/resized |
| Recommended | 480×640 or higher | Higher resolution = better |

**Example:**
```python
import numpy as np

# Single camera, single frame
images = np.random.randint(0, 255, (1, 1, 480, 640, 3), dtype=np.uint8)

observation = {
    "video": {
        "ego_view": images
    }
}
```

**Multiple Cameras:**
```python
observation = {
    "video": {
        "ego_view": ego_images,      # (B, T, H, W, 3)
        "wrist_cam": wrist_images,   # (B, T, H, W, 3)
        "third_person": tp_images    # (B, T, H, W, 3)
    }
}
```

---

### 2. State Input

**Required Format:**
```python
{
    "state": {
        "state_name": np.ndarray  # Shape: (B, T, D)
    }
}
```

**Specifications:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Dtype | `float32` | Raw sensor values |
| Shape | (B, T, D) | Batch, Time, Dimension |
| Units | Physical units | Radians for joints, meters for position |

**Common State Components:**

| State Name | Typical Dim | Description |
|------------|-------------|-------------|
| `joint_position` | 7 | Joint angles in radians |
| `joint_velocity` | 7 | Joint velocities (rad/s) |
| `gripper_position` | 1 | Gripper opening (0-1) |
| `ee_position` | 3 | End-effector XYZ position |
| `ee_orientation` | 4 or 6 | Quaternion or 6D rotation |

**Example:**
```python
# 7-DOF arm with gripper
joint_state = np.array([[
    [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5]  # Joint positions in radians
]], dtype=np.float32)

gripper_state = np.array([[[0.04]]], dtype=np.float32)  # Gripper opening

observation = {
    "state": {
        "joint_position": joint_state,   # (1, 1, 7)
        "gripper_position": gripper_state # (1, 1, 1)
    }
}
```

---

### 3. Language Input

**Required Format:**
```python
{
    "language": {
        "task": List[List[str]]  # Shape: (B, num_instructions)
    }
}
```

**Specifications:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Type | `List[List[str]]` | Nested list for batching |
| Content | Natural language | Task description |
| Length | < 256 tokens | After tokenization |

**Example:**
```python
observation = {
    "language": {
        "task": [
            ["pick up the red cube and place it on the plate"]
        ]
    }
}
```

**Batch Example:**
```python
observation = {
    "language": {
        "task": [
            ["pick up the red cube"],
            ["open the drawer"],
            ["pour water into the cup"]
        ]
    }
}
```

---

### 4. Complete Input Example

```python
import numpy as np

# Batch size 1, single timestep
B, T = 1, 1

observation = {
    # Images: (B, T, H, W, 3) uint8
    "video": {
        "ego_view": np.random.randint(0, 255, (B, T, 480, 640, 3), dtype=np.uint8),
    },

    # State: (B, T, D) float32
    "state": {
        "joint_position": np.zeros((B, T, 7), dtype=np.float32),
        "gripper_position": np.array([[[0.04]]], dtype=np.float32),
    },

    # Language: List[List[str]]
    "language": {
        "task": [["pick up the apple"]]
    }
}
```

---

## Model Outputs

### 1. Action Output

**Format:**
```python
{
    "action_name": np.ndarray  # Shape: (B, horizon, D)
}
```

**Specifications:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Dtype | `float32` | Physical units |
| Shape | (B, horizon, D) | Batch, Future steps, Dimension |
| Horizon | 8-40 steps | Configurable per embodiment |

**Common Action Components:**

| Action Name | Typical Dim | Description | Units |
|-------------|-------------|-------------|-------|
| `joint_action` | 7 | Target joint positions | radians |
| `gripper_action` | 1 | Gripper command | 0=close, 1=open |
| `ee_position` | 3 | End-effector target XYZ | meters |
| `ee_rotation` | 3 or 6 | Rotation (axis-angle or 6D) | various |

---

### 2. Action Output Example

```python
# Get action from policy
action_dict, info_dict = policy.get_action(observation)

# action_dict structure:
{
    "joint_action": np.ndarray,    # Shape: (1, 16, 7) - 16 future steps
    "gripper_action": np.ndarray,  # Shape: (1, 16, 1)
}

# info_dict structure:
{
    "embodiment": "GR1",
    "action_horizon": 16,
    "inference_time": 0.045,  # seconds
}
```

---

### 3. Using Action Output

```python
# Method 1: Execute first action only
first_action = action_dict["joint_action"][0, 0, :]  # Shape: (7,)
robot.set_joint_positions(first_action)

# Method 2: Execute action chunk with temporal smoothing
for t in range(action_horizon):
    action = action_dict["joint_action"][0, t, :]
    robot.set_joint_positions(action)
    time.sleep(1.0 / control_frequency)

# Method 3: Receding horizon (recommended)
while not task_complete:
    action_dict, _ = policy.get_action(observation)
    # Execute only first few actions before re-planning
    for t in range(replan_steps):
        robot.set_joint_positions(action_dict["joint_action"][0, t, :])
        time.sleep(dt)
    observation = get_new_observation()
```

---

## Embodiment-Specific I/O

### Unitree G1

```python
# State: 29 dimensions
state_spec = {
    "left_leg": 6,      # 6 joints
    "right_leg": 6,     # 6 joints
    "waist": 3,         # 3 joints
    "left_arm": 7,      # 7 joints
    "right_arm": 7,     # 7 joints
}

# Action: 29 dimensions, horizon 30
action_spec = {
    "left_leg": (30, 6),
    "right_leg": (30, 6),
    "waist": (30, 3),
    "left_arm": (30, 7),
    "right_arm": (30, 7),
}
```

### Franka Panda (LIBERO)

```python
# State: 8 dimensions
state_spec = {
    "joint_position": 7,   # 7 DOF arm
    "gripper": 1,          # Gripper opening
}

# Action: 8 dimensions, horizon 16
action_spec = {
    "joint_action": (16, 7),
    "gripper_action": (16, 1),
}
```

### WidowX (OXE)

```python
# State: 7 dimensions
state_spec = {
    "joint_position": 6,   # 6 DOF arm
    "gripper": 1,          # Gripper
}

# Action: 7 dimensions, horizon 8
action_spec = {
    "joint_action": (8, 6),
    "gripper_action": (8, 1),
}
```

---

## Validation Checklist

### Input Validation

```python
def validate_observation(obs):
    errors = []

    # Check video
    if "video" in obs:
        for cam_name, video in obs["video"].items():
            if video.dtype != np.uint8:
                errors.append(f"Video {cam_name} must be uint8, got {video.dtype}")
            if len(video.shape) != 5:
                errors.append(f"Video {cam_name} must be 5D (B,T,H,W,C), got {video.shape}")
            if video.shape[-1] != 3:
                errors.append(f"Video {cam_name} must have 3 channels, got {video.shape[-1]}")

    # Check state
    if "state" in obs:
        for state_name, state in obs["state"].items():
            if state.dtype != np.float32:
                errors.append(f"State {state_name} must be float32, got {state.dtype}")
            if len(state.shape) != 3:
                errors.append(f"State {state_name} must be 3D (B,T,D), got {state.shape}")

    # Check language
    if "language" in obs:
        task = obs["language"].get("task")
        if not isinstance(task, list) or not all(isinstance(t, list) for t in task):
            errors.append("Language task must be List[List[str]]")

    return errors
```

### Output Validation

```python
def validate_action(action_dict, expected_horizon, expected_dims):
    errors = []

    for action_name, action in action_dict.items():
        if action.dtype != np.float32:
            errors.append(f"Action {action_name} must be float32")

        if len(action.shape) != 3:
            errors.append(f"Action {action_name} must be 3D (B,H,D)")

        if action.shape[1] != expected_horizon:
            errors.append(f"Action horizon mismatch: got {action.shape[1]}, expected {expected_horizon}")

        # Check for NaN/Inf
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            errors.append(f"Action {action_name} contains NaN or Inf values")

    return errors
```

---

## Common Issues

### Issue: Shape Mismatch
```python
# Wrong: Missing batch dimension
images = np.zeros((480, 640, 3))

# Correct: Include B and T dimensions
images = np.zeros((1, 1, 480, 640, 3))
```

### Issue: Wrong Dtype
```python
# Wrong: float64 for images
images = np.zeros((1, 1, 480, 640, 3), dtype=np.float64)

# Correct: uint8 for images
images = np.zeros((1, 1, 480, 640, 3), dtype=np.uint8)
```

### Issue: BGR vs RGB
```python
# OpenCV loads as BGR
bgr_image = cv2.imread("image.png")

# Convert to RGB for model
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
```

### Issue: Unnormalized State
```python
# The model expects raw physical units
# Normalization is handled internally by the processor

# Wrong: Pre-normalized state
state = np.array([[[-1.0, 0.5, 0.2, ...]]])  # Already scaled

# Correct: Raw sensor values
state = np.array([[[0.0, -0.785, 0.0, -1.571, ...]]])  # Radians
```

---

*I/O Specifications for Isaac GR00T N1.6*
