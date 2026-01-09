# GR00T Dataset Analysis Report

**Generated**: 2026-01-08 13:56:21
**Dataset**: `demo_data/gr1.PickNPlace`

---

## 1. Overview

This report analyzes a LeRobot v2 format dataset used for GR00T VLA model training.

| Property | Value |
|----------|-------|
| Format Version | v2.0 |
| Robot Type | GR1ArmsOnly |
| Total Episodes | 5 |
| Total Frames | 2096 |
| FPS | 20.0 |

---

## 2. Dataset Structure

```
gr1.PickNPlace/
├── meta/
│   ├── info.json          # Dataset configuration
│   ├── episodes.jsonl     # Per-episode metadata
│   ├── tasks.jsonl        # Task descriptions
│   ├── modality.json      # Data layout mapping
│   └── stats.json         # Normalization statistics
├── data/
│   └── chunk-XXX/
│       └── episode_XXXXXX.parquet  # State/action data
└── videos/
    └── chunk-XXX/
        └── <camera_name>/
            └── episode_XXXXXX.mp4  # Video frames
```

---

## 3. Modality Configuration

The `modality.json` defines how raw data maps to model inputs:

### State (Robot Proprioception)
| Body Part | Index Range | Dimensions |
|-----------|-------------|------------|
| left_arm | [0:7] | 7 |
| left_hand | [7:13] | 6 |
| left_leg | [13:19] | 6 |
| neck | [19:22] | 3 |
| right_arm | [22:29] | 7 |
| right_hand | [29:35] | 6 |
| right_leg | [35:41] | 6 |
| waist | [41:44] | 3 |

### Action (Robot Commands)
| Body Part | Index Range | Dimensions |
|-----------|-------------|------------|
| left_arm | [0:7] | 7 |
| left_hand | [7:13] | 6 |
| left_leg | [13:19] | 6 |
| neck | [19:22] | 3 |
| right_arm | [22:29] | 7 |
| right_hand | [29:35] | 6 |
| right_leg | [35:41] | 6 |
| waist | [41:44] | 3 |

### Video (Camera Inputs)
| Key | Original Data Path |
|-----|-------------------|
| ego_view_bg_crop_pad_res256_freq20 | observation.images.ego_view |

---

## 4. Task Descriptions

Language instructions used for training:

- [0] pick the pear from the counter and place it in the plate
- [2] pick the sweet potato from the counter and place it in the plate
- [3] pick the meat from the counter and place it in the plate
- [4] pick the salt from the counter and place it in the plate
- [5] pick the cupcake from the counter and place it in the plate

---

## 5. Data Flow: How the Model Uses This Data

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING DATA FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. LOAD EPISODE                                                │
│     parquet file → DataFrame with state, action, timestamps    │
│     video file → Frame sequence (H×W×3 RGB images)             │
│                                                                 │
│  2. APPLY MODALITY MAPPING                                      │
│     state[0:7] → left_arm joints                               │
│     state[7:13] → left_hand joints                             │
│     ... (as defined in modality.json)                          │
│                                                                 │
│  3. NORMALIZE                                                   │
│     normalized = (raw - mean) / std                            │
│     (using stats.json values)                                  │
│                                                                 │
│  4. CREATE TRAINING SAMPLE                                      │
│     ┌─────────────────┐                                        │
│     │ Video frames    │ → Vision Encoder → Visual tokens       │
│     │ State vector    │ → State Encoder → State tokens         │
│     │ Task text       │ → Text Encoder → Language tokens       │
│     └────────┬────────┘                                        │
│              │                                                  │
│              ▼                                                  │
│     ┌─────────────────┐                                        │
│     │ Fused Features  │ → Diffusion Transformer                │
│     └────────┬────────┘                                        │
│              │                                                  │
│              ▼                                                  │
│     ┌─────────────────┐                                        │
│     │ Action Chunk    │ → Loss vs Ground Truth Actions         │
│     │ (16 future      │                                        │
│     │  timesteps)     │                                        │
│     └─────────────────┘                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Generated Files

| File | Description |
|------|-------------|
| `modality_config.json` | Modality configuration |
| `statistics.json` | Normalization statistics |
| `episode_lengths.png` | Episode length distribution |
| `episode_X_trajectory.png` | State/action plots |
| `episode_X_frames.png` | Sample video frames |
| `episode_X_sample.csv` | Sample data rows |

---

## 7. Key Takeaways

1. **Format**: LeRobot v2 with GR00T extensions
2. **Video**: Stored as H.264 MP4, loaded frame-by-frame
3. **State/Action**: Stored in Parquet, indexed by body part
4. **Language**: Task descriptions map to episode annotations
5. **Normalization**: Mean/std statistics pre-computed for training

---

*Report generated by GR00T Dataset Analyzer*
