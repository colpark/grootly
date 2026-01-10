#!/usr/bin/env python3
"""
Convert ALOHA Transfer Cube Dataset to GR00T Format

This script converts the LeRobot ALOHA bimanual transfer cube dataset
to GR00T's LeRobot v2 format for finetuning.

Key Conversions:
- Sampling rate: 50 Hz → 20 Hz
- State dimensions: 14 → 44 (map to GR1 format)
- Image resolution: 480×640 → 256×256
- Video codec: AV1 → H.264
- Add GR00T-specific metadata

Usage:
    python convert_aloha_to_groot.py \
        --input-dataset lerobot/aloha_sim_transfer_cube_human \
        --output-dir ./converted_data/aloha_transfer_cube \
        --target-fps 20
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# Configuration
# =============================================================================

ALOHA_FPS = 50
GROOT_FPS = 20

# ALOHA state/action layout (14 DOF)
ALOHA_LEFT_ARM = slice(0, 6)      # 6 joints
ALOHA_LEFT_GRIPPER = 6            # 1 joint
ALOHA_RIGHT_ARM = slice(7, 13)    # 6 joints
ALOHA_RIGHT_GRIPPER = 13          # 1 joint

# GR00T GR1 state/action layout (44 DOF)
GROOT_LEFT_ARM = slice(0, 7)      # 7 joints
GROOT_LEFT_HAND = slice(7, 13)    # 6 joints
GROOT_LEFT_LEG = slice(13, 19)    # 6 joints (unused)
GROOT_NECK = slice(19, 22)        # 3 joints (unused)
GROOT_RIGHT_ARM = slice(22, 29)   # 7 joints
GROOT_RIGHT_HAND = slice(29, 35)  # 6 joints
GROOT_RIGHT_LEG = slice(35, 41)   # 6 joints (unused)
GROOT_WAIST = slice(41, 44)       # 3 joints

# Task description for ALOHA transfer cube
TASK_DESCRIPTION = "pick up the cube with one hand and transfer it to the other hand"


# =============================================================================
# Dimension Mapping Functions
# =============================================================================

def expand_gripper_to_hand(gripper_value: float, num_fingers: int = 6) -> np.ndarray:
    """
    Expand a single gripper value to multiple hand joint values.

    For Fourier hands, we assume all fingers move together.
    Gripper value is typically in [0, 1] where 0=open, 1=closed.

    Args:
        gripper_value: Single gripper position
        num_fingers: Number of hand joints (default: 6 for Fourier hand)

    Returns:
        Array of hand joint positions
    """
    # Simple replication strategy - all fingers same position
    # Could be enhanced with learned mapping or finger-specific scaling
    return np.full(num_fingers, gripper_value, dtype=np.float32)


def map_aloha_state_to_groot(aloha_state: np.ndarray) -> np.ndarray:
    """
    Map ALOHA 14-DOF state to GR00T 44-DOF state.

    ALOHA layout: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
    GR00T layout: [left_arm(7), left_hand(6), left_leg(6), neck(3),
                   right_arm(7), right_hand(6), right_leg(6), waist(3)]

    Args:
        aloha_state: ALOHA state vector (14,)

    Returns:
        GR00T state vector (44,)
    """
    groot_state = np.zeros(44, dtype=np.float32)

    # Left arm: ALOHA [0:6] → GR00T [0:7]
    # Map 6 joints to first 6 of 7-DOF arm, pad last with 0
    groot_state[0:6] = aloha_state[ALOHA_LEFT_ARM]
    groot_state[6] = 0.0  # 7th joint not present in ALOHA

    # Left hand: ALOHA gripper → GR00T hand [7:13]
    groot_state[GROOT_LEFT_HAND] = expand_gripper_to_hand(aloha_state[ALOHA_LEFT_GRIPPER])

    # Right arm: ALOHA [7:13] → GR00T [22:29]
    groot_state[22:28] = aloha_state[ALOHA_RIGHT_ARM]
    groot_state[28] = 0.0  # 7th joint not present in ALOHA

    # Right hand: ALOHA gripper → GR00T hand [29:35]
    groot_state[GROOT_RIGHT_HAND] = expand_gripper_to_hand(aloha_state[ALOHA_RIGHT_GRIPPER])

    # Unused joints (legs, neck, waist) remain as zeros

    return groot_state


def map_aloha_action_to_groot(aloha_action: np.ndarray) -> np.ndarray:
    """
    Map ALOHA 14-DOF action to GR00T 44-DOF action.
    Same mapping as state.
    """
    return map_aloha_state_to_groot(aloha_action)


# =============================================================================
# Image Processing Functions
# =============================================================================

def transform_image(img: np.ndarray, target_size: int = 256) -> np.ndarray:
    """
    Transform ALOHA image (480×640) to GR00T format (256×256).

    Steps:
    1. Center crop to square
    2. Resize to target size

    Args:
        img: Input image (H, W, 3)
        target_size: Output size (default: 256)

    Returns:
        Transformed image (target_size, target_size, 3)
    """
    h, w = img.shape[:2]

    # Center crop to square
    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    img_cropped = img[start_h:start_h + min_dim, start_w:start_w + min_dim]

    # Resize to target size
    img_resized = cv2.resize(
        img_cropped,
        (target_size, target_size),
        interpolation=cv2.INTER_AREA
    )

    return img_resized


# =============================================================================
# Resampling Functions
# =============================================================================

def compute_resample_indices(
    num_frames: int,
    source_fps: float,
    target_fps: float
) -> np.ndarray:
    """
    Compute frame indices for resampling from source to target FPS.

    Args:
        num_frames: Number of source frames
        source_fps: Source frame rate
        target_fps: Target frame rate

    Returns:
        Array of frame indices to keep
    """
    duration = num_frames / source_fps
    num_target_frames = int(duration * target_fps)
    indices = np.linspace(0, num_frames - 1, num_target_frames).astype(int)
    return indices


def resample_trajectory(
    states: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
    source_fps: float,
    target_fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample trajectory from source FPS to target FPS.

    Args:
        states: State array (N, state_dim)
        actions: Action array (N, action_dim)
        timestamps: Timestamp array (N,)
        source_fps: Source frame rate
        target_fps: Target frame rate

    Returns:
        Resampled (states, actions, timestamps)
    """
    indices = compute_resample_indices(len(states), source_fps, target_fps)

    return states[indices], actions[indices], timestamps[indices]


# =============================================================================
# Video Processing Functions
# =============================================================================

def reencode_video(
    input_path: str,
    output_path: str,
    target_fps: float,
    target_size: int = 256,
    codec: str = "h264"
) -> None:
    """
    Re-encode video with resampling and resizing.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        target_fps: Target frame rate
        target_size: Target resolution (square)
        codec: Output codec
    """
    cap = cv2.VideoCapture(input_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Compute which frames to keep
    indices = compute_resample_indices(num_frames, source_fps, target_fps)
    indices_set = set(indices)

    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        target_fps,
        (target_size, target_size)
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in indices_set:
            # Transform and write frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_transformed = transform_image(frame_rgb, target_size)
            frame_bgr = cv2.cvtColor(frame_transformed, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        frame_idx += 1

    cap.release()
    out.release()


# =============================================================================
# Metadata Generation Functions
# =============================================================================

def create_info_json(
    num_episodes: int,
    total_frames: int,
    fps: float
) -> dict[str, Any]:
    """Create GR00T-compatible info.json."""
    return {
        "codebase_version": "v2.0",
        "robot_type": "ALOHA_to_GR1",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": num_episodes,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {
            "train": f"0:{num_episodes}"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.ego_view": {
                "dtype": "video",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [44],
                "names": [f"motor_{i}" for i in range(44)]
            },
            "action": {
                "dtype": "float32",
                "shape": [44],
                "names": [f"motor_{i}" for i in range(44)]
            },
            "timestamp": {
                "dtype": "float64",
                "shape": [1]
            },
            "annotation.human.action.task_description": {
                "dtype": "int64",
                "shape": [1]
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1]
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1]
            },
            "index": {
                "dtype": "int64",
                "shape": [1]
            },
            "next.done": {
                "dtype": "bool",
                "shape": [1]
            }
        }
    }


def create_modality_json() -> dict[str, Any]:
    """Create GR00T-compatible modality.json for GR1 embodiment."""
    return {
        "state": {
            "left_arm": {"start": 0, "end": 7},
            "left_hand": {"start": 7, "end": 13},
            "left_leg": {"start": 13, "end": 19},
            "neck": {"start": 19, "end": 22},
            "right_arm": {"start": 22, "end": 29},
            "right_hand": {"start": 29, "end": 35},
            "right_leg": {"start": 35, "end": 41},
            "waist": {"start": 41, "end": 44}
        },
        "action": {
            "left_arm": {"start": 0, "end": 7},
            "left_hand": {"start": 7, "end": 13},
            "left_leg": {"start": 13, "end": 19},
            "neck": {"start": 19, "end": 22},
            "right_arm": {"start": 22, "end": 29},
            "right_hand": {"start": 29, "end": 35},
            "right_leg": {"start": 35, "end": 41},
            "waist": {"start": 41, "end": 44}
        },
        "video": {
            "ego_view": {
                "original_key": "observation.images.ego_view",
                "video_backend": "decord"
            }
        },
        "annotation": {
            "human.action.task_description": {},
            "human.validity": {},
            "human.coarse_action": {
                "original_key": "annotation.human.action.task_description"
            }
        }
    }


def create_tasks_jsonl() -> str:
    """Create tasks.jsonl with task descriptions."""
    task = {
        "task_index": 0,
        "task": TASK_DESCRIPTION
    }
    return json.dumps(task)


def compute_statistics(all_states: list[np.ndarray], all_actions: list[np.ndarray]) -> dict:
    """Compute normalization statistics for states and actions."""
    states_concat = np.concatenate(all_states, axis=0)
    actions_concat = np.concatenate(all_actions, axis=0)

    stats = {
        "observation.state": {
            "mean": states_concat.mean(axis=0).tolist(),
            "std": states_concat.std(axis=0).tolist(),
            "min": states_concat.min(axis=0).tolist(),
            "max": states_concat.max(axis=0).tolist()
        },
        "action": {
            "mean": actions_concat.mean(axis=0).tolist(),
            "std": actions_concat.std(axis=0).tolist(),
            "min": actions_concat.min(axis=0).tolist(),
            "max": actions_concat.max(axis=0).tolist()
        }
    }

    return stats


# =============================================================================
# Main Conversion Function
# =============================================================================

def convert_aloha_to_groot(
    input_dataset: str,
    output_dir: str,
    target_fps: float = 20.0,
    target_image_size: int = 256
) -> None:
    """
    Convert ALOHA dataset to GR00T format.

    Args:
        input_dataset: HuggingFace dataset ID or local path
        output_dir: Output directory for converted dataset
        target_fps: Target frame rate (default: 20 Hz)
        target_image_size: Target image size (default: 256)
    """
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    print(f"Loading dataset: {input_dataset}")
    ds = load_dataset(input_dataset, split="train")

    # Download videos from HuggingFace
    print("Downloading videos from HuggingFace...")
    hf_cache_path = snapshot_download(
        repo_id=input_dataset,
        repo_type='dataset',
        allow_patterns=['videos/**/*']
    )
    hf_video_dir = Path(hf_cache_path) / "videos" / "observation.images.top" / "chunk-000"
    print(f"Videos located at: {hf_video_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    (output_path / "meta").mkdir(exist_ok=True)
    (output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_path / "videos" / "chunk-000" / "observation.images.ego_view").mkdir(parents=True, exist_ok=True)

    # Group by episode
    episodes = {}
    for row in ds:
        ep_idx = row["episode_index"]
        if ep_idx not in episodes:
            episodes[ep_idx] = []
        episodes[ep_idx].append(row)

    print(f"Found {len(episodes)} episodes")

    all_states = []
    all_actions = []
    total_frames = 0
    episodes_jsonl = []

    for ep_idx in sorted(episodes.keys()):
        ep_data = episodes[ep_idx]
        print(f"Processing episode {ep_idx}...")

        # Extract arrays
        states = np.array([row["observation.state"] for row in ep_data], dtype=np.float32)
        actions = np.array([row["action"] for row in ep_data], dtype=np.float32)
        timestamps = np.array([row["timestamp"] for row in ep_data], dtype=np.float64)

        # Resample to target FPS
        states_resampled, actions_resampled, timestamps_resampled = resample_trajectory(
            states, actions, timestamps, ALOHA_FPS, target_fps
        )

        # Map dimensions to GR00T format
        states_groot = np.array([map_aloha_state_to_groot(s) for s in states_resampled])
        actions_groot = np.array([map_aloha_action_to_groot(a) for a in actions_resampled])

        all_states.append(states_groot)
        all_actions.append(actions_groot)

        num_frames = len(states_groot)
        total_frames += num_frames

        # Create parquet data
        data = {
            "observation.state": states_groot.tolist(),
            "action": actions_groot.tolist(),
            "timestamp": timestamps_resampled.tolist(),
            "episode_index": [ep_idx] * num_frames,
            "frame_index": list(range(num_frames)),
            "index": list(range(total_frames - num_frames, total_frames)),
            "task_index": [0] * num_frames,
            "annotation.human.action.task_description": [0] * num_frames,
            "next.done": [False] * (num_frames - 1) + [True]
        }

        # Save parquet
        table = pa.table(data)
        parquet_path = output_path / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(table, parquet_path)

        # Episode metadata
        episodes_jsonl.append({
            "episode_index": ep_idx,
            "tasks": [TASK_DESCRIPTION],
            "length": num_frames
        })

        # Video will be extracted from concatenated file after processing all episodes
        # Store episode frame range for video extraction
        ep_data[0]["_video_start_frame"] = sum(len(episodes[i]) for i in range(ep_idx))
        ep_data[0]["_video_num_frames"] = len(ep_data)

        print(f"  Resampled: {len(states)} → {num_frames} frames")

    # Save metadata files
    print("Saving metadata...")

    # info.json
    info = create_info_json(len(episodes), total_frames, target_fps)
    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # modality.json
    modality = create_modality_json()
    with open(output_path / "meta" / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    # tasks.jsonl
    with open(output_path / "meta" / "tasks.jsonl", "w") as f:
        f.write(create_tasks_jsonl() + "\n")

    # episodes.jsonl
    with open(output_path / "meta" / "episodes.jsonl", "w") as f:
        for ep in episodes_jsonl:
            f.write(json.dumps(ep) + "\n")

    # stats.json
    stats = compute_statistics(all_states, all_actions)
    with open(output_path / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Extract videos from concatenated file
    print("\nExtracting videos from concatenated file...")
    concat_video_path = hf_video_dir / "file-000.mp4"

    if concat_video_path.exists():
        cap = cv2.VideoCapture(str(concat_video_path))
        source_fps = cap.get(cv2.CAP_PROP_FPS)

        frame_idx = 0
        for ep_idx in sorted(episodes.keys()):
            ep_data = episodes[ep_idx]
            ep_num_frames = len(ep_data)  # Original frames at 50Hz

            video_output_path = output_path / "videos" / "chunk-000" / "observation.images.ego_view" / f"episode_{ep_idx:06d}.mp4"

            # Compute which frames to keep for this episode
            indices = compute_resample_indices(ep_num_frames, ALOHA_FPS, target_fps)
            indices_set = set(indices)

            # Setup output writer
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(
                str(video_output_path),
                fourcc,
                target_fps,
                (target_image_size, target_image_size)
            )

            # Process frames for this episode
            for local_idx in range(ep_num_frames):
                ret, frame = cap.read()
                if not ret:
                    print(f"  Warning: Could not read frame {frame_idx}")
                    break

                if local_idx in indices_set:
                    # Transform: crop to square and resize
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_transformed = transform_image(frame_rgb, target_image_size)
                    frame_bgr = cv2.cvtColor(frame_transformed, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)

                frame_idx += 1

            out.release()
            print(f"  Episode {ep_idx}: extracted {len(indices)} frames → {video_output_path.name}")

        cap.release()
    else:
        print(f"Warning: Concatenated video not found at {concat_video_path}")

    print(f"\nConversion complete!")
    print(f"  Output directory: {output_path}")
    print(f"  Total episodes: {len(episodes)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Target FPS: {target_fps}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert ALOHA Transfer Cube dataset to GR00T format"
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        default="lerobot/aloha_sim_transfer_cube_human",
        help="HuggingFace dataset ID or local path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./converted_data/aloha_transfer_cube",
        help="Output directory for converted dataset"
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=20.0,
        help="Target frame rate (default: 20 Hz)"
    )
    parser.add_argument(
        "--target-image-size",
        type=int,
        default=256,
        help="Target image size (default: 256)"
    )

    args = parser.parse_args()

    convert_aloha_to_groot(
        input_dataset=args.input_dataset,
        output_dir=args.output_dir,
        target_fps=args.target_fps,
        target_image_size=args.target_image_size
    )


if __name__ == "__main__":
    main()
