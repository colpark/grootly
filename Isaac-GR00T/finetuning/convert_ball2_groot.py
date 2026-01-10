#!/usr/bin/env python3
"""
Convert ball2_groot dataset to GR00T format for finetuning.

This script converts the Trossen AI Mobile bimanual robot dataset (LeRobot v2 format)
to GR00T-compatible format with train/test splits.

Dataset characteristics:
- Robot: Trossen AI Mobile (bimanual mobile robot)
- 3 cameras: cam_high, cam_left_wrist, cam_right_wrist
- State: 19 DOF (odom_x, odom_y, odom_theta, linear_vel, angular_vel, 7 left joints, 7 right joints)
- Action: 16 DOF (linear_vel, angular_vel, 7 left joints, 7 right joints)
- Original FPS: 30 Hz
- Target FPS: 20 Hz (GR00T standard)

Usage:
    python finetuning/convert_ball2_groot.py

    # Custom paths
    python finetuning/convert_ball2_groot.py \
        --input_path ./inhouse/lerobot_dataset/lerobot/recorded_data/ball2_groot \
        --output_path ./data/ball2_groot_format \
        --train_episodes 19 \
        --test_episodes 5
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Source dataset configuration
SOURCE_FPS = 30
TARGET_FPS = 20
TARGET_IMAGE_SIZE = 256

# Camera keys
CAMERA_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

# DOF configuration
STATE_DIM = 19  # odom_x, odom_y, odom_theta, linear_vel, angular_vel, 7 left, 7 right
ACTION_DIM = 16  # linear_vel, angular_vel, 7 left, 7 right

# Action horizon for relative action computation
ACTION_HORIZON = 16

# Index mappings for state and action components
STATE_INDICES = {
    "base_odom": (0, 3),    # 3 DOF
    "base_vel": (3, 5),     # 2 DOF
    "left_arm": (5, 12),    # 7 DOF
    "right_arm": (12, 19),  # 7 DOF
}

ACTION_INDICES = {
    "base_vel": (0, 2),     # 2 DOF - ABSOLUTE
    "left_arm": (2, 9),     # 7 DOF - RELATIVE
    "right_arm": (9, 16),   # 7 DOF - RELATIVE
}

# Components that use relative action representation
RELATIVE_ACTION_KEYS = ["left_arm", "right_arm"]


def compute_resample_indices(num_frames: int, source_fps: int, target_fps: int) -> list[int]:
    """Compute frame indices for temporal resampling."""
    if source_fps == target_fps:
        return list(range(num_frames))

    duration = num_frames / source_fps
    target_frames = int(duration * target_fps)

    indices = []
    for i in range(target_frames):
        t = i / target_fps
        source_idx = int(t * source_fps)
        source_idx = min(source_idx, num_frames - 1)
        indices.append(source_idx)

    return indices


def process_video(
    video_path: Path,
    output_path: Path,
    resample_indices: list[int],
    target_size: int = 256,
    target_fps: int = 20,
) -> bool:
    """Process and resample a video file."""
    try:
        import imageio.v3 as iio

        # Read all frames (handles AV1 codec)
        frames = iio.imread(str(video_path))

        # Select resampled frames
        resampled_frames = [frames[i] for i in resample_indices if i < len(frames)]

        if not resampled_frames:
            return False

        # Setup output writer
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, target_fps, (target_size, target_size))

        for frame in resampled_frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Center crop and resize
            h, w = frame_bgr.shape[:2]
            min_dim = min(h, w)
            start_h = (h - min_dim) // 2
            start_w = (w - min_dim) // 2
            cropped = frame_bgr[start_h:start_h + min_dim, start_w:start_w + min_dim]
            resized = cv2.resize(cropped, (target_size, target_size))

            out.write(resized)

        out.release()
        return True

    except Exception as e:
        print(f"  Error processing video {video_path}: {e}")
        return False


def convert_episode(
    input_path: Path,
    output_path: Path,
    episode_idx: int,
    new_episode_idx: int,
    task_description: str,
) -> tuple[dict | None, np.ndarray | None, np.ndarray | None]:
    """Convert a single episode to GR00T format. Returns episode_info, states, actions."""
    # Read parquet data
    parquet_path = input_path / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    if not parquet_path.exists():
        print(f"  Parquet not found: {parquet_path}")
        return None, None, None

    df = pd.read_parquet(parquet_path)
    num_frames = len(df)

    # Compute resample indices
    resample_indices = compute_resample_indices(num_frames, SOURCE_FPS, TARGET_FPS)
    resampled_frames = len(resample_indices)

    print(f"  Episode {episode_idx} -> {new_episode_idx}: {num_frames} frames @ {SOURCE_FPS}Hz -> {resampled_frames} frames @ {TARGET_FPS}Hz")

    # Process videos for all cameras
    for cam_key in CAMERA_KEYS:
        video_in = input_path / "videos" / "chunk-000" / cam_key / f"episode_{episode_idx:06d}.mp4"
        video_out = output_path / "videos" / "chunk-000" / f"observation.images.{cam_key}" / f"episode_{new_episode_idx:06d}.mp4"

        if video_in.exists():
            success = process_video(video_in, video_out, resample_indices, TARGET_IMAGE_SIZE, TARGET_FPS)
            if not success:
                print(f"    Failed to process {cam_key}")
        else:
            print(f"    Video not found: {video_in}")

    # Process state and action data
    resampled_df = df.iloc[resample_indices].copy()

    # Extract state (19 DOF) and action (16 DOF) as concatenated arrays
    state_data = None
    action_data = None

    # Source uses "observation.state" column name
    if "observation.state" in resampled_df.columns:
        state_data = np.vstack(resampled_df["observation.state"].values)  # Shape: (T, 19)
    elif "observation" in resampled_df.columns:
        state_data = np.vstack(resampled_df["observation"].values)  # Shape: (T, 19)

    if "action" in resampled_df.columns:
        action_data = np.vstack(resampled_df["action"].values)  # Shape: (T, 16)

    # Build new dataframe with GR00T expected format
    new_data = {
        "frame_index": list(range(resampled_frames)),
        "episode_index": [new_episode_idx] * resampled_frames,
        "index": list(range(resampled_frames)),
        "timestamp": [i / TARGET_FPS for i in range(resampled_frames)],
        "task_index": [0] * resampled_frames,
        "task": [task_description] * resampled_frames,
    }

    # Store concatenated state and action (what GR00T loader expects)
    if state_data is not None:
        new_data["observation.state"] = [state_data[i].tolist() for i in range(resampled_frames)]

    if action_data is not None:
        new_data["action"] = [action_data[i].tolist() for i in range(resampled_frames)]

    # Save parquet
    out_parquet = output_path / "data" / "chunk-000" / f"episode_{new_episode_idx:06d}.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(new_data)
    new_df.to_parquet(out_parquet)

    episode_info = {
        "episode_index": new_episode_idx,
        "num_frames": resampled_frames,
        "original_episode": episode_idx,
    }

    return episode_info, state_data, action_data


def compute_statistics(all_states: list[np.ndarray], all_actions: list[np.ndarray]) -> dict:
    """Compute dataset statistics for normalization."""
    # Concatenate all episodes
    states_concat = np.concatenate(all_states, axis=0)  # Shape: (total_frames, 19)
    actions_concat = np.concatenate(all_actions, axis=0)  # Shape: (total_frames, 16)

    stats = {
        "observation.state": {
            "mean": states_concat.mean(axis=0).tolist(),
            "std": states_concat.std(axis=0).tolist(),
            "min": states_concat.min(axis=0).tolist(),
            "max": states_concat.max(axis=0).tolist(),
            "q01": np.percentile(states_concat, 1, axis=0).tolist(),
            "q99": np.percentile(states_concat, 99, axis=0).tolist(),
        },
        "action": {
            "mean": actions_concat.mean(axis=0).tolist(),
            "std": actions_concat.std(axis=0).tolist(),
            "min": actions_concat.min(axis=0).tolist(),
            "max": actions_concat.max(axis=0).tolist(),
            "q01": np.percentile(actions_concat, 1, axis=0).tolist(),
            "q99": np.percentile(actions_concat, 99, axis=0).tolist(),
        }
    }

    return stats


def compute_relative_action_statistics(
    all_states: list[np.ndarray],
    all_actions: list[np.ndarray],
    action_horizon: int = ACTION_HORIZON,
) -> dict:
    """
    Compute relative action statistics for components using RELATIVE representation.

    Relative action = action[t+i] - state[t] for i in range(action_horizon)

    This is required for GR00T finetuning when using ActionRepresentation.RELATIVE.

    Args:
        all_states: List of state arrays, each shape (T, STATE_DIM)
        all_actions: List of action arrays, each shape (T, ACTION_DIM)
        action_horizon: Number of action steps to predict (default: 16)

    Returns:
        dict mapping component name to statistics with shape (action_horizon, component_dim)
    """
    relative_stats = {}

    for key in RELATIVE_ACTION_KEYS:
        state_start, state_end = STATE_INDICES[key]
        action_start, action_end = ACTION_INDICES[key]
        component_dim = state_end - state_start

        # Collect all relative action chunks
        all_relative_chunks = []

        for states, actions in zip(all_states, all_actions):
            episode_length = len(states)
            # We can only compute relative actions for timesteps where we have
            # enough future actions to fill the action horizon
            usable_length = episode_length - action_horizon

            for t in range(usable_length):
                # Current state for this component
                current_state = states[t, state_start:state_end]  # Shape: (component_dim,)

                # Future actions for this component over the action horizon
                future_actions = actions[t:t + action_horizon, action_start:action_end]  # Shape: (horizon, component_dim)

                # Compute relative action: action - current_state
                relative_chunk = future_actions - current_state  # Broadcasting: (horizon, dim) - (dim,) = (horizon, dim)
                all_relative_chunks.append(relative_chunk)

        if not all_relative_chunks:
            print(f"  Warning: No relative chunks computed for {key}")
            continue

        # Stack all chunks: shape (num_chunks, action_horizon, component_dim)
        all_chunks = np.stack(all_relative_chunks, axis=0)

        # Compute statistics across all chunks (axis=0)
        # Result shape: (action_horizon, component_dim)
        relative_stats[key] = {
            "mean": np.mean(all_chunks, axis=0).tolist(),
            "std": np.std(all_chunks, axis=0).tolist(),
            "min": np.min(all_chunks, axis=0).tolist(),
            "max": np.max(all_chunks, axis=0).tolist(),
            "q01": np.percentile(all_chunks, 1, axis=0).tolist(),
            "q99": np.percentile(all_chunks, 99, axis=0).tolist(),
        }

        print(f"  Computed relative stats for {key}: {len(all_chunks)} chunks, shape ({action_horizon}, {component_dim})")

    return relative_stats


def create_metadata(
    output_path: Path,
    episode_infos: list[dict],
    task_description: str,
    stats: dict,
    relative_stats: dict,
    split_name: str,
):
    """Create metadata files for the dataset."""
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_frames = sum(ep["num_frames"] for ep in episode_infos)
    num_episodes = len(episode_infos)

    # info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "trossen_ai_mobile",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": num_episodes * len(CAMERA_KEYS),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": TARGET_FPS,
        "splits": {
            split_name: f"0:{num_episodes}"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
            "action": {"dtype": "float32", "shape": [ACTION_DIM]},
            **{f"observation.images.{cam}": {
                "dtype": "video",
                "shape": [TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, 3],
                "info": {
                    "video.fps": float(TARGET_FPS),
                    "video.height": TARGET_IMAGE_SIZE,
                    "video.width": TARGET_IMAGE_SIZE,
                    "video.codec": "mp4v",
                }
            } for cam in CAMERA_KEYS},
        }
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episode_infos:
            f.write(json.dumps({
                "episode_index": ep["episode_index"],
                "tasks": [task_description],
                "length": ep["num_frames"],
            }) + "\n")

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({
            "task_index": 0,
            "task": task_description,
        }) + "\n")

    # stats.json
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # relative_stats.json - Required for RELATIVE action representation
    if relative_stats:
        with open(meta_dir / "relative_stats.json", "w") as f:
            json.dump(relative_stats, f, indent=2)
        print(f"  Created relative_stats.json with keys: {list(relative_stats.keys())}")

    # modality.json - Required by GR00T for dimension mapping
    # Maps component names to ranges within concatenated state/action vectors
    modality = {
        "state": {
            # State: 19 DOF total
            "base_odom": {"start": 0, "end": 3},   # 3 DOF: odom_x, odom_y, odom_theta
            "base_vel": {"start": 3, "end": 5},    # 2 DOF: linear_vel, angular_vel
            "left_arm": {"start": 5, "end": 12},   # 7 DOF: joint positions
            "right_arm": {"start": 12, "end": 19}, # 7 DOF: joint positions
        },
        "action": {
            # Action: 16 DOF total
            "base_vel": {"start": 0, "end": 2},    # 2 DOF: linear_vel, angular_vel
            "left_arm": {"start": 2, "end": 9},    # 7 DOF: joint commands
            "right_arm": {"start": 9, "end": 16},  # 7 DOF: joint commands
        },
        "video": {
            "cam_high": {
                "original_key": "observation.images.cam_high",
                "video_backend": "decord"
            },
            "cam_left_wrist": {
                "original_key": "observation.images.cam_left_wrist",
                "video_backend": "decord"
            },
            "cam_right_wrist": {
                "original_key": "observation.images.cam_right_wrist",
                "video_backend": "decord"
            },
        },
        "annotation": {
            "human.action.task_description": {},
            "human.validity": {},
        }
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    print(f"  Created metadata: {num_episodes} episodes, {total_frames} frames")


def main():
    parser = argparse.ArgumentParser(description="Convert ball2_groot to GR00T format")
    parser.add_argument(
        "--input_path",
        type=str,
        default="./inhouse/lerobot_dataset/lerobot/recorded_data/ball2_groot",
        help="Path to source ball2_groot dataset",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="./data",
        help="Base path for output datasets",
    )
    parser.add_argument(
        "--train_episodes",
        type=int,
        default=19,
        help="Number of training episodes (default: 19)",
    )
    parser.add_argument(
        "--test_episodes",
        type=int,
        default=5,
        help="Number of test episodes (default: 5)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="./conversion_log.txt",
        help="Path to log file for debugging",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_base = Path(args.output_base)

    # Setup logging
    import sys
    log_file = open(args.log_file, "w")
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, text):
            for f in self.files:
                f.write(text)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    sys.stdout = TeeOutput(sys.stdout, log_file)
    sys.stderr = TeeOutput(sys.stderr, log_file)

    print(f"Ball2 GR00T Conversion Log")
    print(f"=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_base}")
    print(f"Train episodes: {args.train_episodes}")
    print(f"Test episodes: {args.test_episodes}")
    print(f"=" * 60)

    # Read task description
    tasks_file = input_path / "meta" / "tasks.jsonl"
    with open(tasks_file) as f:
        task_info = json.loads(f.readline())
        task_description = task_info["task"]

    print(f"Task: {task_description}")

    # Read episode info
    info_file = input_path / "meta" / "info.json"
    with open(info_file) as f:
        info = json.load(f)

    total_episodes = info["total_episodes"]
    print(f"Total source episodes: {total_episodes}")

    # Determine episode splits
    train_eps = list(range(args.train_episodes))
    test_start = args.train_episodes
    test_eps = list(range(test_start, min(test_start + args.test_episodes, total_episodes)))

    print(f"Train episodes: {train_eps} ({len(train_eps)} episodes)")
    print(f"Test episodes: {test_eps} ({len(test_eps)} episodes)")

    # Convert training set
    print("\n" + "=" * 60)
    print("Converting TRAINING set...")
    print("=" * 60)

    train_output = output_base / "ball2_groot_train"
    if train_output.exists():
        shutil.rmtree(train_output)

    train_infos = []
    train_states = []
    train_actions = []

    for new_idx, orig_idx in enumerate(train_eps):
        ep_info, states, actions = convert_episode(input_path, train_output, orig_idx, new_idx, task_description)
        if ep_info:
            train_infos.append(ep_info)
            if states is not None:
                train_states.append(states)
            if actions is not None:
                train_actions.append(actions)

    if train_states and train_actions:
        train_stats = compute_statistics(train_states, train_actions)
        print("\nComputing relative action statistics for training set...")
        train_relative_stats = compute_relative_action_statistics(train_states, train_actions)
    else:
        train_stats = {}
        train_relative_stats = {}
    create_metadata(train_output, train_infos, task_description, train_stats, train_relative_stats, "train")

    # Convert test set
    print("\n" + "=" * 60)
    print("Converting TEST set...")
    print("=" * 60)

    test_output = output_base / "ball2_groot_test"
    if test_output.exists():
        shutil.rmtree(test_output)

    test_infos = []
    test_states = []
    test_actions = []

    for new_idx, orig_idx in enumerate(test_eps):
        ep_info, states, actions = convert_episode(input_path, test_output, orig_idx, new_idx, task_description)
        if ep_info:
            test_infos.append(ep_info)
            if states is not None:
                test_states.append(states)
            if actions is not None:
                test_actions.append(actions)

    if test_states and test_actions:
        test_stats = compute_statistics(test_states, test_actions)
        print("\nComputing relative action statistics for test set...")
        test_relative_stats = compute_relative_action_statistics(test_states, test_actions)
    else:
        test_stats = {}
        test_relative_stats = {}
    create_metadata(test_output, test_infos, task_description, test_stats, test_relative_stats, "test")

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Training data: {train_output}")
    print(f"  Episodes: {len(train_infos)}")
    print(f"  Total frames: {sum(ep['num_frames'] for ep in train_infos)}")
    print(f"Test data: {test_output}")
    print(f"  Episodes: {len(test_infos)}")
    print(f"  Total frames: {sum(ep['num_frames'] for ep in test_infos)}")
    print(f"\nLog saved to: {args.log_file}")

    log_file.close()


if __name__ == "__main__":
    main()
