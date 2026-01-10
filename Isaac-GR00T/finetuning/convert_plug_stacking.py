#!/usr/bin/env python3
"""
Convert plug_stacking dataset (v1-v5) to GR00T format for finetuning.

This script converts the Trossen AI Mobile bimanual robot plug stacking dataset
(LeRobot v2 format, 5 versions) to GR00T-compatible format with train/test splits.

Dataset characteristics:
- Robot: Trossen AI Mobile (bimanual mobile robot) - same as ball2_groot
- 5 versions: v1(3), v2(6), v3(17), v4(10), v5(20) = 56 total episodes
- 3 cameras: cam_high, cam_left_wrist, cam_right_wrist
- State: 19 DOF (odom_x, odom_y, odom_theta, linear_vel, angular_vel, 7 left joints, 7 right joints)
- Action: 16 DOF (linear_vel, angular_vel, 7 left joints, 7 right joints)
- Original FPS: 30 Hz
- Target FPS: 20 Hz (GR00T standard)

Usage:
    python finetuning/convert_plug_stacking.py

    # Custom settings
    python finetuning/convert_plug_stacking.py \
        --input_base ./inhouse/lerobot_dataset/lerobot/recorded_data/plug_stacking_data \
        --output_base ./data \
        --test_episodes 10
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Source dataset configuration
SOURCE_FPS = 30
TARGET_FPS = 20
TARGET_IMAGE_SIZE = 256

# Camera keys (same as ball2_groot)
CAMERA_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

# DOF configuration (same as ball2_groot)
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

# Version directories
VERSIONS = ["plug_stacking_v1", "plug_stacking_v2", "plug_stacking_v3",
            "plug_stacking_v4", "plug_stacking_v5"]


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
    source_path: Path,
    output_path: Path,
    source_episode_idx: int,
    new_episode_idx: int,
    task_description: str,
) -> tuple[dict | None, np.ndarray | None, np.ndarray | None]:
    """Convert a single episode to GR00T format. Returns episode_info, states, actions."""
    # Read parquet data
    parquet_path = source_path / "data" / "chunk-000" / f"episode_{source_episode_idx:06d}.parquet"
    if not parquet_path.exists():
        print(f"  Parquet not found: {parquet_path}")
        return None, None, None

    df = pd.read_parquet(parquet_path)
    num_frames = len(df)

    # Compute resample indices
    resample_indices = compute_resample_indices(num_frames, SOURCE_FPS, TARGET_FPS)
    resampled_frames = len(resample_indices)

    print(f"  Episode {source_episode_idx} -> {new_episode_idx}: {num_frames} frames @ {SOURCE_FPS}Hz -> {resampled_frames} frames @ {TARGET_FPS}Hz")

    # Process videos for all cameras
    for cam_key in CAMERA_KEYS:
        video_in = source_path / "videos" / "chunk-000" / f"observation.images.{cam_key}" / f"episode_{source_episode_idx:06d}.mp4"
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
        "source_version": str(source_path.name),
        "source_episode": source_episode_idx,
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
            usable_length = episode_length - action_horizon

            for t in range(usable_length):
                current_state = states[t, state_start:state_end]
                future_actions = actions[t:t + action_horizon, action_start:action_end]
                relative_chunk = future_actions - current_state
                all_relative_chunks.append(relative_chunk)

        if not all_relative_chunks:
            print(f"  Warning: No relative chunks computed for {key}")
            continue

        all_chunks = np.stack(all_relative_chunks, axis=0)

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
                "source_version": ep.get("source_version", "unknown"),
                "source_episode": ep.get("source_episode", -1),
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

    # relative_stats.json
    if relative_stats:
        with open(meta_dir / "relative_stats.json", "w") as f:
            json.dump(relative_stats, f, indent=2)
        print(f"  Created relative_stats.json with keys: {list(relative_stats.keys())}")

    # modality.json - Required by GR00T for dimension mapping
    modality = {
        "state": {
            "base_odom": {"start": 0, "end": 3},
            "base_vel": {"start": 3, "end": 5},
            "left_arm": {"start": 5, "end": 12},
            "right_arm": {"start": 12, "end": 19},
        },
        "action": {
            "base_vel": {"start": 0, "end": 2},
            "left_arm": {"start": 2, "end": 9},
            "right_arm": {"start": 9, "end": 16},
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


def collect_all_episodes(input_base: Path) -> list[tuple[Path, int]]:
    """Collect all episode references from all versions."""
    all_episodes = []

    for version in VERSIONS:
        version_path = input_base / version
        if not version_path.exists():
            print(f"Warning: Version not found: {version_path}")
            continue

        # Read info to get episode count
        info_file = version_path / "meta" / "info.json"
        if not info_file.exists():
            print(f"Warning: info.json not found for {version}")
            continue

        with open(info_file) as f:
            info = json.load(f)

        num_episodes = info["total_episodes"]

        for ep_idx in range(num_episodes):
            all_episodes.append((version_path, ep_idx))

        print(f"  {version}: {num_episodes} episodes")

    return all_episodes


def main():
    parser = argparse.ArgumentParser(description="Convert plug_stacking (v1-v5) to GR00T format")
    parser.add_argument(
        "--input_base",
        type=str,
        default="./inhouse/lerobot_dataset/lerobot/recorded_data/plug_stacking_data",
        help="Base path containing plug_stacking_v1 through v5",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="./data",
        help="Base path for output datasets",
    )
    parser.add_argument(
        "--test_episodes",
        type=int,
        default=10,
        help="Number of test episodes (randomly selected, default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="./conversion_plug_stacking_log.txt",
        help="Path to log file for debugging",
    )
    args = parser.parse_args()

    input_base = Path(args.input_base)
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

    print(f"Plug Stacking GR00T Conversion Log")
    print(f"=" * 60)
    print(f"Input base: {input_base}")
    print(f"Output: {output_base}")
    print(f"Test episodes: {args.test_episodes}")
    print(f"Random seed: {args.seed}")
    print(f"=" * 60)

    # Task description
    task_description = "Plug stacking"

    # Collect all episodes from all versions
    print("\nCollecting episodes from all versions...")
    all_episodes = collect_all_episodes(input_base)
    total_episodes = len(all_episodes)
    print(f"Total episodes across all versions: {total_episodes}")

    if total_episodes == 0:
        print("ERROR: No episodes found!")
        sys.exit(1)

    if args.test_episodes >= total_episodes:
        print(f"ERROR: test_episodes ({args.test_episodes}) >= total ({total_episodes})")
        sys.exit(1)

    # Random train/test split
    random.seed(args.seed)
    all_indices = list(range(total_episodes))
    random.shuffle(all_indices)

    test_indices = set(all_indices[:args.test_episodes])
    train_indices = [i for i in range(total_episodes) if i not in test_indices]

    print(f"\nSplit: {len(train_indices)} train / {len(test_indices)} test")
    print(f"Test episode indices (in combined list): {sorted(test_indices)}")

    # Convert training set
    print("\n" + "=" * 60)
    print("Converting TRAINING set...")
    print("=" * 60)

    train_output = output_base / "plug_stacking_train"
    if train_output.exists():
        shutil.rmtree(train_output)

    train_infos = []
    train_states = []
    train_actions = []

    for new_idx, orig_idx in enumerate(train_indices):
        source_path, source_ep_idx = all_episodes[orig_idx]
        print(f"\n[Train {new_idx}] From {source_path.name} episode {source_ep_idx}")

        ep_info, states, actions = convert_episode(
            source_path, train_output, source_ep_idx, new_idx, task_description
        )
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

    test_output = output_base / "plug_stacking_test"
    if test_output.exists():
        shutil.rmtree(test_output)

    test_infos = []
    test_states = []
    test_actions = []

    for new_idx, orig_idx in enumerate(sorted(test_indices)):
        source_path, source_ep_idx = all_episodes[orig_idx]
        print(f"\n[Test {new_idx}] From {source_path.name} episode {source_ep_idx}")

        ep_info, states, actions = convert_episode(
            source_path, test_output, source_ep_idx, new_idx, task_description
        )
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

    # Print source breakdown
    print("\nSource breakdown (training):")
    version_counts = {}
    for ep in train_infos:
        v = ep.get("source_version", "unknown")
        version_counts[v] = version_counts.get(v, 0) + 1
    for v, c in sorted(version_counts.items()):
        print(f"  {v}: {c} episodes")

    print("\nSource breakdown (test):")
    version_counts = {}
    for ep in test_infos:
        v = ep.get("source_version", "unknown")
        version_counts[v] = version_counts.get(v, 0) + 1
    for v, c in sorted(version_counts.items()):
        print(f"  {v}: {c} episodes")

    print(f"\nLog saved to: {args.log_file}")

    log_file.close()


if __name__ == "__main__":
    main()
