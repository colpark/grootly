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
) -> dict | None:
    """Convert a single episode to GR00T format."""
    # Read parquet data
    parquet_path = input_path / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    if not parquet_path.exists():
        print(f"  Parquet not found: {parquet_path}")
        return None

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

    # Build new dataframe
    new_data = {
        "frame_index": list(range(resampled_frames)),
        "episode_index": [new_episode_idx] * resampled_frames,
        "index": list(range(resampled_frames)),  # Will be updated later for global index
        "timestamp": [i / TARGET_FPS for i in range(resampled_frames)],
        "task_index": [0] * resampled_frames,
        "task": [task_description] * resampled_frames,
    }

    # State observation (19 DOF)
    if "observation" in resampled_df.columns:
        state_data = np.vstack(resampled_df["observation"].values)
        # Split into components for GR00T format
        new_data["state.base_odom"] = [state_data[i, :3].tolist() for i in range(resampled_frames)]  # odom_x, odom_y, odom_theta
        new_data["state.base_vel"] = [state_data[i, 3:5].tolist() for i in range(resampled_frames)]  # linear_vel, angular_vel
        new_data["state.left_arm"] = [state_data[i, 5:12].tolist() for i in range(resampled_frames)]  # 7 left joints
        new_data["state.right_arm"] = [state_data[i, 12:19].tolist() for i in range(resampled_frames)]  # 7 right joints

    # Action (16 DOF)
    if "action" in resampled_df.columns:
        action_data = np.vstack(resampled_df["action"].values)
        new_data["action.base_vel"] = [action_data[i, :2].tolist() for i in range(resampled_frames)]  # linear_vel, angular_vel
        new_data["action.left_arm"] = [action_data[i, 2:9].tolist() for i in range(resampled_frames)]  # 7 left joints
        new_data["action.right_arm"] = [action_data[i, 9:16].tolist() for i in range(resampled_frames)]  # 7 right joints

    # Save parquet
    out_parquet = output_path / "data" / "chunk-000" / f"episode_{new_episode_idx:06d}.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(new_data)
    new_df.to_parquet(out_parquet)

    return {
        "episode_index": new_episode_idx,
        "num_frames": resampled_frames,
        "original_episode": episode_idx,
    }


def compute_statistics(output_path: Path) -> dict:
    """Compute dataset statistics for normalization."""
    all_data = {
        "state.base_odom": [],
        "state.base_vel": [],
        "state.left_arm": [],
        "state.right_arm": [],
        "action.base_vel": [],
        "action.left_arm": [],
        "action.right_arm": [],
    }

    parquet_dir = output_path / "data" / "chunk-000"
    for parquet_file in sorted(parquet_dir.glob("episode_*.parquet")):
        df = pd.read_parquet(parquet_file)
        for key in all_data.keys():
            if key in df.columns:
                all_data[key].extend(df[key].tolist())

    stats = {}
    for key, values in all_data.items():
        if values:
            arr = np.array(values)
            stats[key] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
                "q01": np.percentile(arr, 1, axis=0).tolist(),
                "q99": np.percentile(arr, 99, axis=0).tolist(),
            }

    return stats


def create_metadata(
    output_path: Path,
    episode_infos: list[dict],
    task_description: str,
    stats: dict,
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
            "state.base_odom": {"dtype": "float32", "shape": [3]},
            "state.base_vel": {"dtype": "float32", "shape": [2]},
            "state.left_arm": {"dtype": "float32", "shape": [7]},
            "state.right_arm": {"dtype": "float32", "shape": [7]},
            "action.base_vel": {"dtype": "float32", "shape": [2]},
            "action.left_arm": {"dtype": "float32", "shape": [7]},
            "action.right_arm": {"dtype": "float32", "shape": [7]},
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
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_base = Path(args.output_base)

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
    for new_idx, orig_idx in enumerate(train_eps):
        ep_info = convert_episode(input_path, train_output, orig_idx, new_idx, task_description)
        if ep_info:
            train_infos.append(ep_info)

    train_stats = compute_statistics(train_output)
    create_metadata(train_output, train_infos, task_description, train_stats, "train")

    # Convert test set
    print("\n" + "=" * 60)
    print("Converting TEST set...")
    print("=" * 60)

    test_output = output_base / "ball2_groot_test"
    if test_output.exists():
        shutil.rmtree(test_output)

    test_infos = []
    for new_idx, orig_idx in enumerate(test_eps):
        ep_info = convert_episode(input_path, test_output, orig_idx, new_idx, task_description)
        if ep_info:
            test_infos.append(ep_info)

    test_stats = compute_statistics(test_output)
    create_metadata(test_output, test_infos, task_description, test_stats, "test")

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Training data: {train_output}")
    print(f"  Episodes: {len(train_infos)}")
    print(f"  Total frames: {sum(ep['num_frames'] for ep in train_infos)}")
    print(f"Test data: {test_output}")
    print(f"  Episodes: {len(test_infos)}")
    print(f"  Total frames: {sum(ep['num_frames'] for ep in test_infos)}")


if __name__ == "__main__":
    main()
