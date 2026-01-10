#!/usr/bin/env python3
"""
Evaluation script for plug_stacking-finetuned GR00T model.

This script:
1. Loads the finetuned checkpoint
2. Runs open-loop evaluation on the plug_stacking test dataset
3. Generates visualization plots comparing predicted vs ground truth actions
4. Handles 3-camera setup (cam_high, cam_left_wrist, cam_right_wrist)

Robot Configuration (Trossen AI Mobile - same as ball2_groot):
- State: 19 DOF (base_odom[3], base_vel[2], left_arm[7], right_arm[7])
- Action: 16 DOF (base_vel[2], left_arm[7], right_arm[7])

Usage:
    python finetuning/evaluate_plug_stacking.py \
        --checkpoint_path ./outputs/plug_stacking/checkpoint-1000 \
        --dataset_path ./data/plug_stacking_test \
        --output_dir ./eval_results/plug_stacking
"""

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import to register GR00T model with transformers
import gr00t.model.gr00t_n1d6.gr00t_n1d6  # noqa: F401

# Import modality config registration (same as ball2_groot - Trossen AI Mobile)
from finetuning.trossen_modality_config import TROSSEN_MOBILE_CONFIG  # noqa: F401

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Action dimension labels for plug_stacking (Trossen AI Mobile bimanual robot)
# Base: velocities, Arms: joint positions in radians
ACTION_LABELS = [
    # Base velocity (2 DOF)
    "base_linear_vel (m/s)",
    "base_angular_vel (rad/s)",
    # Left arm joints (7 DOF) - typical 7-DOF arm configuration
    "L_waist (rad)",
    "L_shoulder (rad)",
    "L_elbow (rad)",
    "L_forearm_roll (rad)",
    "L_wrist_pitch (rad)",
    "L_wrist_roll (rad)",
    "L_gripper (rad)",
    # Right arm joints (7 DOF)
    "R_waist (rad)",
    "R_shoulder (rad)",
    "R_elbow (rad)",
    "R_forearm_roll (rad)",
    "R_wrist_pitch (rad)",
    "R_wrist_roll (rad)",
    "R_gripper (rad)",
]

# State dimension labels for plug_stacking
STATE_LABELS = [
    # Base odometry (3 DOF)
    "odom_x (m)",
    "odom_y (m)",
    "odom_theta (rad)",
    # Base velocity (2 DOF)
    "base_linear_vel (m/s)",
    "base_angular_vel (rad/s)",
    # Left arm joints (7 DOF)
    "L_waist (rad)",
    "L_shoulder (rad)",
    "L_elbow (rad)",
    "L_forearm_roll (rad)",
    "L_wrist_pitch (rad)",
    "L_wrist_roll (rad)",
    "L_gripper (rad)",
    # Right arm joints (7 DOF)
    "R_waist (rad)",
    "R_shoulder (rad)",
    "R_elbow (rad)",
    "R_forearm_roll (rad)",
    "R_wrist_pitch (rad)",
    "R_wrist_roll (rad)",
    "R_gripper (rad)",
]


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint in the output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = list(output_path.glob("checkpoint-*"))
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
    return str(checkpoints[-1])


def plot_trajectory_comparison(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    traj_id: int,
    action_horizon: int,
    save_path: str,
) -> None:
    """Plot comparison of ground truth vs predicted actions."""
    actual_steps = len(gt_actions)
    action_dim = gt_actions.shape[1]

    # Create subplot grid
    n_cols = 4
    n_rows = (action_dim + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    fig.suptitle(f"Trajectory {traj_id} - Predicted vs Ground Truth Actions\n(Trossen AI Mobile - Plug Stacking Task)", fontsize=16)

    for dim_idx in range(min(action_dim, len(axes))):
        ax = axes[dim_idx]

        ax.plot(gt_actions[:, dim_idx], 'b-', label='Ground Truth', linewidth=2)
        ax.plot(pred_actions[:, dim_idx], 'r--', label='Predicted', linewidth=2)

        # Mark inference points
        for j in range(0, actual_steps, action_horizon):
            ax.axvline(x=j, color='gray', linestyle=':', alpha=0.3)

        label = ACTION_LABELS[dim_idx] if dim_idx < len(ACTION_LABELS) else f"Dim {dim_idx}"
        ax.set_title(label)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action Value')
        if dim_idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(action_dim, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot to {save_path}")


def plot_multi_camera_comparison(
    dataset_path: str,
    traj_id: int,
    step_idx: int,
    save_path: str,
) -> None:
    """Plot side-by-side view of all 3 cameras at a given step."""
    try:
        import cv2
        import imageio.v3 as iio
    except ImportError:
        logger.warning("imageio or cv2 not available, skipping camera visualization")
        return

    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    camera_titles = ["Central Camera (High)", "Left Wrist Camera", "Right Wrist Camera"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Trajectory {traj_id} - Step {step_idx} - Multi-Camera View", fontsize=14)

    for idx, (cam, title) in enumerate(zip(cameras, camera_titles)):
        video_path = Path(dataset_path) / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{traj_id:06d}.mp4"

        if video_path.exists():
            try:
                frames = iio.imread(str(video_path))
                if step_idx < len(frames):
                    axes[idx].imshow(frames[step_idx])
                    axes[idx].set_title(title)
                else:
                    axes[idx].text(0.5, 0.5, f"Step {step_idx} out of range", ha='center', va='center')
            except Exception as e:
                axes[idx].text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
        else:
            axes[idx].text(0.5, 0.5, "Video not found", ha='center', va='center')

        axes[idx].axis('off')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved multi-camera plot to {save_path}")


def create_video_visualization(
    dataset_path: str,
    traj_id: int,
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    save_path: str,
    fps: int = 20,
) -> None:
    """Create a video showing all 3 cameras with action overlay."""
    try:
        import cv2
        import imageio.v3 as iio
    except ImportError:
        logger.warning("imageio or cv2 not available, skipping video generation")
        return

    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    all_frames = {}

    # Load all camera videos
    min_frames = float('inf')
    for cam in cameras:
        video_path = Path(dataset_path) / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{traj_id:06d}.mp4"
        if video_path.exists():
            try:
                frames = iio.imread(str(video_path))
                all_frames[cam] = frames
                min_frames = min(min_frames, len(frames))
            except Exception as e:
                logger.warning(f"Could not read {cam} video: {e}")

    if not all_frames:
        logger.warning(f"No videos found for trajectory {traj_id}")
        return

    # Create output video
    frame_height = 256
    frame_width = 256
    canvas_width = frame_width * 3  # 3 cameras side by side
    canvas_height = frame_height + 150  # Extra space for action info

    output_frames = []

    for i in range(int(min_frames)):
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Place camera frames
        for cam_idx, cam in enumerate(cameras):
            if cam in all_frames:
                frame = all_frames[cam][i]
                x_offset = cam_idx * frame_width
                canvas[:frame_height, x_offset:x_offset + frame_width] = frame

        # Add text overlay
        cv2.putText(canvas, f"Frame {i}/{int(min_frames)} | Cam: High | Left Wrist | Right Wrist",
                    (10, frame_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if i < len(gt_actions) and i < len(pred_actions):
            y_offset = frame_height + 50
            # Show base velocity and first arm joint for each arm
            dims_to_show = [0, 1, 2, 9]  # base_lin, base_ang, left_arm_0, right_arm_0
            labels = ["Base Lin", "Base Ang", "Left Arm 0", "Right Arm 0"]

            for dim_idx, label in zip(dims_to_show, labels):
                if dim_idx < gt_actions.shape[1]:
                    gt_val = gt_actions[i, dim_idx]
                    pred_val = pred_actions[i, dim_idx]
                    error = abs(gt_val - pred_val)

                    text = f"{label}: GT={gt_val:.3f} Pred={pred_val:.3f} Err={error:.3f}"
                    color = (0, 255, 0) if error < 0.1 else (0, 165, 255) if error < 0.2 else (0, 0, 255)
                    cv2.putText(canvas, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    y_offset += 22

        output_frames.append(canvas)

    # Write output video
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(save_path), fourcc, fps, (canvas_width, canvas_height))

    for frame in output_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    logger.info(f"Saved video to {save_path}")


def parse_observation_gr00t(obs: dict, modality_configs: dict) -> dict:
    """Parse observation into GR00T format."""
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"
            arr = obs[parsed_key]
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def parse_action_gr00t(action: dict) -> dict:
    """Parse GR00T action output."""
    return {f"action.{key}": action[key][0] for key in action}


def compute_success_metrics(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    thresholds: list[float] = [0.05, 0.1, 0.2],
) -> dict:
    """Compute proxy success metrics based on action prediction accuracy."""
    errors = np.abs(gt_actions - pred_actions)

    metrics = {}

    # Overall accuracy at different thresholds
    for thresh in thresholds:
        within_thresh = (errors < thresh).mean() * 100
        metrics[f"accuracy_within_{thresh}"] = within_thresh

    # Per-step accuracy (all dims must be within threshold)
    for thresh in thresholds:
        step_success = (errors < thresh).all(axis=1).mean() * 100
        metrics[f"step_success_rate_{thresh}"] = step_success

    # Temporal analysis - first half vs second half
    mid = len(gt_actions) // 2
    early_mae = np.abs(gt_actions[:mid] - pred_actions[:mid]).mean()
    late_mae = np.abs(gt_actions[mid:] - pred_actions[mid:]).mean()
    metrics["early_mae"] = early_mae
    metrics["late_mae"] = late_mae
    metrics["error_drift"] = late_mae - early_mae

    # Per-dimension analysis
    metrics["max_error_per_dim"] = errors.max(axis=0).tolist()
    metrics["mean_error_per_dim"] = errors.mean(axis=0).tolist()

    # Separate analysis for base vs arms
    base_errors = errors[:, :2]  # base_vel
    left_arm_errors = errors[:, 2:9]  # left_arm
    right_arm_errors = errors[:, 9:16]  # right_arm

    metrics["base_mae"] = base_errors.mean()
    metrics["left_arm_mae"] = left_arm_errors.mean()
    metrics["right_arm_mae"] = right_arm_errors.mean()

    return metrics


def evaluate_trajectory(
    policy: Gr00tPolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    steps: int = 300,
    action_horizon: int = 16,
) -> tuple[np.ndarray, np.ndarray, float, float, dict]:
    """Evaluate model on a single trajectory."""
    traj = loader[traj_id]
    traj_length = len(traj)
    actual_steps = min(steps, traj_length)

    logger.info(f"Evaluating trajectory {traj_id}: {actual_steps} steps (traj length: {traj_length})")

    pred_actions = []
    action_keys = loader.modality_configs["action"].modality_keys

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")

    for step in range(0, actual_steps, action_horizon):
        data_point = extract_step_data(traj, step, modality_configs, embodiment_tag)

        obs = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)
        for lang_key in loader.modality_configs["language"].modality_keys:
            obs[lang_key] = data_point.text

        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
        action_chunk, _ = policy.get_action(parsed_obs)
        action_chunk = parse_action_gr00t(action_chunk)

        for j in range(action_horizon):
            concat_action = np.concatenate([
                np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j])
                for key in action_keys
            ], axis=0)
            pred_actions.append(concat_action)

    # Extract ground truth
    def extract_joints(traj, columns):
        np_dict = {}
        for col in columns:
            np_dict[col] = np.vstack([arr for arr in traj[col]])
        return np.concatenate([np_dict[col] for col in columns], axis=-1)

    gt_actions = extract_joints(traj, [f"action.{key}" for key in action_keys])[:actual_steps]
    pred_actions = np.array(pred_actions)[:actual_steps]

    # Compute metrics
    mse = np.mean((gt_actions - pred_actions) ** 2)
    mae = np.mean(np.abs(gt_actions - pred_actions))

    # Compute success metrics
    success_metrics = compute_success_metrics(gt_actions, pred_actions)

    return gt_actions, pred_actions, mse, mae, success_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate plug_stacking-finetuned GR00T model")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint (default: auto-detect latest)")
    parser.add_argument("--dataset_path", type=str, default="./data/plug_stacking_test",
                        help="Path to test dataset")
    parser.add_argument("--output_dir", type=str, default="./eval_results/plug_stacking",
                        help="Directory to save evaluation results")
    parser.add_argument("--traj_ids", type=int, nargs="+", default=None,
                        help="Trajectory IDs to evaluate (default: all)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Maximum steps per trajectory")
    parser.add_argument("--action_horizon", type=int, default=16,
                        help="Action horizon for inference")
    parser.add_argument("--generate_videos", action="store_true",
                        help="Generate video visualizations")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    # Find checkpoint
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint("./outputs/plug_stacking")
        if checkpoint_path is None:
            logger.error("No checkpoint found in ./outputs/plug_stacking/. Please specify --checkpoint_path")
            sys.exit(1)
        logger.info(f"Using latest checkpoint: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Check dataset
    if not Path(args.dataset_path).exists():
        logger.error(f"Dataset not found: {args.dataset_path}")
        logger.error("Please run: python finetuning/convert_plug_stacking.py")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy
    logger.info(f"Loading model from {checkpoint_path}...")
    device = args.device if torch.cuda.is_available() else "cpu"

    # Use NEW_EMBODIMENT tag for custom robot (Trossen AI Mobile)
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=checkpoint_path,
        device=device,
    )

    # Get modality config
    modality_config = policy.get_modality_config()
    logger.info(f"Modality config keys: video={modality_config['video'].modality_keys}, "
                f"state={modality_config['state'].modality_keys}, "
                f"action={modality_config['action'].modality_keys}")

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}...")
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
    )

    logger.info(f"Dataset has {len(dataset)} episodes")

    # Determine trajectories to evaluate
    traj_ids = args.traj_ids
    if traj_ids is None:
        traj_ids = list(range(len(dataset)))

    # Evaluate trajectories
    all_mse = []
    all_mae = []
    all_success_metrics = []

    for traj_id in traj_ids:
        if traj_id >= len(dataset):
            logger.warning(f"Trajectory {traj_id} out of range, skipping")
            continue

        gt_actions, pred_actions, mse, mae, success_metrics = evaluate_trajectory(
            policy=policy,
            loader=dataset,
            traj_id=traj_id,
            embodiment_tag=embodiment_tag,
            steps=args.steps,
            action_horizon=args.action_horizon,
        )

        all_mse.append(mse)
        all_mae.append(mae)
        all_success_metrics.append(success_metrics)

        logger.info(f"Trajectory {traj_id}: MSE={mse:.6f}, MAE={mae:.6f}")
        logger.info(f"  Step Success Rate (0.1 thresh): {success_metrics['step_success_rate_0.1']:.1f}%")
        logger.info(f"  Base MAE: {success_metrics['base_mae']:.4f}, "
                    f"Left Arm MAE: {success_metrics['left_arm_mae']:.4f}, "
                    f"Right Arm MAE: {success_metrics['right_arm_mae']:.4f}")

        # Generate plot
        plot_path = output_dir / f"trajectory_{traj_id}_comparison.png"
        plot_trajectory_comparison(
            gt_actions=gt_actions,
            pred_actions=pred_actions,
            traj_id=traj_id,
            action_horizon=args.action_horizon,
            save_path=str(plot_path),
        )

        # Generate multi-camera snapshot
        camera_plot_path = output_dir / f"trajectory_{traj_id}_cameras.png"
        plot_multi_camera_comparison(
            dataset_path=args.dataset_path,
            traj_id=traj_id,
            step_idx=len(gt_actions) // 2,  # Middle of trajectory
            save_path=str(camera_plot_path),
        )

        # Generate video if requested
        if args.generate_videos:
            video_path = output_dir / f"trajectory_{traj_id}_visualization.mp4"
            create_video_visualization(
                dataset_path=args.dataset_path,
                traj_id=traj_id,
                gt_actions=gt_actions,
                pred_actions=pred_actions,
                save_path=str(video_path),
            )

    # Summary statistics
    if all_mse:
        avg_mse = np.mean(all_mse)
        avg_mae = np.mean(all_mae)

        # Aggregate success metrics
        avg_success_005 = np.mean([m['step_success_rate_0.05'] for m in all_success_metrics])
        avg_success_01 = np.mean([m['step_success_rate_0.1'] for m in all_success_metrics])
        avg_success_02 = np.mean([m['step_success_rate_0.2'] for m in all_success_metrics])
        avg_accuracy_01 = np.mean([m['accuracy_within_0.1'] for m in all_success_metrics])
        avg_accuracy_02 = np.mean([m['accuracy_within_0.2'] for m in all_success_metrics])
        avg_error_drift = np.mean([m['error_drift'] for m in all_success_metrics])
        avg_base_mae = np.mean([m['base_mae'] for m in all_success_metrics])
        avg_left_arm_mae = np.mean([m['left_arm_mae'] for m in all_success_metrics])
        avg_right_arm_mae = np.mean([m['right_arm_mae'] for m in all_success_metrics])

        logger.info("=" * 60)
        logger.info(f"EVALUATION SUMMARY ({len(all_mse)} trajectories)")
        logger.info("=" * 60)
        logger.info(f"  Average MSE: {avg_mse:.6f}")
        logger.info(f"  Average MAE: {avg_mae:.6f}")
        logger.info("")
        logger.info("  Component-wise MAE:")
        logger.info(f"    Base Velocity: {avg_base_mae:.4f}")
        logger.info(f"    Left Arm: {avg_left_arm_mae:.4f}")
        logger.info(f"    Right Arm: {avg_right_arm_mae:.4f}")
        logger.info("")
        logger.info("  Success Rates (all dims within threshold):")
        logger.info(f"    @ 0.05 threshold: {avg_success_005:.1f}%")
        logger.info(f"    @ 0.10 threshold: {avg_success_01:.1f}%")
        logger.info(f"    @ 0.20 threshold: {avg_success_02:.1f}%")
        logger.info("")
        logger.info("  Action Accuracy (% of predictions within threshold):")
        logger.info(f"    @ 0.10 threshold: {avg_accuracy_01:.1f}%")
        logger.info(f"    @ 0.20 threshold: {avg_accuracy_02:.1f}%")
        logger.info("")
        logger.info(f"  Error Drift (late - early MAE): {avg_error_drift:+.6f}")
        logger.info(f"  Results saved to: {output_dir}")
        logger.info("=" * 60)

        # Save summary
        summary_path = output_dir / "evaluation_summary.txt"
        with open(summary_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("Plug Stacking GR00T Evaluation Results\n")
            f.write("Trossen AI Mobile - Plug Stacking Task\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Dataset: {args.dataset_path}\n")
            f.write(f"Trajectories evaluated: {traj_ids}\n")
            f.write(f"Steps per trajectory: {args.steps}\n")
            f.write(f"Action horizon: {args.action_horizon}\n\n")
            f.write("Per-Trajectory Results:\n")
            for traj_id, mse, mae, sm in zip(traj_ids, all_mse, all_mae, all_success_metrics):
                f.write(f"  Trajectory {traj_id}: MSE={mse:.6f}, MAE={mae:.6f}, "
                        f"Success@0.1={sm['step_success_rate_0.1']:.1f}%, "
                        f"Success@0.2={sm['step_success_rate_0.2']:.1f}%\n")
                f.write(f"    Base MAE: {sm['base_mae']:.4f}, "
                        f"Left Arm MAE: {sm['left_arm_mae']:.4f}, "
                        f"Right Arm MAE: {sm['right_arm_mae']:.4f}\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"AGGREGATE METRICS ({len(all_mse)} trajectories)\n")
            f.write(f"{'='*60}\n")
            f.write(f"Average MSE: {avg_mse:.6f}\n")
            f.write(f"Average MAE: {avg_mae:.6f}\n\n")
            f.write("Component-wise MAE:\n")
            f.write(f"  Base Velocity: {avg_base_mae:.4f}\n")
            f.write(f"  Left Arm: {avg_left_arm_mae:.4f}\n")
            f.write(f"  Right Arm: {avg_right_arm_mae:.4f}\n\n")
            f.write("Success Rates (all dims within threshold per step):\n")
            f.write(f"  @ 0.05 threshold: {avg_success_005:.1f}%\n")
            f.write(f"  @ 0.10 threshold: {avg_success_01:.1f}%\n")
            f.write(f"  @ 0.20 threshold: {avg_success_02:.1f}%\n\n")
            f.write("Action Accuracy (% of individual predictions within threshold):\n")
            f.write(f"  @ 0.10 threshold: {avg_accuracy_01:.1f}%\n")
            f.write(f"  @ 0.20 threshold: {avg_accuracy_02:.1f}%\n\n")
            f.write(f"Error Drift (late - early MAE): {avg_error_drift:+.6f}\n")
            f.write(f"  (positive = errors increase over trajectory)\n")

        logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
