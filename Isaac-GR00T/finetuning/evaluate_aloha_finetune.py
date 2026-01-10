#!/usr/bin/env python3
"""
Evaluation script for ALOHA-finetuned GR00T model.

This script:
1. Loads the finetuned checkpoint
2. Runs open-loop evaluation on the ALOHA dataset
3. Generates visualization plots comparing predicted vs ground truth actions
4. Optionally generates video visualizations

Usage:
    python finetuning/evaluate_aloha_finetune.py \
        --checkpoint_path ./outputs/checkpoint-1000 \
        --dataset_path ./data/aloha_groot_format \
        --output_dir ./eval_results

If checkpoint_path is not provided, it will search for the latest checkpoint in ./outputs/
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import to register GR00T model with transformers
import gr00t.model.gr00t_n1d6.gr00t_n1d6  # noqa: F401

# Import modality config registration
from finetuning.gr1_modality_config import GR1_CONFIG  # noqa: F401

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint in the output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = list(output_path.glob("checkpoint-*"))
    if not checkpoints:
        return None

    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
    return str(checkpoints[-1])


def plot_trajectory_comparison(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    state_joints: np.ndarray,
    traj_id: int,
    action_keys: list[str],
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

    fig.suptitle(f"Trajectory {traj_id} - Predicted vs Ground Truth Actions", fontsize=16)

    # Get action dimension labels
    dim_labels = []
    dim_offset = 0
    for key in action_keys:
        if key == "left_arm":
            for i in range(7):
                dim_labels.append(f"left_arm_{i}")
            dim_offset += 7
        elif key == "left_hand":
            for i in range(6):
                dim_labels.append(f"left_hand_{i}")
            dim_offset += 6
        elif key == "right_arm":
            for i in range(7):
                dim_labels.append(f"right_arm_{i}")
            dim_offset += 7
        elif key == "right_hand":
            for i in range(6):
                dim_labels.append(f"right_hand_{i}")
            dim_offset += 6
        elif key == "waist":
            for i in range(3):
                dim_labels.append(f"waist_{i}")
            dim_offset += 3
        else:
            dim_labels.append(key)
            dim_offset += 1

    for dim_idx in range(min(action_dim, len(axes))):
        ax = axes[dim_idx]

        # Plot state if same dimensionality
        if state_joints.shape == gt_actions.shape:
            ax.plot(state_joints[:, dim_idx], 'g-', alpha=0.5, label='State')

        ax.plot(gt_actions[:, dim_idx], 'b-', label='Ground Truth', linewidth=2)
        ax.plot(pred_actions[:, dim_idx], 'r--', label='Predicted', linewidth=2)

        # Mark inference points
        for j in range(0, actual_steps, action_horizon):
            ax.axvline(x=j, color='gray', linestyle=':', alpha=0.3)

        label = dim_labels[dim_idx] if dim_idx < len(dim_labels) else f"Dim {dim_idx}"
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


def create_video_visualization(
    dataset_path: str,
    traj_id: int,
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    save_path: str,
    fps: int = 20,
) -> None:
    """Create a video showing the episode frames with action overlay."""
    try:
        import cv2
        import imageio.v3 as iio
    except ImportError:
        logger.warning("imageio or cv2 not available, skipping video generation")
        return

    # Load video frames - try multiple path patterns
    video_paths = [
        Path(dataset_path) / "videos" / "chunk-000" / "observation.images.ego_view" / f"episode_{traj_id:06d}.mp4",
        Path(dataset_path) / "videos" / f"ego_view_episode_{traj_id:06d}.mp4",
        Path(dataset_path) / "videos" / f"episode_{traj_id:06d}.mp4",
    ]

    video_path = None
    for vp in video_paths:
        if vp.exists():
            video_path = vp
            break

    if video_path is None:
        logger.warning(f"Video not found. Tried: {video_paths[0]}")
        return

    try:
        frames = iio.imread(str(video_path))
    except Exception as e:
        logger.warning(f"Could not read video: {e}")
        return

    # Create output video
    height, width = frames.shape[1:3]

    # Add space for action visualization
    canvas_height = height + 200
    canvas_width = max(width, 400)

    output_frames = []

    for i, frame in enumerate(frames):
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Place video frame
        canvas[:height, :width] = frame

        # Add text overlay
        cv2.putText(canvas, f"Frame {i}/{len(frames)}", (10, height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if i < len(gt_actions) and i < len(pred_actions):
            # Show action comparison for first few dimensions
            y_offset = height + 60
            for dim_idx in range(min(4, gt_actions.shape[1])):
                gt_val = gt_actions[i, dim_idx]
                pred_val = pred_actions[i, dim_idx]
                error = abs(gt_val - pred_val)

                text = f"Dim{dim_idx}: GT={gt_val:.3f} Pred={pred_val:.3f} Err={error:.3f}"
                color = (0, 255, 0) if error < 0.1 else (0, 165, 255) if error < 0.2 else (0, 0, 255)
                cv2.putText(canvas, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 25

        output_frames.append(canvas)

    # Write output video
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(save_path), fourcc, fps, (canvas_width, canvas_height))

    for frame in output_frames:
        # Convert RGB to BGR for OpenCV
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
    """
    Compute proxy success metrics based on action prediction accuracy.

    Returns:
        dict with:
        - accuracy_at_threshold: % of predictions within threshold of ground truth
        - per_dim_accuracy: accuracy broken down by action dimension
        - temporal_accuracy: accuracy over time (early vs late in trajectory)
    """
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
    metrics["error_drift"] = late_mae - early_mae  # positive = getting worse

    # Per-dimension max error
    metrics["max_error_per_dim"] = errors.max(axis=0).tolist()
    metrics["mean_error_per_dim"] = errors.mean(axis=0).tolist()

    return metrics


def evaluate_trajectory(
    policy: Gr00tPolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    steps: int = 300,
    action_horizon: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, dict]:
    """Evaluate model on a single trajectory. Returns gt_actions, pred_actions, state_joints, mse, mae, success_metrics."""
    traj = loader[traj_id]
    traj_length = len(traj)
    actual_steps = min(steps, traj_length)

    logger.info(f"Evaluating trajectory {traj_id}: {actual_steps} steps (traj length: {traj_length})")

    pred_actions = []
    state_keys = loader.modality_configs["state"].modality_keys
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

    state_joints = extract_joints(traj, [f"state.{key}" for key in state_keys])
    gt_actions = extract_joints(traj, [f"action.{key}" for key in action_keys])[:actual_steps]
    pred_actions = np.array(pred_actions)[:actual_steps]

    # Compute metrics
    mse = np.mean((gt_actions - pred_actions) ** 2)
    mae = np.mean(np.abs(gt_actions - pred_actions))

    # Compute success metrics
    success_metrics = compute_success_metrics(gt_actions, pred_actions)

    return gt_actions, pred_actions, state_joints, mse, mae, success_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ALOHA-finetuned GR00T model")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint (default: auto-detect latest)")
    parser.add_argument("--dataset_path", type=str, default="./data/aloha_groot_format",
                        help="Path to converted ALOHA dataset")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--traj_ids", type=int, nargs="+", default=[0, 1, 2],
                        help="Trajectory IDs to evaluate")
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
        checkpoint_path = find_latest_checkpoint("./outputs")
        if checkpoint_path is None:
            logger.error("No checkpoint found in ./outputs/. Please specify --checkpoint_path")
            sys.exit(1)
        logger.info(f"Using latest checkpoint: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Check dataset
    if not Path(args.dataset_path).exists():
        logger.error(f"Dataset not found: {args.dataset_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy
    logger.info(f"Loading model from {checkpoint_path}...")
    device = args.device if torch.cuda.is_available() else "cpu"

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.GR1,
        model_path=checkpoint_path,
        device=device,
    )

    # Get modality config
    modality_config = policy.get_modality_config()
    logger.info(f"Modality config: {modality_config}")

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}...")
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend="decord",  # Use decord for mp4v compatibility
        video_backend_kwargs=None,
    )

    logger.info(f"Dataset has {len(dataset)} episodes")

    # Evaluate trajectories
    all_mse = []
    all_mae = []
    all_success_metrics = []
    action_keys = modality_config["action"].modality_keys

    for traj_id in args.traj_ids:
        if traj_id >= len(dataset):
            logger.warning(f"Trajectory {traj_id} out of range, skipping")
            continue

        gt_actions, pred_actions, state_joints, mse, mae, success_metrics = evaluate_trajectory(
            policy=policy,
            loader=dataset,
            traj_id=traj_id,
            embodiment_tag=EmbodimentTag.GR1,
            steps=args.steps,
            action_horizon=args.action_horizon,
        )

        all_mse.append(mse)
        all_mae.append(mae)
        all_success_metrics.append(success_metrics)

        logger.info(f"Trajectory {traj_id}: MSE={mse:.6f}, MAE={mae:.6f}")
        logger.info(f"  Step Success Rate (0.1 thresh): {success_metrics['step_success_rate_0.1']:.1f}%")
        logger.info(f"  Step Success Rate (0.2 thresh): {success_metrics['step_success_rate_0.2']:.1f}%")

        # Generate plot
        plot_path = output_dir / f"trajectory_{traj_id}_comparison.png"
        plot_trajectory_comparison(
            gt_actions=gt_actions,
            pred_actions=pred_actions,
            state_joints=state_joints,
            traj_id=traj_id,
            action_keys=action_keys,
            action_horizon=args.action_horizon,
            save_path=str(plot_path),
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

        logger.info("=" * 60)
        logger.info(f"Evaluation Summary ({len(all_mse)} trajectories)")
        logger.info(f"  Average MSE: {avg_mse:.6f}")
        logger.info(f"  Average MAE: {avg_mae:.6f}")
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
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Dataset: {args.dataset_path}\n")
            f.write(f"Trajectories evaluated: {args.traj_ids}\n")
            f.write(f"Steps per trajectory: {args.steps}\n")
            f.write(f"Action horizon: {args.action_horizon}\n\n")
            f.write("Per-Trajectory Results:\n")
            for i, (traj_id, mse, mae, sm) in enumerate(zip(args.traj_ids, all_mse, all_mae, all_success_metrics)):
                f.write(f"  Trajectory {traj_id}: MSE={mse:.6f}, MAE={mae:.6f}, "
                        f"Success@0.1={sm['step_success_rate_0.1']:.1f}%, "
                        f"Success@0.2={sm['step_success_rate_0.2']:.1f}%\n")
            f.write(f"\n{'='*50}\n")
            f.write(f"AGGREGATE METRICS ({len(all_mse)} trajectories)\n")
            f.write(f"{'='*50}\n")
            f.write(f"Average MSE: {avg_mse:.6f}\n")
            f.write(f"Average MAE: {avg_mae:.6f}\n\n")
            f.write(f"Success Rates (all dims within threshold per step):\n")
            f.write(f"  @ 0.05 threshold: {avg_success_005:.1f}%\n")
            f.write(f"  @ 0.10 threshold: {avg_success_01:.1f}%\n")
            f.write(f"  @ 0.20 threshold: {avg_success_02:.1f}%\n\n")
            f.write(f"Action Accuracy (% of individual predictions within threshold):\n")
            f.write(f"  @ 0.10 threshold: {avg_accuracy_01:.1f}%\n")
            f.write(f"  @ 0.20 threshold: {avg_accuracy_02:.1f}%\n\n")
            f.write(f"Error Drift (late - early MAE): {avg_error_drift:+.6f}\n")
            f.write(f"  (positive = errors increase over trajectory)\n")

        logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
