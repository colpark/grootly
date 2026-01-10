#!/usr/bin/env python3
"""
Run ALOHA simulation with finetuned GR00T model.

This script:
1. Loads the finetuned checkpoint
2. Runs rollouts in gym-aloha simulation
3. Measures actual task success rate
4. Generates rollout videos

Usage:
    # First install gym-aloha
    pip install gym-aloha

    # Then run simulation
    python finetuning/run_aloha_sim.py \
        --checkpoint_path ./outputs/checkpoint-1000 \
        --n_episodes 50 \
        --save_videos
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import to register GR00T model
import gr00t.model.gr00t_n1d6.gr00t_n1d6  # noqa: F401

# Import modality config
from finetuning.gr1_modality_config import GR1_CONFIG  # noqa: F401

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir: str = "./outputs") -> Optional[str]:
    """Find the latest checkpoint in output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    checkpoints = list(output_path.glob("checkpoint-*"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
    return str(checkpoints[-1])


def parse_observation_for_policy(obs: dict, modality_config: dict) -> dict:
    """Convert environment observation to policy input format."""
    parsed = {"video": {}, "state": {}, "language": {}}

    # Video
    for key in modality_config["video"].modality_keys:
        obs_key = f"video.{key}"
        if obs_key in obs:
            # Add batch and time dimensions: (H, W, C) -> (1, 1, H, W, C)
            parsed["video"][key] = obs[obs_key][None, None, ...]

    # State
    for key in modality_config["state"].modality_keys:
        obs_key = f"state.{key}"
        if obs_key in obs:
            # Add batch and time dimensions: (D,) -> (1, 1, D)
            parsed["state"][key] = obs[obs_key][None, None, ...]

    # Language
    for key in modality_config["language"].modality_keys:
        if key in obs:
            parsed["language"][key] = [[obs[key]]]

    return parsed


def parse_action_from_policy(action: dict, action_keys: list) -> dict:
    """Convert policy output to environment action format."""
    env_action = {}
    for key in action_keys:
        action_key = f"action.{key}"
        if key in action:
            # Remove batch dimension and take first action in horizon
            env_action[action_key] = action[key][0][0]
    return env_action


class VideoRecorder:
    """Simple video recorder for rollouts."""

    def __init__(self, save_path: str, fps: int = 20):
        self.save_path = save_path
        self.fps = fps
        self.frames = []

    def add_frame(self, frame: np.ndarray):
        self.frames.append(frame.copy())

    def save(self):
        if not self.frames:
            return

        import cv2
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)

        height, width = self.frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.save_path, fourcc, self.fps, (width, height))

        for frame in self.frames:
            # Convert RGB to BGR for OpenCV
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()
        logger.info(f"Saved video to {self.save_path}")


def run_episode(
    env,
    policy: Gr00tPolicy,
    modality_config: dict,
    max_steps: int = 400,
    action_horizon: int = 8,
    video_recorder: Optional[VideoRecorder] = None,
) -> tuple[bool, int, float]:
    """
    Run a single episode.

    Returns:
        success: Whether task was completed successfully
        steps: Number of steps taken
        total_reward: Cumulative reward
    """
    obs, info = env.reset()
    total_reward = 0.0
    success = False

    action_keys = modality_config["action"].modality_keys

    for step in range(0, max_steps, action_horizon):
        # Record frame
        if video_recorder is not None:
            frame = obs.get("video.ego_view", env.render())
            video_recorder.add_frame(frame)

        # Get action from policy
        parsed_obs = parse_observation_for_policy(obs, modality_config)
        action_chunk, _ = policy.get_action(parsed_obs)

        # Execute action horizon
        for action_idx in range(action_horizon):
            if step + action_idx >= max_steps:
                break

            # Extract single action from chunk
            single_action = {}
            for key in action_keys:
                if key in action_chunk:
                    single_action[f"action.{key}"] = action_chunk[key][0][action_idx]

            obs, reward, terminated, truncated, info = env.step(single_action)
            total_reward += reward

            if info.get("success", False):
                success = True
                logger.info(f"  Success at step {step + action_idx}!")
                return success, step + action_idx, total_reward

            if terminated or truncated:
                return success, step + action_idx, total_reward

    return success, max_steps, total_reward


def main():
    parser = argparse.ArgumentParser(description="Run ALOHA simulation with finetuned GR00T")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint (default: auto-detect latest)")
    parser.add_argument("--task", type=str, default="AlohaTransferCube-v0",
                        choices=["AlohaTransferCube-v0", "AlohaInsertion-v0"],
                        help="ALOHA task to run")
    parser.add_argument("--n_episodes", type=int, default=50,
                        help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=400,
                        help="Maximum steps per episode")
    parser.add_argument("--action_horizon", type=int, default=8,
                        help="Number of actions to execute per policy call")
    parser.add_argument("--save_videos", action="store_true",
                        help="Save rollout videos")
    parser.add_argument("--video_dir", type=str, default="./sim_rollouts",
                        help="Directory to save videos")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Find checkpoint
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            logger.error("No checkpoint found. Please specify --checkpoint_path")
            sys.exit(1)
        logger.info(f"Using latest checkpoint: {checkpoint_path}")

    # Check gym-aloha installation
    try:
        import gym_aloha  # noqa: F401
    except ImportError:
        logger.error("gym-aloha not installed. Install with: pip install gym-aloha")
        sys.exit(1)

    # Import ALOHA env wrapper
    from gr00t.eval.sim.ALOHA.aloha_env import AlohaEnv

    # Load policy
    logger.info(f"Loading model from {checkpoint_path}...")
    device = args.device if torch.cuda.is_available() else "cpu"

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.GR1,
        model_path=checkpoint_path,
        device=device,
    )
    modality_config = policy.get_modality_config()

    # Create environment
    logger.info(f"Creating ALOHA environment: {args.task}")
    env = AlohaEnv(task=args.task)

    # Run episodes
    successes = []
    rewards = []
    steps_list = []

    logger.info(f"Running {args.n_episodes} episodes...")
    for ep in range(args.n_episodes):
        # Setup video recorder
        video_recorder = None
        if args.save_videos:
            video_path = Path(args.video_dir) / f"episode_{ep:03d}.mp4"
            video_recorder = VideoRecorder(str(video_path))

        # Run episode
        success, steps, total_reward = run_episode(
            env=env,
            policy=policy,
            modality_config=modality_config,
            max_steps=args.max_steps,
            action_horizon=args.action_horizon,
            video_recorder=video_recorder,
        )

        successes.append(success)
        rewards.append(total_reward)
        steps_list.append(steps)

        # Save video
        if video_recorder is not None:
            video_recorder.save()

        status = "SUCCESS" if success else "FAIL"
        logger.info(f"Episode {ep+1}/{args.n_episodes}: {status}, steps={steps}, reward={total_reward:.2f}")

    # Summary
    success_rate = np.mean(successes) * 100
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps_list)

    logger.info("=" * 60)
    logger.info(f"SIMULATION RESULTS ({args.n_episodes} episodes)")
    logger.info("=" * 60)
    logger.info(f"  Success Rate: {success_rate:.1f}%")
    logger.info(f"  Average Reward: {avg_reward:.2f}")
    logger.info(f"  Average Steps: {avg_steps:.1f}")
    logger.info(f"  Successful Episodes: {sum(successes)}/{args.n_episodes}")
    logger.info("=" * 60)

    # Save summary
    summary_path = Path(args.video_dir) / "simulation_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Episodes: {args.n_episodes}\n")
        f.write(f"Max Steps: {args.max_steps}\n")
        f.write(f"Action Horizon: {args.action_horizon}\n\n")
        f.write(f"SUCCESS RATE: {success_rate:.1f}%\n")
        f.write(f"Average Reward: {avg_reward:.2f}\n")
        f.write(f"Average Steps: {avg_steps:.1f}\n\n")
        f.write("Per-Episode Results:\n")
        for ep, (s, r, st) in enumerate(zip(successes, rewards, steps_list)):
            status = "SUCCESS" if s else "FAIL"
            f.write(f"  Episode {ep}: {status}, reward={r:.2f}, steps={st}\n")

    logger.info(f"Summary saved to {summary_path}")

    env.close()


if __name__ == "__main__":
    main()
