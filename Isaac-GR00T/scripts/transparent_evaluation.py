#!/usr/bin/env python3
"""
Transparent RoboCasa Evaluation

Uses the same evaluation setup as rollout_policy.py (which achieves 100% success)
but adds transparency logging to show what's happening.

Usage:
    # Terminal 1 - Start policy server
    uv run python gr00t/eval/run_gr00t_server.py \
        --model-path nvidia/GR00T-N1.6-3B \
        --embodiment-tag ROBOCASA_PANDA_OMRON \
        --use-sim-policy-wrapper

    # Terminal 2 - Run transparent evaluation
    gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python scripts/transparent_evaluation.py \
        --save-dir ./transparent_eval \
        --max-steps 100 \
        --camera-view side

Camera view options:
    --camera-view side   : Single high-res side view (512px) - recommended, clearest view
    --camera-view wrist  : Wrist camera view (512px) - good for close-up manipulation
    --camera-view both   : Side + wrist views side by side
    --camera-view all    : All 6 cameras (original wide concatenation)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the working rollout_policy.py
from gr00t.eval.rollout_policy import (
    get_robocasa_env_fn,
    create_gr00t_sim_policy,
    WrapperConfigs,
    VideoConfig,
    MultiStepConfig,
)
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.eval.sim.wrapper.video_recording_wrapper import VideoRecorder, VideoRecordingWrapper
from gr00t.data.embodiment_tags import EmbodimentTag
import cv2


class SelectiveCameraVideoWrapper(VideoRecordingWrapper):
    """Video wrapper that records only selected camera views for cleaner output."""

    def __init__(self, env, video_recorder, camera_keys=None, **kwargs):
        """
        Args:
            camera_keys: List of camera keys to include (e.g., ['video.side_0.res512']).
                        If None, uses all video keys (default behavior).
                        Options typically include:
                        - 'video.side_0.res256_freq20' - Side view 256px
                        - 'video.side_0.res512_freq20' - Side view 512px (recommended)
                        - 'video.side_1.res256_freq20' - Other side view 256px
                        - 'video.side_1.res512_freq20' - Other side view 512px
                        - 'video.wrist_0.res256_freq20' - Wrist camera 256px
                        - 'video.wrist_0.res512_freq20' - Wrist camera 512px
        """
        super().__init__(env, video_recorder, **kwargs)
        self.camera_keys = camera_keys

    def step(self, action):
        result = super(VideoRecordingWrapper, self).step(action)  # Call grandparent's step
        self.step_count += 1

        if self.file_path is not None and ((self.step_count % self.steps_per_render) == 0):
            if not self.video_recorder.is_ready():
                self.video_recorder.start(self.file_path)

            obs = result[0]
            video_frames = []

            if self.camera_keys:
                # Use only selected camera keys
                for key in self.camera_keys:
                    if key in obs:
                        video_frames.append(obs[key])
                    else:
                        # Try partial match
                        for k, v in obs.items():
                            if key in k and "video" in k:
                                video_frames.append(v)
                                break
            else:
                # Default: use all video keys
                for k, v in obs.items():
                    if "video" in k:
                        video_frames.append(v)

            if len(video_frames) == 0:
                # Fallback: use any video key
                for k, v in obs.items():
                    if "video" in k:
                        video_frames.append(v)
                        break

            assert len(video_frames) > 0, "No video frame found in the observation"

            # Resize frames to common height for horizontal concatenation
            if len(video_frames) > 1:
                video_frames = self._resize_frames_to_common_height(video_frames)

            # Concatenate all video frames horizontally
            if len(video_frames) == 1:
                frame = video_frames[0]
            else:
                frame = np.concatenate(video_frames, axis=1)

            assert frame.dtype == np.uint8

            if self.overlay_text:
                auto_language_key = [
                    k for k in result[0].keys()
                    if k.startswith("annotation.") or k.startswith("language.")
                ][0]
                language = result[0][auto_language_key]
                language = language + " (" + str(int(result[-1]["success"])) + ")"

                # Dynamic font scaling
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_thickness = 2
                font_color = (255, 255, 255)
                padding = 5
                target_width = frame.shape[1] - 2 * padding
                font_scale = 1.0

                text_size = cv2.getTextSize(language, font, font_scale, font_thickness)[0]
                if text_size[0] > target_width:
                    while text_size[0] > target_width and font_scale > 0.1:
                        font_scale *= 0.9
                        text_size = cv2.getTextSize(language, font, font_scale, font_thickness)[0]
                else:
                    while text_size[0] < target_width and font_scale < 2.0:
                        font_scale *= 1.1
                        text_size = cv2.getTextSize(language, font, font_scale, font_thickness)[0]
                    font_scale *= 0.9

                text_x = padding
                text_y = frame.shape[0] - 20

                cv2.rectangle(
                    frame,
                    (text_x - padding, text_y - text_size[1] - padding),
                    (text_x + text_size[0] + padding, text_y + padding),
                    (0, 0, 0), -1,
                )
                cv2.putText(
                    frame, language, (text_x, text_y), font, font_scale, font_color, font_thickness
                )

            self.video_recorder.write_frame(frame)

        info = result[-1]
        self.is_success |= info["success"]

        # Update intermediate signals (copied from parent)
        if "intermediate_signals" in info:
            for key, value in info["intermediate_signals"].items():
                if key in ["grasp_obj", "grasp_distractor_obj", "contact_obj", "contact_distractor_obj"]:
                    initial_value = False
                elif key in ["gripper_obj_dist", "gripper_distractor_dist"]:
                    initial_value = 1e9
                elif key.startswith("_"):
                    continue
                else:
                    continue

                if key not in self.intermediate_signals:
                    self.intermediate_signals[key] = initial_value

                if key in ["grasp_obj", "grasp_distractor_obj", "contact_obj", "contact_distractor_obj"]:
                    self.intermediate_signals[key] |= value
                elif key in ["gripper_obj_dist", "gripper_distractor_dist"]:
                    self.intermediate_signals[key] = min(self.intermediate_signals[key], value)

        return result


class TransparentEvaluator:
    """Run evaluation with full transparency, using the same setup as rollout_policy.py."""

    def __init__(
        self,
        env_name: str = "robocasa_panda_omron/CloseDrawer_PandaOmron_Env",
        policy_host: str = "127.0.0.1",
        policy_port: int = 5555,
        save_dir: str = "./transparent_eval",
        n_action_steps: int = 8,
        max_episode_steps: int = 504,
        save_video: bool = True,
        camera_view: str = "side",  # "side", "wrist", "both", or "all"
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_action_steps = n_action_steps
        self.max_episode_steps = max_episode_steps
        self.save_video = save_video

        print("\n" + "="*70)
        print("TRANSPARENT ROBOCASA EVALUATION")
        print("(Using same setup as rollout_policy.py)")
        print("="*70)

        # Create environment using the SAME function as rollout_policy.py
        print(f"\nCreating environment: {env_name}")
        env_fn = get_robocasa_env_fn(env_name)
        self.base_env = env_fn()

        # Wrap with VideoRecordingWrapper if saving video
        env_to_wrap = self.base_env
        if save_video:
            video_recorder = VideoRecorder.create_h264(
                fps=20,
                codec="h264",
                input_pix_fmt="rgb24",
                crf=22,
            )

            # Select camera keys based on view preference
            if camera_view == "side":
                # Single high-res side view - best for seeing the task
                camera_keys = ["res512_freq20", "side_0"]
                print(f"📹 Camera: Side view (512px)")
            elif camera_view == "wrist":
                # Wrist camera - good for close-up manipulation
                camera_keys = ["res512_freq20", "wrist"]
                print(f"📹 Camera: Wrist view (512px)")
            elif camera_view == "both":
                # Side + wrist for comprehensive view
                camera_keys = ["side_0.res512", "wrist_0.res512"]
                print(f"📹 Camera: Side + Wrist views (512px)")
            else:  # "all"
                # All cameras (original behavior)
                camera_keys = None
                print(f"📹 Camera: All views (wide concatenation)")

            env_to_wrap = SelectiveCameraVideoWrapper(
                self.base_env,
                video_recorder,
                camera_keys=camera_keys,
                video_dir=self.save_dir,
                steps_per_render=2,
                max_episode_steps=max_episode_steps,
                overlay_text=True,
            )
            print(f"Video recording enabled: {self.save_dir}")

        # Wrap with MultiStepWrapper (same as rollout_policy.py)
        self.env = MultiStepWrapper(
            env_to_wrap,
            video_delta_indices=np.array([0]),
            state_delta_indices=np.array([0]),
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        )

        self.env_name = env_name
        self.task_name = env_name.split("/")[1].replace("_PandaOmron_Env", "")
        print(f"Task: {self.task_name}")

        # Create policy using the SAME function as rollout_policy.py
        print(f"\nConnecting to policy server at {policy_host}:{policy_port}")
        self.policy = create_gr00t_sim_policy(
            model_path="",
            embodiment_tag=EmbodimentTag.ROBOCASA_PANDA_OMRON,
            policy_client_host=policy_host,
            policy_client_port=policy_port,
        )
        print("Connected!")

        # Storage
        self.episode_log = []

    def explain_observation(self, obs: Dict) -> None:
        """Print what's in the observation."""
        print("\n📋 OBSERVATION STRUCTURE:")

        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                if "video" in key:
                    print(f"   📷 {key}: shape={value.shape}, range=[{value.min()}, {value.max()}]")
                elif "state" in key:
                    print(f"   🤖 {key}: shape={value.shape}")
            elif "annotation" in key or "task" in key:
                print(f"   📝 {key}: \"{value}\"")

    def explain_action(self, action: Dict) -> None:
        """Print what actions are being taken."""
        print("\n🎮 ACTION CHUNK (8 future actions):")

        for key, value in action.items():
            if isinstance(value, np.ndarray):
                if "position" in key:
                    print(f"   • {key}: shape={value.shape}")
                    print(f"     First action: {value[0, :3] if len(value.shape) > 1 else value[:3]}...")
                elif "gripper" in key:
                    if len(value.shape) > 1:
                        print(f"   • {key}: {value[:, 0].tolist()}")
                    else:
                        print(f"   • {key}: {value.tolist()}")

    def _add_batch_dim(self, obs: Dict) -> Dict:
        """Add batch dimension to observation for policy input."""
        batched = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                # Add batch dimension: (T, ...) -> (1, T, ...)
                batched[key] = value[np.newaxis, ...]
            else:
                # For strings (language), wrap in list
                batched[key] = [value]
        return batched

    def _remove_batch_dim(self, action: Dict) -> Dict:
        """Remove batch dimension from action for env input."""
        unbatched = {}
        for key, value in action.items():
            if isinstance(value, np.ndarray) and len(value.shape) > 0:
                # Remove batch dimension: (1, ...) -> (...)
                unbatched[key] = value[0]
            else:
                unbatched[key] = value
        return unbatched

    def run_episode(self, verbose: bool = True) -> Dict:
        """Run one episode with transparency logging."""

        print("\n" + "="*70)
        print("STARTING EPISODE")
        print("="*70)

        # Reset
        obs, info = self.env.reset()
        self.policy.reset()

        if verbose:
            self.explain_observation(obs)

        # Get task instruction
        task_key = [k for k in obs.keys() if "annotation" in k or "task" in k]
        if task_key:
            task = obs[task_key[0]]
            print(f"\n📋 TASK: \"{task}\"")

        print("\n" + "="*70)
        print("EXECUTION LOOP")
        print(f"(Policy returns {self.n_action_steps} actions per call)")
        print("="*70)

        step = 0
        total_reward = 0
        success = False
        step_logs = []

        while True:
            # Add batch dimension for policy (VectorEnv does this automatically)
            obs_batched = self._add_batch_dim(obs)

            # Get action from policy (returns action chunk with batch dim)
            action_batched, _ = self.policy.get_action(obs_batched)

            # Remove batch dimension for env
            action = self._remove_batch_dim(action_batched)

            if verbose and step % 8 == 0:  # Log every policy call
                print(f"\n{'─'*70}")
                print(f"POLICY CALL at step {step}")
                print(f"{'─'*70}")
                self.explain_action(action)

            # Execute action chunk (MultiStepWrapper handles the 8-action execution)
            obs, reward, done, truncated, info = self.env.step(action)

            # The wrapper executes n_action_steps internally
            step += self.n_action_steps
            total_reward += reward

            # Check success (handle list, array, or bool)
            success_val = info.get("success", False)
            if isinstance(success_val, (list, np.ndarray)):
                current_success = bool(np.any(success_val))
            else:
                current_success = bool(success_val)

            # Log
            step_log = {
                "step": step,
                "reward": float(reward),
                "total_reward": float(total_reward),
                "done": bool(done),
                "success": current_success,
            }
            step_logs.append(step_log)

            if current_success:
                success = True
                print(f"\n✅ SUCCESS at step {step}!")
                break

            if verbose and step % 24 == 0:  # Progress update
                print(f"\n📊 Step {step}: reward={reward:.4f}, total={total_reward:.4f}")

            if done or truncated:
                break

            if step >= self.max_episode_steps:
                break

        # Summary
        print("\n" + "="*70)
        print("EPISODE SUMMARY")
        print("="*70)
        print(f"""
📊 Results:
   • Total steps: {step}
   • Total reward: {total_reward:.4f}
   • Success: {'✅ YES' if success else '❌ NO'}

🎯 What happened:
   {'The robot successfully completed the task!' if success else 'The robot did not complete the task.'}
""")

        # Save logs
        logs_path = self.save_dir / "step_logs.json"
        with open(logs_path, "w") as f:
            json.dump(step_logs, f, indent=2)
        print(f"📝 Logs saved: {logs_path}")

        # Save report
        self._save_report(step_logs, success, total_reward, step)

        return {
            "success": success,
            "total_steps": step,
            "total_reward": total_reward,
        }

    def run_multiple_episodes(self, n_episodes: int = 10) -> Dict:
        """Run multiple episodes and report success rate."""

        print("\n" + "="*70)
        print(f"RUNNING {n_episodes} EPISODES")
        print("="*70)

        results = []
        for ep in range(n_episodes):
            print(f"\n{'─'*70}")
            print(f"EPISODE {ep + 1}/{n_episodes}")
            print(f"{'─'*70}")

            result = self.run_episode(verbose=(ep == 0))  # Verbose only for first episode
            results.append(result)

        # Summary
        successes = [r["success"] for r in results]
        success_rate = sum(successes) / len(successes) * 100

        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"""
📊 Summary over {n_episodes} episodes:
   • Success rate: {success_rate:.1f}%
   • Successes: {sum(successes)}/{n_episodes}
   • Average reward: {np.mean([r['total_reward'] for r in results]):.4f}
   • Average steps: {np.mean([r['total_steps'] for r in results]):.1f}
""")

        return {
            "success_rate": success_rate,
            "results": results,
        }

    def _save_report(self, step_logs: List[Dict], success: bool, total_reward: float, total_steps: int):
        """Save markdown report."""
        report = f"""# Transparent Evaluation Report

**Task**: {self.task_name}
**Environment**: {self.env_name}
**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Result

| Metric | Value |
|--------|-------|
| **Success** | {'✅ YES' if success else '❌ NO'} |
| **Total Steps** | {total_steps} |
| **Total Reward** | {total_reward:.4f} |
| **Action Steps per Call** | {self.n_action_steps} |

---

## How This Evaluation Works

This script uses the **exact same setup** as `rollout_policy.py`:

1. **Environment**: Created via `get_robocasa_env_fn()` with `enable_render=True`
2. **MultiStepWrapper**: Executes {self.n_action_steps} actions per policy call
3. **Policy**: Connected via `create_gr00t_sim_policy()` to the server

### Evaluation Flow

```
FOR EACH POLICY CALL:
  1. Observe: Get camera images + robot state + task instruction
  2. Predict: Policy returns 8 future actions (action chunk)
  3. Execute: MultiStepWrapper executes all 8 actions
  4. Check: If success detected, episode ends
  5. Repeat until success or max steps
```

---

## Understanding the Model

**GR00T N1.6** is a Vision-Language-Action model that:
- Takes camera images as visual input
- Takes robot joint states as proprioception
- Takes language instruction as task specification
- Outputs 8 future actions as a coherent sequence

The model uses **action chunking** because:
- Predicting multiple future actions improves temporal consistency
- Reduces compounding errors from frequent re-planning
- Enables smoother, more human-like motion

---

*Report generated by transparent_evaluation.py*
"""
        report_path = self.save_dir / "transparent_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"📄 Report saved: {report_path}")

    def close(self):
        """Clean up."""
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Transparent RoboCasa Evaluation")
    parser.add_argument("--env-name", type=str,
                        default="robocasa_panda_omron/CloseDrawer_PandaOmron_Env")
    parser.add_argument("--policy-host", type=str, default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--save-dir", type=str, default="./transparent_eval")
    parser.add_argument("--max-steps", type=int, default=504)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--n-episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--no-video", action="store_true",
                        help="Disable video recording")
    parser.add_argument("--camera-view", type=str, default="side",
                        choices=["side", "wrist", "both", "all"],
                        help="Camera view for video: 'side' (recommended), 'wrist', 'both', or 'all' (default: side)")

    args = parser.parse_args()

    evaluator = TransparentEvaluator(
        env_name=args.env_name,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        save_dir=args.save_dir,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_steps,
        save_video=not args.no_video,
        camera_view=args.camera_view,
    )

    try:
        if args.n_episodes == 1:
            result = evaluator.run_episode()
        else:
            result = evaluator.run_multiple_episodes(args.n_episodes)
    finally:
        evaluator.close()

    return result


if __name__ == "__main__":
    main()
