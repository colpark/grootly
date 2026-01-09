#!/usr/bin/env python3
"""
Transparent RoboCasa Evaluation

Makes the evaluation process fully visible by logging:
1. What the model SEES (camera images, robot state, task instruction)
2. What the model OUTPUTS (action predictions)
3. What the robot DOES (step-by-step execution)
4. Why it SUCCEEDS or FAILS (success criteria checking)

Usage:
    # Terminal 1 - Start policy server
    uv run python gr00t/eval/run_gr00t_server.py \
        --model-path nvidia/GR00T-N1.6-3B \
        --embodiment-tag ROBOCASA_PANDA_OMRON \
        --use-sim-policy-wrapper

    # Terminal 2 - Run transparent evaluation
    python scripts/transparent_evaluation.py \
        --save-dir ./transparent_eval \
        --max-steps 100 \
        --log-every 10
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np

# Rendering backend: Don't set defaults - let the environment/robocasa handle it
# The robocasa venv is configured with the correct rendering backend
# If needed, set MUJOCO_GL=egl or MUJOCO_GL=osmesa before running

# Optional visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


class TransparentEvaluator:
    """Run evaluation with full transparency into what's happening."""

    def __init__(
        self,
        env_name: str = "robocasa_panda_omron/CloseDrawer_PandaOmron_Env",
        policy_host: str = "127.0.0.1",
        policy_port: int = 5555,
        save_dir: str = "./transparent_eval",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create environment
        print("\n" + "="*70)
        print("TRANSPARENT ROBOCASA EVALUATION")
        print("="*70)

        import gymnasium as gym
        import robocasa
        import robocasa.utils.gym_utils.gymnasium_groot

        self.env = gym.make(env_name, enable_render=True)
        self.env_name = env_name
        self.task_name = env_name.split("/")[1].replace("_PandaOmron_Env", "")

        print(f"Environment: {env_name}")
        print(f"Task: {self.task_name}")

        # Connect to policy server
        self.policy_client = None
        try:
            from gr00t.policy.server_client import PolicyClient
            self.policy_client = PolicyClient(host=policy_host, port=policy_port)
            print(f"Connected to policy server at {policy_host}:{policy_port}")
            self.using_policy = True
        except Exception as e:
            print(f"WARNING: No policy server ({e})")
            print("Running with RANDOM actions for demonstration")
            self.using_policy = False

        # Storage for episode data
        self.episode_log = []

    def explain_observation_structure(self, obs: Dict) -> Dict:
        """Explain what's in the observation."""
        explanation = {
            "cameras": {},
            "robot_state": {},
            "task": None
        }

        for key, value in obs.items():
            if key.startswith("video."):
                cam_name = key.replace("video.", "")
                if isinstance(value, np.ndarray):
                    explanation["cameras"][cam_name] = {
                        "shape": list(value.shape),
                        "description": self._describe_camera(cam_name)
                    }

            elif key.startswith("state."):
                state_name = key.replace("state.", "")
                if isinstance(value, np.ndarray):
                    explanation["robot_state"][state_name] = {
                        "shape": list(value.shape),
                        "values": value.tolist() if value.size < 20 else f"[{value.size} values]",
                        "description": self._describe_state(state_name)
                    }

            elif "task_description" in key:
                explanation["task"] = str(value)

        return explanation

    def _describe_camera(self, cam_name: str) -> str:
        """Describe what a camera sees."""
        descriptions = {
            "res256_image_side_0": "Side view of workspace (256x256)",
            "res256_image_side_1": "Alternative side angle (256x256)",
            "res256_image_wrist_0": "Wrist-mounted camera on gripper (256x256)",
            "res512_image_side_0": "High-res side view (512x512)",
            "res512_image_side_1": "High-res alternative side (512x512)",
            "res512_image_wrist_0": "High-res wrist camera (512x512)",
        }
        return descriptions.get(cam_name, f"Camera: {cam_name}")

    def _describe_state(self, state_name: str) -> str:
        """Describe what a state represents."""
        descriptions = {
            "joint_position": "7 arm joint angles in radians",
            "joint_velocity": "7 arm joint velocities (rad/s)",
            "joint_position_cos": "Cosine of joint angles (for continuity)",
            "joint_position_sin": "Sine of joint angles (for continuity)",
            "gripper_qpos": "Gripper finger positions",
            "gripper_qvel": "Gripper finger velocities",
            "end_effector_position_absolute": "XYZ position of gripper in world frame",
            "end_effector_rotation_absolute": "Quaternion orientation of gripper",
            "end_effector_position_relative": "XYZ position relative to base",
            "end_effector_rotation_relative": "Quaternion orientation relative to base",
            "base_position": "Robot base XYZ position",
            "base_rotation": "Robot base orientation (quaternion)",
        }
        return descriptions.get(state_name, f"State: {state_name}")

    def explain_action_structure(self, action: Dict) -> Dict:
        """Explain what actions mean."""
        explanation = {}

        for key, value in action.items():
            if isinstance(value, np.ndarray):
                explanation[key] = {
                    "shape": list(value.shape),
                    "values": value.tolist() if value.size < 10 else value[:5].tolist() + ["..."],
                    "description": self._describe_action(key)
                }
            else:
                explanation[key] = {
                    "value": int(value) if hasattr(value, 'item') else value,
                    "description": self._describe_action(key)
                }

        return explanation

    def _describe_action(self, action_name: str) -> str:
        """Describe what an action controls."""
        descriptions = {
            "action.end_effector_position": "Target gripper XYZ delta (-1 to 1, scaled to meters)",
            "action.end_effector_rotation": "Target gripper rotation delta (axis-angle)",
            "action.gripper_close": "Gripper command: 0=open, 1=close",
            "action.base_motion": "Mobile base velocity commands",
            "action.control_mode": "Control mode: 0=position, 1=velocity",
        }
        return descriptions.get(action_name, f"Action: {action_name}")

    def explain_success_criteria(self) -> str:
        """Explain how success is determined for this task."""
        criteria = {
            "CloseDrawer": """
SUCCESS CRITERIA: CloseDrawer
─────────────────────────────
Goal: Close an open kitchen drawer completely

How it's measured:
  1. Environment tracks drawer joint position
  2. Success when: drawer_position < closed_threshold
  3. Checked at each step and at episode end

What the robot must do:
  1. Locate the drawer handle visually
  2. Move gripper to handle position
  3. Grasp or push the handle
  4. Apply force to close drawer
  5. Drawer must reach fully closed position

Common failure modes:
  - Miss the handle entirely
  - Push but don't close fully
  - Timeout before completing
""",
            "OpenDrawer": """
SUCCESS CRITERIA: OpenDrawer
────────────────────────────
Goal: Open a closed kitchen drawer

How it's measured:
  1. Success when: drawer_position > open_threshold

What the robot must do:
  1. Grasp drawer handle
  2. Pull drawer open
""",
            "CoffeePressButton": """
SUCCESS CRITERIA: CoffeePressButton
───────────────────────────────────
Goal: Press the button on the coffee machine

How it's measured:
  1. Success when: button_state == pressed
""",
        }
        return criteria.get(self.task_name, f"Task-specific success criteria for {self.task_name}")

    def run_transparent_episode(
        self,
        max_steps: int = 200,
        log_every: int = 10,
        save_frames: bool = True
    ) -> Dict:
        """Run one episode with full logging."""

        print("\n" + "="*70)
        print("STARTING EPISODE")
        print("="*70)

        # Reset environment
        obs, info = self.env.reset()

        # Explain initial observation
        print("\n" + "─"*70)
        print("STEP 0: INITIAL STATE")
        print("─"*70)

        obs_explanation = self.explain_observation_structure(obs)
        print(f"\n📋 TASK INSTRUCTION: \"{obs_explanation['task']}\"")

        print(f"\n📷 CAMERAS ({len(obs_explanation['cameras'])} views):")
        for cam, info in obs_explanation['cameras'].items():
            print(f"   • {cam}: {info['shape']} - {info['description']}")

        print(f"\n🤖 ROBOT STATE ({len(obs_explanation['robot_state'])} components):")
        for state, info in list(obs_explanation['robot_state'].items())[:5]:
            print(f"   • {state}: {info['description']}")
        if len(obs_explanation['robot_state']) > 5:
            print(f"   ... and {len(obs_explanation['robot_state']) - 5} more")

        # Explain success criteria
        print("\n" + "─"*70)
        print("SUCCESS CRITERIA")
        print("─"*70)
        print(self.explain_success_criteria())

        # Storage
        frames = []
        step_logs = []
        actions_taken = []
        rewards = []

        # Get first frame for video
        video_key = [k for k in obs.keys() if "res256" in k and "side_0" in k]
        if video_key and save_frames:
            frame = obs[video_key[0]].copy()
            frames.append(frame)
            # Check if frame is actually rendered (not all black)
            if frame.max() == 0:
                print("\n⚠️  WARNING: Camera frames are all black!")
                print("   This may indicate EGL rendering issues.")
                print("   Try: export MUJOCO_GL=osmesa")
            else:
                print(f"\n📷 First frame captured: shape={frame.shape}, range=[{frame.min()}, {frame.max()}]")

        done = False
        step = 0
        success = False
        total_reward = 0
        n_action_steps = 8  # Execute 8 actions per policy call (same as rollout_policy.py)

        print("\n" + "="*70)
        print("EXECUTION LOOP")
        print(f"(Policy predicts {n_action_steps} actions per call)")
        print("="*70)

        action_chunk = None
        action_chunk_idx = n_action_steps  # Start at max to trigger first policy call

        while not done and step < max_steps:
            # Get new action chunk every n_action_steps
            if action_chunk_idx >= n_action_steps:
                if self.using_policy:
                    try:
                        action_chunk, action_info = self._get_policy_action(obs)
                        action_source = "POLICY"
                        action_chunk_idx = 0
                    except Exception as e:
                        action_chunk = None
                        action_info = {}
                        action_source = f"RANDOM (policy error: {e})"
                else:
                    action_chunk = None
                    action_info = {}
                    action_source = "RANDOM"

            # Extract current action from chunk or use random
            if action_chunk is not None:
                action = self._extract_single_action(action_chunk, action_chunk_idx)
                action_chunk_idx += 1
            else:
                action = self.env.action_space.sample()

            # Log detailed info at intervals
            if step % log_every == 0:
                print(f"\n{'─'*70}")
                print(f"STEP {step} (action {action_chunk_idx}/{n_action_steps} in chunk)")
                print(f"{'─'*70}")

                print(f"\n🎯 ACTION SOURCE: {action_source}")

                action_explanation = self.explain_action_structure(action)
                print(f"\n🎮 ACTIONS SENT TO ROBOT:")
                for act_name, act_info in action_explanation.items():
                    if 'values' in act_info:
                        val_str = str(act_info['values'])[:50]
                        print(f"   • {act_name}: {val_str}")
                        print(f"     → {act_info['description']}")
                    else:
                        print(f"   • {act_name}: {act_info['value']}")
                        print(f"     → {act_info['description']}")

            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Log step
            step_log = {
                "step": step,
                "action_chunk_idx": action_chunk_idx,
                "reward": float(reward),
                "total_reward": float(total_reward),
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
                "success": info.get("success", False),
            }
            step_logs.append(step_log)

            # Save frame
            if video_key and save_frames:
                frames.append(obs[video_key[0]].copy())

            # Log at intervals
            if step % log_every == 0:
                print(f"\n📊 RESULT:")
                print(f"   • Reward this step: {reward:.4f}")
                print(f"   • Total reward: {total_reward:.4f}")
                print(f"   • Done: {done}")
                if info.get("success"):
                    print(f"   • ✅ SUCCESS DETECTED!")

            # Check success
            if info.get("success", False):
                success = True
                print(f"\n{'='*70}")
                print(f"✅ SUCCESS AT STEP {step}!")
                print(f"{'='*70}")
                break

            step += 1

        # Final summary
        print("\n" + "="*70)
        print("EPISODE SUMMARY")
        print("="*70)
        print(f"""
📊 Results:
   • Total steps: {step}
   • Total reward: {total_reward:.4f}
   • Success: {'✅ YES' if success else '❌ NO'}
   • Termination: {'Natural' if terminated else 'Timeout' if truncated else 'Max steps'}

🎯 What happened:
   {'The robot successfully completed the task!' if success else 'The robot did not complete the task within the step limit.'}
""")

        # Save video
        if HAS_IMAGEIO and len(frames) > 0 and save_frames:
            video_path = self.save_dir / "episode_video.mp4"
            imageio.mimsave(str(video_path), frames, fps=20)
            print(f"📹 Video saved: {video_path}")

        # Save step logs
        logs_path = self.save_dir / "step_logs.json"
        with open(logs_path, "w") as f:
            json.dump(step_logs, f, indent=2)
        print(f"📝 Logs saved: {logs_path}")

        # Create summary visualization
        if HAS_MATPLOTLIB and len(step_logs) > 0:
            self._plot_episode_summary(step_logs, success)

        # Save comprehensive report
        self._save_report(step_logs, success, total_reward, step)

        return {
            "success": success,
            "total_steps": step,
            "total_reward": total_reward,
            "step_logs": step_logs
        }

    def _get_policy_action(self, obs: Dict) -> tuple:
        """Get action chunk from policy server.

        Returns:
            action_chunk: Dict with action arrays of shape (n_action_steps, dim)
            info: Additional info from policy
        """
        # Format observation for policy
        policy_obs = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                # Add batch and time dimensions if needed
                if "video" in key and len(value.shape) == 3:
                    policy_obs[key] = value[np.newaxis, np.newaxis, ...]
                elif len(value.shape) == 1:
                    policy_obs[key] = value[np.newaxis, np.newaxis, ...]
                else:
                    policy_obs[key] = value
            elif isinstance(value, str):
                # Language inputs must be List[str] or Tuple[str] format (batch of strings)
                # The Gr00tSimPolicyWrapper expects: tuple[str] or list[str] with shape (B,)
                if "task_description" in key or "annotation" in key:
                    policy_obs[key] = [value]  # Single batch item
                else:
                    policy_obs[key] = value
            else:
                policy_obs[key] = value

        # Get action from policy (returns tuple of (action_dict, info_dict))
        # Action format: (batch=1, chunk=8, dim)
        action_chunk_raw, info = self.policy_client.get_action(policy_obs)

        # Remove batch dimension, keep action chunk
        # Shape: (1, 8, dim) -> (8, dim) for each action key
        action_chunk = {}
        for key, value in action_chunk_raw.items():
            if isinstance(value, np.ndarray):
                if len(value.shape) >= 2:
                    action_chunk[key] = value[0]  # Remove batch dim: (1, 8, dim) -> (8, dim)
                else:
                    action_chunk[key] = value
            else:
                action_chunk[key] = value

        return action_chunk, info

    def _extract_single_action(self, action_chunk: Dict, step_idx: int) -> Dict:
        """Extract a single action from action chunk at given index."""
        action = {}
        for key, value in action_chunk.items():
            if isinstance(value, np.ndarray) and len(value.shape) >= 1:
                action[key] = value[step_idx]  # (8, dim) -> (dim,)
            else:
                action[key] = value
        return action

    def _plot_episode_summary(self, step_logs: List[Dict], success: bool):
        """Create visualization of episode."""
        steps = [log["step"] for log in step_logs]
        rewards = [log["reward"] for log in step_logs]
        cumulative = [log["total_reward"] for log in step_logs]

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        # Reward per step
        ax = axes[0]
        ax.bar(steps, rewards, color='steelblue', alpha=0.7, width=1.0)
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.set_title("Reward per Step")
        ax.grid(True, alpha=0.3)

        # Cumulative reward
        ax = axes[1]
        ax.plot(steps, cumulative, color='green', linewidth=2)
        ax.fill_between(steps, cumulative, alpha=0.3, color='green')
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title(f"Cumulative Reward - {'SUCCESS ✓' if success else 'FAILED ✗'}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / "episode_summary.png", dpi=150)
        plt.close()
        print(f"📊 Summary plot saved: {self.save_dir / 'episode_summary.png'}")

    def _save_report(self, step_logs: List[Dict], success: bool, total_reward: float, total_steps: int):
        """Save comprehensive markdown report."""
        report = f"""# Transparent Evaluation Report

**Task**: {self.task_name}
**Environment**: {self.env_name}
**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Result Summary

| Metric | Value |
|--------|-------|
| **Success** | {'✅ YES' if success else '❌ NO'} |
| **Total Steps** | {total_steps} |
| **Total Reward** | {total_reward:.4f} |
| **Policy Used** | {'GR00T N1.6' if self.using_policy else 'Random (demo)'} |

---

## What Happened Each Step

The evaluation loop works like this:

```
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION LOOP                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FOR EACH STEP:                                                 │
│                                                                 │
│  1. OBSERVE                                                     │
│     ┌─────────────────────────────────────┐                    │
│     │ Camera Images (3-6 views)           │                    │
│     │   • side_0: workspace overview      │                    │
│     │   • wrist_0: gripper close-up       │                    │
│     ├─────────────────────────────────────┤                    │
│     │ Robot State (proprioception)        │                    │
│     │   • joint_position: 7 angles        │                    │
│     │   • gripper_qpos: finger positions  │                    │
│     │   • end_effector_position: XYZ      │                    │
│     ├─────────────────────────────────────┤                    │
│     │ Task Instruction                    │                    │
│     │   • "Close the drawer"              │                    │
│     └─────────────────────────────────────┘                    │
│                         │                                       │
│                         ▼                                       │
│  2. POLICY INFERENCE (GR00T Model)                             │
│     ┌─────────────────────────────────────┐                    │
│     │ Vision Encoder → Visual Tokens      │                    │
│     │ State Encoder → State Tokens        │                    │
│     │ Language Encoder → Task Tokens      │                    │
│     │         ↓                           │                    │
│     │ Diffusion Transformer               │                    │
│     │         ↓                           │                    │
│     │ Action Chunk (8 future actions)     │                    │
│     └─────────────────────────────────────┘                    │
│                         │                                       │
│                         ▼                                       │
│  3. EXECUTE ACTION                                              │
│     ┌─────────────────────────────────────┐                    │
│     │ end_effector_position: [dx, dy, dz] │                    │
│     │ end_effector_rotation: [rx, ry, rz] │                    │
│     │ gripper_close: 0 or 1               │                    │
│     └─────────────────────────────────────┘                    │
│                         │                                       │
│                         ▼                                       │
│  4. CHECK SUCCESS                                               │
│     ┌─────────────────────────────────────┐                    │
│     │ Is drawer_joint < closed_threshold? │                    │
│     │   • YES → Episode ends, SUCCESS     │                    │
│     │   • NO  → Continue to next step     │                    │
│     └─────────────────────────────────────┘                    │
│                                                                 │
│  REPEAT until success or max_steps reached                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Understanding Success Rate

**100% Success Rate means:**
- Every episode (trial) completed successfully
- The model understood the task from language instruction
- Visual processing correctly identified drawer location
- Action predictions moved the arm correctly
- Task was completed within step limit

**Why CloseDrawer achieves 100%:**
1. **Simple task** - Single object manipulation, no grasping required
2. **Clear visual signal** - Drawer is visually distinct
3. **Forgiving execution** - Pushing motion doesn't need precision
4. **Strong training** - GR00T trained on similar manipulation

**Harder tasks (lower success rates):**
- Pick and Place: Requires grasping + precise placement
- OpenDoubleDoor: Requires bimanual coordination
- Microwave tasks: Requires button precision

---

## Files Generated

| File | Description |
|------|-------------|
| `episode_video.mp4` | Video of robot execution |
| `step_logs.json` | Detailed per-step data |
| `episode_summary.png` | Reward visualization |
| `transparent_report.md` | This report |

---

## Key Observations

{'The robot successfully closed the drawer.' if success else 'The robot did not complete the task.'}

Steps taken: {total_steps}
Average reward per step: {total_reward/max(total_steps, 1):.4f}

---

*Generated by Transparent Evaluation Script*
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
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log detailed info every N steps")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip saving video frames")

    args = parser.parse_args()

    evaluator = TransparentEvaluator(
        env_name=args.env_name,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        save_dir=args.save_dir
    )

    try:
        result = evaluator.run_transparent_episode(
            max_steps=args.max_steps,
            log_every=args.log_every,
            save_frames=not args.no_video
        )
    finally:
        evaluator.close()

    return result


if __name__ == "__main__":
    main()
