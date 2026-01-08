#!/usr/bin/env python3
"""
GR00T RoboCasa Evaluation Visualizer

This script provides detailed visualization and analysis of:
1. Sample inputs (images, robot state, language instructions)
2. Model outputs (action predictions)
3. Simulation rollout with step-by-step visualization
4. Success criteria explanation

Usage:
    python scripts/visualize_evaluation.py --save_dir ./eval_visualization

Requires: RoboCasa environment and GR00T policy server running on port 5555
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np

# Set rendering backend before imports
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym
import robocasa
import robocasa.utils.gym_utils.gymnasium_groot

# Optional: for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Some visualizations will be skipped.")

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("WARNING: imageio not installed. Video saving will be skipped.")


class EvaluationVisualizer:
    """Visualize and analyze GR00T evaluation on RoboCasa."""

    def __init__(self, env_name: str, save_dir: str, policy_host: str = "127.0.0.1", policy_port: int = 5555):
        self.env_name = env_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.policy_host = policy_host
        self.policy_port = policy_port

        # Create environment
        print(f"\n{'='*60}")
        print("CREATING ROBOCASA ENVIRONMENT")
        print(f"{'='*60}")
        self.env = gym.make(env_name, enable_render=False)
        print(f"Environment: {env_name}")
        print(f"Observation space: {self.env.observation_space}")
        print(f"Action space: {self.env.action_space}")

        # Try to connect to policy server
        self.policy_client = None
        try:
            from gr00t.policy.server_client import PolicyClient
            self.policy_client = PolicyClient(host=policy_host, port=policy_port)
            print(f"\nConnected to policy server at {policy_host}:{policy_port}")
        except Exception as e:
            print(f"\nWARNING: Could not connect to policy server: {e}")
            print("Running in observation-only mode (no actions from model)")

    def analyze_observation_space(self):
        """Analyze and document the observation space structure."""
        print(f"\n{'='*60}")
        print("OBSERVATION SPACE ANALYSIS")
        print(f"{'='*60}")

        obs_space = self.env.observation_space

        analysis = {
            "video_inputs": {},
            "state_inputs": {},
            "language_inputs": {}
        }

        for key, space in obs_space.spaces.items():
            if key.startswith("video."):
                camera_name = key.replace("video.", "")
                analysis["video_inputs"][camera_name] = {
                    "shape": space.shape,
                    "dtype": str(space.dtype),
                    "description": f"RGB image from {camera_name}"
                }
                print(f"📷 {key}: shape={space.shape}, dtype={space.dtype}")

            elif key.startswith("state."):
                state_name = key.replace("state.", "")
                analysis["state_inputs"][state_name] = {
                    "shape": space.shape,
                    "dtype": str(space.dtype),
                    "low": float(space.low.min()),
                    "high": float(space.high.max())
                }
                print(f"🤖 {key}: shape={space.shape}, range=[{space.low.min():.2f}, {space.high.max():.2f}]")

            elif key.startswith("annotation."):
                analysis["language_inputs"][key] = str(type(space))
                print(f"💬 {key}: {type(space).__name__}")

        # Save analysis
        with open(self.save_dir / "observation_space_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        return analysis

    def analyze_action_space(self):
        """Analyze and document the action space structure."""
        print(f"\n{'='*60}")
        print("ACTION SPACE ANALYSIS")
        print(f"{'='*60}")

        action_space = self.env.action_space

        analysis = {}
        for key, space in action_space.spaces.items():
            if hasattr(space, 'shape'):
                analysis[key] = {
                    "shape": space.shape,
                    "dtype": str(space.dtype) if hasattr(space, 'dtype') else "N/A",
                    "type": type(space).__name__
                }
                if hasattr(space, 'low'):
                    analysis[key]["range"] = [float(space.low.min()), float(space.high.max())]
                print(f"🎮 {key}: {type(space).__name__}, shape={getattr(space, 'shape', 'N/A')}")
            else:
                analysis[key] = {"type": type(space).__name__, "n": getattr(space, 'n', 'N/A')}
                print(f"🎮 {key}: {type(space).__name__}, n={getattr(space, 'n', 'N/A')}")

        # Save analysis
        with open(self.save_dir / "action_space_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        return analysis

    def get_sample_observation(self):
        """Get and visualize a sample observation from the environment."""
        print(f"\n{'='*60}")
        print("SAMPLE OBSERVATION")
        print(f"{'='*60}")

        obs, info = self.env.reset()

        # Extract components
        sample = {
            "video": {},
            "state": {},
            "language": {}
        }

        print("\n📷 VIDEO INPUTS (Camera Images):")
        print("-" * 40)
        for key, value in obs.items():
            if key.startswith("video."):
                camera_name = key.replace("video.", "")
                sample["video"][camera_name] = value
                print(f"  {camera_name}: shape={value.shape}, dtype={value.dtype}, range=[{value.min()}, {value.max()}]")

                # Save image
                if HAS_MATPLOTLIB:
                    img_path = self.save_dir / f"sample_{camera_name}.png"
                    plt.imsave(str(img_path), value)
                    print(f"    → Saved to {img_path}")

        print("\n🤖 STATE INPUTS (Robot Proprioception):")
        print("-" * 40)
        for key, value in obs.items():
            if key.startswith("state."):
                state_name = key.replace("state.", "")
                sample["state"][state_name] = value.tolist()
                print(f"  {state_name}:")
                print(f"    shape={value.shape}, values={np.round(value, 4)}")

        print("\n💬 LANGUAGE INPUT (Task Description):")
        print("-" * 40)
        for key, value in obs.items():
            if key.startswith("annotation."):
                sample["language"][key] = value
                print(f"  {key}: \"{value}\"")

        # Save sample data
        sample_serializable = {
            "video": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in sample["video"].items()},
            "state": sample["state"],
            "language": sample["language"]
        }
        with open(self.save_dir / "sample_observation.json", "w") as f:
            json.dump(sample_serializable, f, indent=2)

        return obs, info

    def visualize_observation_grid(self, obs, step: int = 0):
        """Create a grid visualization of all camera views."""
        if not HAS_MATPLOTLIB:
            return None

        video_keys = [k for k in obs.keys() if k.startswith("video.") and "256" in k]
        n_cameras = len(video_keys)

        if n_cameras == 0:
            return None

        fig, axes = plt.subplots(1, n_cameras, figsize=(4 * n_cameras, 4))
        if n_cameras == 1:
            axes = [axes]

        for ax, key in zip(axes, video_keys):
            camera_name = key.replace("video.", "")
            ax.imshow(obs[key])
            ax.set_title(camera_name, fontsize=10)
            ax.axis('off')

        plt.suptitle(f"Camera Views - Step {step}", fontsize=12)
        plt.tight_layout()

        save_path = self.save_dir / f"camera_grid_step_{step:04d}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

        return save_path

    def explain_success_criteria(self):
        """Explain how success is measured for this task."""
        print(f"\n{'='*60}")
        print("SUCCESS CRITERIA EXPLANATION")
        print(f"{'='*60}")

        task_name = self.env_name.split("/")[1].replace("_PandaOmron_Env", "")

        explanations = {
            "CloseDrawer": {
                "goal": "Close an open drawer completely",
                "success_condition": "Drawer joint position < threshold (fully closed)",
                "failure_modes": [
                    "Robot doesn't reach the drawer handle",
                    "Robot pushes drawer but doesn't close fully",
                    "Robot knocks over objects while attempting task"
                ],
                "metrics": [
                    "Binary success (0 or 1)",
                    "Measured at end of episode (max steps or early termination)"
                ]
            },
            "OpenDrawer": {
                "goal": "Open a closed drawer",
                "success_condition": "Drawer joint position > threshold (sufficiently open)",
                "failure_modes": [
                    "Robot doesn't grasp handle correctly",
                    "Robot pulls but drawer doesn't open enough"
                ],
                "metrics": ["Binary success (0 or 1)"]
            },
            "CoffeePressButton": {
                "goal": "Press the button on the coffee machine",
                "success_condition": "Button state changes to pressed",
                "failure_modes": [
                    "Robot misses the button",
                    "Insufficient force applied"
                ],
                "metrics": ["Binary success (0 or 1)"]
            }
        }

        if task_name in explanations:
            exp = explanations[task_name]
            print(f"\n📋 Task: {task_name}")
            print(f"\n🎯 Goal: {exp['goal']}")
            print(f"\n✅ Success Condition: {exp['success_condition']}")
            print(f"\n❌ Common Failure Modes:")
            for mode in exp['failure_modes']:
                print(f"   • {mode}")
            print(f"\n📊 Metrics:")
            for metric in exp['metrics']:
                print(f"   • {metric}")
        else:
            print(f"\nTask '{task_name}' - Generic manipulation task")
            print("Success typically measured by task-specific conditions in environment")

        print(f"\n{'='*60}")
        print("SUCCESS RATE CALCULATION")
        print(f"{'='*60}")
        print("""
Success Rate = (# Successful Episodes) / (Total Episodes) × 100%

For example, with 10 episodes:
- 10 successes → 100% success rate
- 8 successes  → 80% success rate
- 5 successes  → 50% success rate

The paper reports:
- CloseDrawer: 100% (zero-shot, N1.6)
- This is the EASIEST task in RoboCasa benchmark
- More complex tasks like PnP (Pick and Place) have lower rates
""")

        return explanations.get(task_name, {})

    def run_episode_with_visualization(self, max_steps: int = 200, save_video: bool = True):
        """Run a full episode with step-by-step visualization."""
        print(f"\n{'='*60}")
        print("RUNNING EPISODE WITH VISUALIZATION")
        print(f"{'='*60}")

        obs, info = self.env.reset()

        frames = []
        states = []
        actions_taken = []
        rewards = []

        print(f"\n🎬 Starting episode...")
        print(f"   Max steps: {max_steps}")
        print(f"   Task: {obs.get('annotation.human.action.task_description', 'N/A')}")

        done = False
        step = 0
        total_reward = 0
        success = False

        while not done and step < max_steps:
            # Save frame for video
            video_key = [k for k in obs.keys() if "res256" in k and "side_0" in k]
            if video_key:
                frames.append(obs[video_key[0]].copy())

            # Get action from policy server or use random
            if self.policy_client:
                try:
                    # Format observation for policy
                    policy_obs = self._format_obs_for_policy(obs)
                    action = self.policy_client.get_action(policy_obs)
                except Exception as e:
                    print(f"   Step {step}: Policy error: {e}, using random action")
                    action = self.env.action_space.sample()
            else:
                action = self.env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            total_reward += reward
            actions_taken.append(self._action_to_dict(action))
            rewards.append(reward)

            # Log progress every 50 steps
            if step % 50 == 0:
                print(f"   Step {step}: reward={reward:.4f}, total_reward={total_reward:.4f}")

            # Check for success
            if info.get("success", False):
                success = True
                print(f"\n   ✅ SUCCESS at step {step}!")
                break

            step += 1

        # Save final frame
        if video_key and len(frames) > 0:
            frames.append(obs[video_key[0]].copy())

        print(f"\n📊 Episode Summary:")
        print(f"   Total steps: {step}")
        print(f"   Total reward: {total_reward:.4f}")
        print(f"   Success: {'✅ YES' if success else '❌ NO'}")

        # Save video
        if save_video and HAS_IMAGEIO and len(frames) > 0:
            video_path = self.save_dir / "episode_rollout.mp4"
            imageio.mimsave(str(video_path), frames, fps=30)
            print(f"   Video saved: {video_path}")

        # Save episode data
        episode_data = {
            "total_steps": step,
            "total_reward": total_reward,
            "success": success,
            "rewards_per_step": rewards[:20],  # First 20 for brevity
            "actions_sample": actions_taken[:5]  # First 5 for brevity
        }
        with open(self.save_dir / "episode_summary.json", "w") as f:
            json.dump(episode_data, f, indent=2)

        return success, step, total_reward

    def _format_obs_for_policy(self, obs):
        """Format gym observation for GR00T policy."""
        # This is a simplified version - actual formatting depends on policy requirements
        formatted = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                # Add batch dimension if needed
                if len(value.shape) == 3:  # Image: (H, W, C)
                    formatted[key] = value[np.newaxis, np.newaxis, ...]  # (1, 1, H, W, C)
                else:
                    formatted[key] = value[np.newaxis, np.newaxis, ...]  # (1, 1, D)
            else:
                formatted[key] = value
        return formatted

    def _action_to_dict(self, action):
        """Convert action to serializable dict."""
        if isinstance(action, dict):
            return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in action.items()}
        elif isinstance(action, np.ndarray):
            return action.tolist()
        return action

    def generate_report(self):
        """Generate a comprehensive markdown report."""
        print(f"\n{'='*60}")
        print("GENERATING COMPREHENSIVE REPORT")
        print(f"{'='*60}")

        report = f"""# GR00T RoboCasa Evaluation Analysis

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Environment**: {self.env_name}

## 1. Overview

This report analyzes the GR00T N1.6 model's performance on the RoboCasa kitchen manipulation benchmark.

### What is being tested?
- **Model**: GR00T N1.6-3B (Vision-Language-Action foundation model)
- **Task**: {self.env_name.split("/")[1].replace("_PandaOmron_Env", "")}
- **Robot**: Panda arm with Omron gripper
- **Mode**: Zero-shot (no task-specific fine-tuning)

## 2. Input Format

The model receives multimodal inputs at each timestep:

### 2.1 Visual Input (Cameras)
| Camera | Resolution | Purpose |
|--------|------------|---------|
| res256_image_side_0 | 256×256 | Side view of workspace |
| res256_image_side_1 | 256×256 | Alternative side view |
| res256_image_wrist_0 | 256×256 | Wrist-mounted camera |
| res512_image_* | 512×512 | High-res versions |

### 2.2 State Input (Proprioception)
| State | Dimension | Description |
|-------|-----------|-------------|
| joint_position | 7 | Arm joint angles (radians) |
| joint_velocity | 7 | Arm joint velocities |
| gripper_qpos | 2 | Gripper finger positions |
| end_effector_position | 3 | EE XYZ position |
| end_effector_rotation | 4 | EE orientation (quaternion) |

### 2.3 Language Input
- Natural language task description
- Example: "Close the drawer"

## 3. Output Format (Actions)

The model outputs a trajectory of actions:

| Action | Dimension | Description |
|--------|-----------|-------------|
| end_effector_position | 3 | Target EE XYZ delta |
| end_effector_rotation | 3 | Target EE rotation delta |
| gripper_close | 1 (discrete) | 0=open, 1=close |
| base_motion | 4 | Mobile base control |
| control_mode | 1 (discrete) | Control mode selection |

## 4. Evaluation Protocol

### 4.1 Episode Structure
1. Environment resets to random initial state
2. Robot receives task instruction
3. Model predicts actions for up to N steps (typically 200-500)
4. Episode ends on success, failure, or timeout

### 4.2 Success Criteria
For **CloseDrawer**:
- ✅ Success: Drawer joint position < closed threshold
- ❌ Failure: Timeout without closing drawer

### 4.3 Success Rate Calculation
```
Success Rate = (Successful Episodes / Total Episodes) × 100%
```

## 5. Understanding Model Behavior

### 5.1 Action Prediction Process
```
Observation → [Vision Encoder] → Visual Features
                                        ↓
Language → [Language Encoder] ────→ [Fusion] → [Diffusion Transformer] → Actions
                                        ↑
State → [State Encoder] ──────────────────
```

### 5.2 Key Model Components
1. **Eagle Vision Encoder**: Processes camera images at multiple resolutions
2. **Language Encoder**: Encodes task instructions
3. **Diffusion Transformer**: Generates action trajectories
4. **Action Horizon**: Predicts 8-16 future actions at once

### 5.3 Inference Loop
```python
while not done:
    # 1. Get observation from environment
    obs = env.get_observation()

    # 2. Model predicts action chunk (e.g., 8 actions)
    actions = model.predict(obs)  # Shape: (8, action_dim)

    # 3. Execute actions (can execute all 8 or just first few)
    for action in actions[:n_action_steps]:
        obs, reward, done, info = env.step(action)
        if done:
            break
```

## 6. Paper Results Reference

From the GR00T N1 paper (arXiv:2503.14734):

| Task | Diffusion Policy | GR00T N1 (100 demos) | GR00T N1.6 (Zero-shot) |
|------|------------------|----------------------|------------------------|
| CloseDrawer | 88.2% | 96.1% | **100.0%** |
| Average (24 tasks) | 25.6% | 32.1% | **66.2%** |

## 7. Files Generated

- `sample_*.png` - Camera view samples
- `observation_space_analysis.json` - Observation space details
- `action_space_analysis.json` - Action space details
- `sample_observation.json` - Sample observation data
- `episode_summary.json` - Episode rollout results
- `episode_rollout.mp4` - Video of episode (if available)

## 8. Interpreting Results

### High Success Rate (>90%)
- Model has strong understanding of task
- Visual-language grounding is effective
- Action predictions are accurate

### Medium Success Rate (50-90%)
- Model understands task but execution varies
- May struggle with certain initial configurations
- Consider fine-tuning for improvement

### Low Success Rate (<50%)
- Task may be out of distribution
- Consider more demonstrations
- Check observation/action space compatibility

---

*Report generated by GR00T Evaluation Visualizer*
"""

        report_path = self.save_dir / "evaluation_report.md"
        with open(report_path, "w") as f:
            f.write(report)

        print(f"Report saved to: {report_path}")
        return report_path

    def close(self):
        """Clean up resources."""
        self.env.close()
        if self.policy_client:
            try:
                self.policy_client.close()
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description="Visualize GR00T RoboCasa Evaluation")
    parser.add_argument("--env_name", type=str,
                        default="robocasa_panda_omron/CloseDrawer_PandaOmron_Env",
                        help="RoboCasa environment name")
    parser.add_argument("--save_dir", type=str, default="./eval_visualization",
                        help="Directory to save visualization outputs")
    parser.add_argument("--policy_host", type=str, default="127.0.0.1",
                        help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=5555,
                        help="Policy server port")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Maximum steps per episode")
    parser.add_argument("--run_episode", action="store_true",
                        help="Run a full episode with the policy")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("GR00T ROBOCASA EVALUATION VISUALIZER")
    print("="*60)

    visualizer = EvaluationVisualizer(
        env_name=args.env_name,
        save_dir=args.save_dir,
        policy_host=args.policy_host,
        policy_port=args.policy_port
    )

    try:
        # Analyze spaces
        visualizer.analyze_observation_space()
        visualizer.analyze_action_space()

        # Get sample observation
        visualizer.get_sample_observation()

        # Explain success criteria
        visualizer.explain_success_criteria()

        # Optionally run episode
        if args.run_episode:
            visualizer.run_episode_with_visualization(max_steps=args.max_steps)

        # Generate report
        visualizer.generate_report()

        print(f"\n{'='*60}")
        print("VISUALIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"All outputs saved to: {args.save_dir}")

    finally:
        visualizer.close()


if __name__ == "__main__":
    main()
