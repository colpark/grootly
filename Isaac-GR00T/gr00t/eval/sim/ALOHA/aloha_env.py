"""
ALOHA simulation environment wrapper for GR00T.

This file wraps gym-aloha as a Gymnasium environment compatible with GR00T's
evaluation infrastructure.

Installation:
    pip install gym-aloha

Or via LeRobot:
    pip install 'lerobot[aloha]'
"""

import os
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


class AlohaEnv(gym.Env):
    """
    GR00T-compatible wrapper for gym-aloha environments.

    Maps ALOHA's bimanual action space to GR1 format:
    - left_arm: 6 joint positions + 1 gripper
    - right_arm: 6 joint positions + 1 gripper
    """

    def __init__(
        self,
        task: str = "AlohaTransferCube-v0",
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        image_size: int = 256,
    ):
        import gym_aloha  # noqa: F401

        self._env = gym.make(
            f"gym_aloha/{task}",
            obs_type=obs_type,
            render_mode=render_mode,
        )
        self._task = task
        self._image_size = image_size

        # Task description for language conditioning
        if "TransferCube" in task:
            self._task_description = "Transfer the cube from left gripper to right gripper"
        elif "Insertion" in task:
            self._task_description = "Insert the peg into the socket"
        else:
            self._task_description = "Complete the manipulation task"

        # Define observation space matching GR1 format
        self.observation_space = gym.spaces.Dict({
            # Video observation
            "video.ego_view": gym.spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
            ),
            # State observations - matching GR1 modality config
            "state.left_arm": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(7,), dtype=np.float32),
            "state.left_hand": gym.spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
            "state.right_arm": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(7,), dtype=np.float32),
            "state.right_hand": gym.spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
            "state.waist": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32),
            # Language
            "task": gym.spaces.Text(max_length=512),
        })

        # Define action space matching GR1 format
        self.action_space = spaces.Dict({
            "action.left_arm": spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32),
            "action.left_hand": spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
            "action.right_arm": spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32),
            "action.right_hand": spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
            "action.waist": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        })

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        import cv2
        if image.shape[:2] != (self._image_size, self._image_size):
            image = cv2.resize(image, (self._image_size, self._image_size))
        return image

    def _process_observation(self, obs: dict) -> dict:
        """Convert gym-aloha observation to GR1 format."""
        # Get image - gym-aloha uses 'pixels' key
        if "pixels" in obs:
            image = obs["pixels"]["top"]  # Use top camera as ego view
        elif "image" in obs:
            image = obs["image"]
        else:
            # Render if not in observation
            image = self._env.render()

        image = self._resize_image(image)

        # Get agent positions - gym-aloha provides 14 DOF (7 per arm)
        agent_pos = obs.get("agent_pos", np.zeros(14))

        # Split into left and right arms (each has 6 joints + 1 gripper)
        left_pos = agent_pos[:7]   # First 7: left arm joints + gripper
        right_pos = agent_pos[7:]  # Last 7: right arm joints + gripper

        # Map to GR1 format with zero padding for unused DOFs
        new_obs = {
            "video.ego_view": image.astype(np.uint8),
            # Left arm: 6 joints from ALOHA, pad to 7
            "state.left_arm": np.concatenate([left_pos[:6], [0.0]]).astype(np.float32),
            # Left hand: gripper value, pad to 6
            "state.left_hand": np.concatenate([[left_pos[6]], np.zeros(5)]).astype(np.float32),
            # Right arm: 6 joints from ALOHA, pad to 7
            "state.right_arm": np.concatenate([right_pos[:6], [0.0]]).astype(np.float32),
            # Right hand: gripper value, pad to 6
            "state.right_hand": np.concatenate([[right_pos[6]], np.zeros(5)]).astype(np.float32),
            # Waist: zeros (ALOHA doesn't have waist)
            "state.waist": np.zeros(3, dtype=np.float32),
            # Task description
            "task": self._task_description,
        }
        return new_obs

    def _process_action(self, action: dict) -> np.ndarray:
        """Convert GR1 format action to gym-aloha format (14 DOF)."""
        # Extract relevant DOFs from GR1 action
        left_arm = action.get("action.left_arm", np.zeros(7))[:6]  # 6 joints
        left_gripper = action.get("action.left_hand", np.zeros(6))[0:1]  # 1 gripper
        right_arm = action.get("action.right_arm", np.zeros(7))[:6]  # 6 joints
        right_gripper = action.get("action.right_hand", np.zeros(6))[0:1]  # 1 gripper

        # Combine into 14 DOF action for gym-aloha
        action_vector = np.concatenate([
            left_arm, left_gripper,
            right_arm, right_gripper,
        ])
        return action_vector.astype(np.float32)

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        obs = self._process_observation(obs)
        return obs, info

    def step(self, action: dict):
        action_vector = self._process_action(action)
        obs, reward, terminated, truncated, info = self._env.step(action_vector)
        obs = self._process_observation(obs)

        # Check success if available
        if hasattr(self._env, "check_success"):
            info["success"] = self._env.check_success()
        elif "is_success" in info:
            info["success"] = info["is_success"]
        else:
            info["success"] = reward > 0

        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


def register_aloha_envs():
    """Register ALOHA environments with gymnasium."""
    tasks = [
        ("AlohaTransferCube-v0", "Transfer the cube from left gripper to right gripper"),
        ("AlohaInsertion-v0", "Insert the peg into the socket"),
    ]

    for task, description in tasks:
        env_id = f"aloha_sim/{task}"
        try:
            register(
                id=env_id,
                entry_point="gr00t.eval.sim.ALOHA.aloha_env:AlohaEnv",
                kwargs={"task": task},
            )
        except Exception:
            pass  # Already registered


if __name__ == "__main__":
    # Test the environment
    print("Testing ALOHA environment wrapper...")

    try:
        import gym_aloha  # noqa: F401
        print("gym-aloha is installed")
    except ImportError:
        print("gym-aloha not installed. Install with: pip install gym-aloha")
        exit(1)

    env = AlohaEnv(task="AlohaTransferCube-v0")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"Observation keys: {obs.keys()}")
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")

    # Take a random action
    action = {
        "action.left_arm": np.random.uniform(-0.1, 0.1, 7).astype(np.float32),
        "action.left_hand": np.random.uniform(0, 1, 6).astype(np.float32),
        "action.right_arm": np.random.uniform(-0.1, 0.1, 7).astype(np.float32),
        "action.right_hand": np.random.uniform(0, 1, 6).astype(np.float32),
        "action.waist": np.zeros(3, dtype=np.float32),
    }

    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step result: reward={reward}, terminated={terminated}, info={info}")

    env.close()
    print("Test complete!")
