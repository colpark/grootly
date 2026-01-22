"""
Trossen AI Mobile Robot Adapter for GR00T Policy

This module provides the adapter layer between:
    - Raw Trossen AI Mobile robot observations
    - GR00T VLA model input format
    - GR00T action chunks back to robot motor commands

Robot Configuration (Trossen AI Mobile):
    Cameras (3):
        - cam_high: Central overhead camera
        - cam_left_wrist: Left arm wrist-mounted camera
        - cam_right_wrist: Right arm wrist-mounted camera

    State (19 DOF):
        - base_odom: [odom_x, odom_y, odom_theta] (3)
        - base_vel: [linear_vel, angular_vel] (2)
        - left_arm: [waist, shoulder, elbow, forearm_roll, wrist_pitch, wrist_roll, gripper] (7)
        - right_arm: [waist, shoulder, elbow, forearm_roll, wrist_pitch, wrist_roll, gripper] (7)

    Action (16 DOF):
        - base_vel: [linear_vel, angular_vel] (2) - ABSOLUTE
        - left_arm: [7 joint commands] (7) - RELATIVE
        - right_arm: [7 joint commands] (7) - RELATIVE
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from gr00t.policy.server_client import PolicyClient


def recursive_add_extra_dim(obs: Dict) -> Dict:
    """
    Recursively add an extra dimension to arrays or scalars.

    GR00T Policy Server expects:
        obs: (batch=1, time=1, ...)
    Calling this function twice achieves (B=1, T=1, ...) format.
    """
    result = {}
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            result[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            result[key] = recursive_add_extra_dim(val)
        else:
            result[key] = [val]  # scalar -> [scalar]
    return result


class TrossenMobileAdapter:
    """
    Adapter between Trossen AI Mobile robot and GR00T VLA policy.

    Responsibilities:
        - Package camera frames as obs["video"]
        - Build obs["state"] for base + dual arms
        - Add language instruction
        - Add batch/time dimensions
        - Decode model action chunks into robot motor commands
    """

    # Camera names matching modality config
    CAMERA_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    # State component definitions (19 DOF total)
    STATE_COMPONENTS = {
        "base_odom": {"indices": (0, 3), "dof": 3},   # odom_x, odom_y, odom_theta
        "base_vel": {"indices": (3, 5), "dof": 2},    # linear_vel, angular_vel
        "left_arm": {"indices": (5, 12), "dof": 7},   # 7 joint positions
        "right_arm": {"indices": (12, 19), "dof": 7}, # 7 joint positions
    }

    # Action component definitions (16 DOF total)
    ACTION_COMPONENTS = {
        "base_vel": {"indices": (0, 2), "dof": 2},    # linear_vel, angular_vel
        "left_arm": {"indices": (2, 9), "dof": 7},    # 7 joint commands
        "right_arm": {"indices": (9, 16), "dof": 7},  # 7 joint commands
    }

    # Joint names for left and right arms (same structure)
    ARM_JOINT_NAMES = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_pitch",
        "wrist_roll",
        "gripper",
    ]

    def __init__(self, policy_client: PolicyClient):
        """
        Initialize the adapter with a policy client.

        Args:
            policy_client: Connected PolicyClient instance
        """
        self.policy = policy_client
        self._action_horizon = None  # Will be determined from first action

    # -------------------------------------------------------------------------
    # Observation -> Model Input
    # -------------------------------------------------------------------------

    def obs_to_policy_inputs(
        self,
        cameras: Dict[str, np.ndarray],
        state: np.ndarray,
        language: str,
    ) -> Dict:
        """
        Convert robot observation into GR00T VLA model input format.

        Args:
            cameras: Dict mapping camera name to image array
                     Expected keys: cam_high, cam_left_wrist, cam_right_wrist
                     Image shape: (H, W, 3) in BGR or RGB format
            state: Robot state vector, shape (19,)
                   [base_odom(3), base_vel(2), left_arm(7), right_arm(7)]
            language: Task instruction string

        Returns:
            Dict formatted for GR00T policy input with (B=1, T=1) dimensions
        """
        model_obs = {}

        # (1) Cameras - package all camera frames
        model_obs["video"] = {}
        for cam_key in self.CAMERA_KEYS:
            if cam_key in cameras:
                model_obs["video"][cam_key] = cameras[cam_key]
            else:
                raise ValueError(f"Missing camera: {cam_key}. Expected: {self.CAMERA_KEYS}")

        # (2) State - decompose into components
        state = np.asarray(state, dtype=np.float32)
        if state.shape[0] != 19:
            raise ValueError(f"Expected state dim 19, got {state.shape[0]}")

        model_obs["state"] = {}
        for component, info in self.STATE_COMPONENTS.items():
            start, end = info["indices"]
            model_obs["state"][component] = state[start:end]

        # (3) Language instruction
        model_obs["language"] = {"task": language}

        # (4) Add batch and time dimensions (B=1, T=1)
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)

        return model_obs

    # -------------------------------------------------------------------------
    # Model Action Chunk -> Robot Commands
    # -------------------------------------------------------------------------

    def decode_action_chunk(
        self,
        action_chunk: Dict[str, np.ndarray],
        timestep: int,
    ) -> Dict[str, np.ndarray]:
        """
        Decode a single timestep from the action chunk.

        Args:
            action_chunk: Dict from policy with shape (B, T, D) per component
                          Keys: base_vel, left_arm, right_arm
            timestep: Which timestep to extract (0 to horizon-1)

        Returns:
            Dict with decoded action for each component:
                - base_vel: np.ndarray shape (2,)
                - left_arm: np.ndarray shape (7,)
                - right_arm: np.ndarray shape (7,)
        """
        result = {}

        for component in self.ACTION_COMPONENTS.keys():
            if component not in action_chunk:
                raise ValueError(f"Missing action component: {component}")

            # Action shape is (B, T, D), extract (D,) for this timestep
            action_data = action_chunk[component]
            result[component] = action_data[0, timestep, :]  # (D,)

        return result

    def get_action(
        self,
        cameras: Dict[str, np.ndarray],
        state: np.ndarray,
        language: str,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Get action sequence from the policy.

        Args:
            cameras: Dict mapping camera name to image array
            state: Robot state vector, shape (19,)
            language: Task instruction string

        Returns:
            List of action dicts, one per timestep in the action horizon.
            Each dict contains:
                - base_vel: np.ndarray shape (2,) - velocity commands
                - left_arm: np.ndarray shape (7,) - joint commands
                - right_arm: np.ndarray shape (7,) - joint commands
        """
        # Convert observation to model input format
        model_input = self.obs_to_policy_inputs(cameras, state, language)

        # Query the policy server
        action_chunk, info = self.policy.get_action(model_input)

        # Determine action horizon from the output
        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]  # (B, T, D) -> T
        self._action_horizon = horizon

        # Decode all timesteps
        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]

    def action_to_flat_vector(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert structured action dict to flat 16-DOF vector.

        Args:
            action: Dict with base_vel, left_arm, right_arm arrays

        Returns:
            np.ndarray shape (16,) in order: [base_vel, left_arm, right_arm]
        """
        return np.concatenate([
            action["base_vel"],   # (2,)
            action["left_arm"],   # (7,)
            action["right_arm"],  # (7,)
        ], axis=0)

    def action_to_joint_dict(self, action: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Convert structured action to named joint dictionary.

        Useful for robot APIs that expect named joint commands.

        Args:
            action: Dict with base_vel, left_arm, right_arm arrays

        Returns:
            Dict mapping joint names to float values:
                - base_linear_vel, base_angular_vel
                - left_waist, left_shoulder, ..., left_gripper
                - right_waist, right_shoulder, ..., right_gripper
        """
        result = {}

        # Base velocity
        result["base_linear_vel"] = float(action["base_vel"][0])
        result["base_angular_vel"] = float(action["base_vel"][1])

        # Left arm joints
        for i, joint_name in enumerate(self.ARM_JOINT_NAMES):
            result[f"left_{joint_name}"] = float(action["left_arm"][i])

        # Right arm joints
        for i, joint_name in enumerate(self.ARM_JOINT_NAMES):
            result[f"right_{joint_name}"] = float(action["right_arm"][i])

        return result

    @property
    def action_horizon(self) -> int:
        """Return the action horizon (number of timesteps per action chunk)."""
        if self._action_horizon is None:
            return 16  # Default value
        return self._action_horizon


class TrossenMobileObservationBuilder:
    """
    Helper class to build observations from raw sensor data.

    This class helps construct the observation dict from various
    sensor sources on the Trossen AI Mobile robot.
    """

    def __init__(self):
        self.cameras: Dict[str, np.ndarray] = {}
        self.state: np.ndarray = np.zeros(19, dtype=np.float32)
        self.language: str = ""

    def set_camera(self, name: str, image: np.ndarray) -> "TrossenMobileObservationBuilder":
        """
        Set a camera image.

        Args:
            name: Camera name (cam_high, cam_left_wrist, cam_right_wrist)
            image: Image array, shape (H, W, 3)
        """
        self.cameras[name] = np.asarray(image, dtype=np.uint8)
        return self

    def set_base_odom(self, odom_x: float, odom_y: float, odom_theta: float) -> "TrossenMobileObservationBuilder":
        """Set base odometry (position and heading)."""
        self.state[0] = odom_x
        self.state[1] = odom_y
        self.state[2] = odom_theta
        return self

    def set_base_vel(self, linear_vel: float, angular_vel: float) -> "TrossenMobileObservationBuilder":
        """Set base velocity."""
        self.state[3] = linear_vel
        self.state[4] = angular_vel
        return self

    def set_left_arm(self, joints: List[float]) -> "TrossenMobileObservationBuilder":
        """
        Set left arm joint positions.

        Args:
            joints: List of 7 joint values in order:
                    [waist, shoulder, elbow, forearm_roll, wrist_pitch, wrist_roll, gripper]
        """
        if len(joints) != 7:
            raise ValueError(f"Expected 7 joints for left arm, got {len(joints)}")
        self.state[5:12] = joints
        return self

    def set_right_arm(self, joints: List[float]) -> "TrossenMobileObservationBuilder":
        """
        Set right arm joint positions.

        Args:
            joints: List of 7 joint values in order:
                    [waist, shoulder, elbow, forearm_roll, wrist_pitch, wrist_roll, gripper]
        """
        if len(joints) != 7:
            raise ValueError(f"Expected 7 joints for right arm, got {len(joints)}")
        self.state[12:19] = joints
        return self

    def set_state_vector(self, state: np.ndarray) -> "TrossenMobileObservationBuilder":
        """
        Set full state vector directly.

        Args:
            state: Full 19-DOF state vector
        """
        if len(state) != 19:
            raise ValueError(f"Expected 19-DOF state, got {len(state)}")
        self.state = np.asarray(state, dtype=np.float32)
        return self

    def set_language(self, instruction: str) -> "TrossenMobileObservationBuilder":
        """Set the language instruction."""
        self.language = instruction
        return self

    def build(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, str]:
        """
        Build the observation tuple.

        Returns:
            Tuple of (cameras, state, language) ready for TrossenMobileAdapter
        """
        # Validate all cameras are set
        required_cameras = TrossenMobileAdapter.CAMERA_KEYS
        for cam in required_cameras:
            if cam not in self.cameras:
                raise ValueError(f"Missing camera: {cam}")

        return self.cameras.copy(), self.state.copy(), self.language
