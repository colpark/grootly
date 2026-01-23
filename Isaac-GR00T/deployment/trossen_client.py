#!/usr/bin/env python3
"""
Trossen AI Mobile Robot Client for GR00T Policy Server

VERSION: 2.3 - Very conservative joint limits to debug LeRobot scaling

This script runs on the robot's computer and connects to a remote
GR00T inference server to get action predictions in real-time.

Architecture:
    ┌─────────────────────┐                    ┌─────────────────────┐
    │   Robot Computer    │                    │    GPU Server       │
    │                     │                    │                     │
    │  ┌───────────────┐  │   ZeroMQ (TCP)    │  ┌───────────────┐  │
    │  │ trossen_client│──┼───────────────────┼──│ PolicyServer  │  │
    │  └───────────────┘  │   Port 5559       │  │ (GR00T Model) │  │
    │         │           │                    │  └───────────────┘  │
    │         ▼           │                    │                     │
    │  ┌───────────────┐  │                    └─────────────────────┘
    │  │ Trossen Robot │  │
    │  │  Hardware     │  │
    │  └───────────────┘  │
    └─────────────────────┘

Usage:
    python deployment/trossen_client.py --server-ip 130.199.95.27 --server-port 5559

    # With custom task instruction:
    python deployment/trossen_client.py \
        --server-ip 130.199.95.27 \
        --server-port 5559 \
        --task "Stack the lego blocks"

Configuration:
    Server IP: 130.199.95.27
    Server Port: 5559
    Action Horizon: 16 steps (configurable)
    Control Frequency: ~30 Hz

Requirements (on robot computer):
    - lerobot from Interbotix/lerobot (trossen-ai branch)
    - pyrealsense2 for Intel RealSense cameras
    - zmq for server communication
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

# Add project root to path for imports
sys.path.insert(0, ".")

from gr00t.policy.server_client import PolicyClient
from deployment.trossen_adapter import TrossenMobileAdapter, TrossenMobileObservationBuilder


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# LeRobot Integration - Import with fallback for development
# =============================================================================
LEROBOT_AVAILABLE = False
try:
    from lerobot.common.robot_devices.robots.utils import make_robot, make_robot_from_config
    from lerobot.common.robot_devices.robots.configs import TrossenAIMobileRobotConfig
    LEROBOT_AVAILABLE = True
    logger.info("LeRobot integration available")
except ImportError as e:
    logger.warning(f"LeRobot not available: {e}")
    logger.warning("Running in mock mode - install lerobot for real robot control")


@dataclass
class ClientConfig:
    """Configuration for the Trossen robot client."""

    # Server connection
    server_ip: str = "130.199.95.27"
    server_port: int = 5559
    timeout_ms: int = 15000

    # Task configuration
    task_instruction: str = "Stack the lego blocks"

    # Control parameters
    action_horizon: int = 8  # How many steps to execute per inference
    control_frequency: float = 30.0  # Hz

    # Safety limits
    max_base_linear_vel: float = 0.5  # m/s
    max_base_angular_vel: float = 1.0  # rad/s

    # Robot hardware configuration
    left_arm_ip: str = "192.168.1.5"
    right_arm_ip: str = "192.168.1.4"

    # Camera serial numbers (Intel RealSense)
    # Update these with your actual camera serial numbers
    cam_high_serial: str = "130322274102"
    cam_left_wrist_serial: str = "130322271087"
    cam_right_wrist_serial: str = "130322270184"

    # Camera resolution
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # Hardware options
    disable_leader_arms: bool = True   # Disable leader arms (not needed for policy)
    disable_base: bool = False         # Disable mobile base
    arms_only: bool = False            # Only connect follower arms (no base, no leader)

    # Debug
    verbose: bool = False
    dry_run: bool = False  # If True, don't send commands to robot
    mock_robot: bool = False  # If True, use mock robot (no hardware)


class TrossenRobotInterface:
    """
    Interface to the physical Trossen AI Mobile robot using LeRobot.

    This implementation uses the LeRobot library from Interbotix/lerobot
    (trossen-ai branch) to communicate with the Trossen AI Mobile robot.

    State Vector (19 DOF):
        [0:3]   base_odom: odom_x, odom_y, odom_theta
        [3:5]   base_vel: linear_vel, angular_vel
        [5:12]  left_arm: 7 joint positions
        [12:19] right_arm: 7 joint positions

    Action Vector (16 DOF):
        [0:2]   base_vel: linear_vel, angular_vel
        [2:9]   left_arm: 7 joint commands
        [9:16]  right_arm: 7 joint commands
    """

    # Joint feature name patterns for Trossen AI Mobile
    # These map to the keys returned by robot.get_observation()
    LEFT_ARM_JOINTS = [
        "left_waist.pos",
        "left_shoulder.pos",
        "left_elbow.pos",
        "left_forearm_roll.pos",
        "left_wrist_pitch.pos",
        "left_wrist_roll.pos",
        "left_gripper.pos",
    ]

    RIGHT_ARM_JOINTS = [
        "right_waist.pos",
        "right_shoulder.pos",
        "right_elbow.pos",
        "right_forearm_roll.pos",
        "right_wrist_pitch.pos",
        "right_wrist_roll.pos",
        "right_gripper.pos",
    ]

    # Action joint names (without .pos suffix)
    LEFT_ARM_ACTION_KEYS = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_pitch",
        "left_wrist_roll",
        "left_gripper",
    ]

    RIGHT_ARM_ACTION_KEYS = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_pitch",
        "right_wrist_roll",
        "right_gripper",
    ]

    CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    def __init__(self, config: ClientConfig):
        self.config = config
        self.connected = False
        self.robot = None

        # For mock mode or when LeRobot is not available
        self._use_mock = config.mock_robot or not LEROBOT_AVAILABLE

        # Track current arm positions for RELATIVE action conversion
        # GR00T model outputs RELATIVE (delta) actions for arms
        # We need to convert: absolute_position = current_position + delta
        self._current_left_arm = np.zeros(7, dtype=np.float32)
        self._current_right_arm = np.zeros(7, dtype=np.float32)
        self._arm_positions_initialized = False

    def _create_robot_config(self):
        """Create LeRobot configuration for Trossen AI Mobile."""
        if not LEROBOT_AVAILABLE:
            raise RuntimeError("LeRobot is not available. Install it first.")

        # Create config with mock mode if needed
        robot_config = TrossenAIMobileRobotConfig(mock=self._use_mock)

        # Update follower arm IPs if different from defaults
        if self.config.left_arm_ip != "192.168.1.5":
            robot_config.follower_arms["left"].ip = self.config.left_arm_ip
        if self.config.right_arm_ip != "192.168.1.4":
            robot_config.follower_arms["right"].ip = self.config.right_arm_ip

        # Update camera serial numbers if different from defaults
        if self.config.cam_high_serial != "130322274102":
            robot_config.cameras["cam_high"].serial_number = int(self.config.cam_high_serial)
        if self.config.cam_left_wrist_serial != "130322271087":
            robot_config.cameras["cam_left_wrist"].serial_number = int(self.config.cam_left_wrist_serial)
        if self.config.cam_right_wrist_serial != "130322270184":
            robot_config.cameras["cam_right_wrist"].serial_number = int(self.config.cam_right_wrist_serial)

        # Disable leader arms if requested (not needed for policy deployment)
        if self.config.disable_leader_arms or self.config.arms_only:
            robot_config.leader_arms = {}
            logger.info("Leader arms disabled")

        return robot_config

    def connect(self) -> bool:
        """Connect to the robot hardware."""
        logger.info("Connecting to Trossen AI Mobile robot...")

        if self._use_mock:
            logger.warning("Running in MOCK mode - no real robot connection")
            self.connected = True
            return True

        try:
            robot_config = self._create_robot_config()
            self.robot = make_robot_from_config(robot_config)
            self.robot.connect()
            self.connected = True
            logger.info("Robot connected successfully")
            logger.info(f"  Left arm IP: {self.config.left_arm_ip}")
            logger.info(f"  Right arm IP: {self.config.right_arm_ip}")
            logger.info(f"  Cameras: {self.CAMERA_NAMES}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            return False

    def disconnect(self):
        """Disconnect from the robot."""
        if self.connected:
            logger.info("Disconnecting from robot...")
            if self.robot is not None:
                try:
                    self.robot.disconnect()
                except Exception as e:
                    logger.warning(f"Error during disconnect: {e}")
            self.connected = False
            self.robot = None

    def get_observation(self) -> Dict:
        """
        Get current observation from robot sensors.

        Returns:
            Dict containing:
                - cameras: Dict[str, np.ndarray] - camera images (H, W, 3) RGB
                - state: np.ndarray - 19-DOF state vector

        LeRobot TrossenAIMobile observation format:
            - observation.state: torch.Tensor [left_arm(7), right_arm(7), base_state(5)]
              where base_state = [odom_x, odom_y, odom_theta, linear_vel, angular_vel]
            - observation.images.{cam_name}: torch.Tensor (H, W, 3)

        GR00T expected state format (19-DOF):
            [base_odom(3), base_vel(2), left_arm(7), right_arm(7)]
        """
        if self._use_mock:
            return self._get_mock_observation()

        # Get raw observation from LeRobot
        obs = self.robot.capture_observation()

        # Extract camera images
        cameras = {}
        for cam_name in self.CAMERA_NAMES:
            obs_key = f"observation.images.{cam_name}"
            if obs_key in obs:
                image = obs[obs_key]
                # Convert torch tensor to numpy if needed
                if hasattr(image, 'numpy'):
                    image = image.numpy()
                # Ensure uint8 format
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)
                # Convert BGR to RGB if needed (LeRobot may return BGR)
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cameras[cam_name] = image
            else:
                logger.warning(f"Camera {cam_name} not found in observation (key: {obs_key})")
                cameras[cam_name] = np.zeros((self.config.camera_height, self.config.camera_width, 3), dtype=np.uint8)

        # Parse LeRobot state: [left_arm(7), right_arm(7), base_state(5)]
        raw_state = obs.get("observation.state")
        if hasattr(raw_state, 'numpy'):
            raw_state = raw_state.numpy()

        # Remap to GR00T format: [base_odom(3), base_vel(2), left_arm(7), right_arm(7)]
        state = np.zeros(19, dtype=np.float32)

        if raw_state is not None and len(raw_state) >= 19:
            # LeRobot format: [left_arm(7), right_arm(7), odom_x, odom_y, odom_theta, linear_vel, angular_vel]
            left_arm = raw_state[0:7]      # indices 0-6
            right_arm = raw_state[7:14]    # indices 7-13
            base_state = raw_state[14:19]  # indices 14-18: [odom_x, odom_y, odom_theta, linear_vel, angular_vel]

            # GR00T format: [base_odom(3), base_vel(2), left_arm(7), right_arm(7)]
            state[0:3] = base_state[0:3]   # odom_x, odom_y, odom_theta
            state[3:5] = base_state[3:5]   # linear_vel, angular_vel
            state[5:12] = left_arm         # left arm joints
            state[12:19] = right_arm       # right arm joints

            # Store current arm positions for relative action conversion
            # This is critical: GR00T outputs RELATIVE (delta) actions for arms
            self._current_left_arm = left_arm.astype(np.float32).copy()
            self._current_right_arm = right_arm.astype(np.float32).copy()
            self._arm_positions_initialized = True

        if self.config.verbose:
            logger.debug(f"State: base_odom={state[0:3]}, base_vel={state[3:5]}")
            logger.debug(f"       left_arm={state[5:12]}, right_arm={state[12:19]}")

        return {"cameras": cameras, "state": state}

    def _get_mock_observation(self) -> Dict:
        """Return mock observation for testing without hardware."""
        cameras = {
            "cam_high": np.zeros((self.config.camera_height, self.config.camera_width, 3), dtype=np.uint8),
            "cam_left_wrist": np.zeros((self.config.camera_height, self.config.camera_width, 3), dtype=np.uint8),
            "cam_right_wrist": np.zeros((self.config.camera_height, self.config.camera_width, 3), dtype=np.uint8),
        }
        state = np.zeros(19, dtype=np.float32)
        return {"cameras": cameras, "state": state}

    # Motor position limits (radians) for Trossen AI arms
    # These are conservative limits to prevent out-of-bounds errors
    JOINT_LIMITS = {
        # [min, max] in radians for each joint
        # Joints 0-5: arm joints, Joint 6: gripper
        "default": [-3.14, 3.14],  # ~180 degrees each way
        "gripper": [0.0, 1.0],     # gripper range
    }
    # Motor limit is [-12.5, 12.5] but there seems to be internal scaling in LeRobot
    # The gripper especially seems to have ~10x scaling (1.4 rad -> 14.68 motor units)
    # Use very conservative limit until we understand the scaling
    MAX_JOINT_POSITION = 1.0  # VERY conservative - test with small movements first
    MAX_GRIPPER_POSITION = 0.1  # Gripper seems to have different scaling

    def send_action(self, action: Dict[str, np.ndarray]):
        """
        Send action command to robot actuators.

        Args:
            action: Dict with:
                - base_vel: np.ndarray (2,) - [linear_vel, angular_vel] (ABSOLUTE)
                - left_arm: np.ndarray (7,) - joint position commands (ABSOLUTE)
                - right_arm: np.ndarray (7,) - joint position commands (ABSOLUTE)

        NOTE: Despite modality config indicating RELATIVE actions, empirical testing
        shows the model outputs ABSOLUTE positions. We apply safety clamping and
        send directly to the robot.

        GR00T action format: [base_vel(2), left_arm(7), right_arm(7)]
        LeRobot action format: [left_arm(7), right_arm(7), linear_vel, angular_vel]
        """
        # Apply safety limits to base velocity (ABSOLUTE values)
        base_vel = action["base_vel"].copy()
        base_vel[0] = np.clip(base_vel[0], -self.config.max_base_linear_vel, self.config.max_base_linear_vel)
        base_vel[1] = np.clip(base_vel[1], -self.config.max_base_angular_vel, self.config.max_base_angular_vel)

        # Get arm action values from model
        # NOTE: Despite modality config saying RELATIVE, the model appears to output
        # ABSOLUTE positions (values similar to current positions, not small deltas).
        # We treat them as ABSOLUTE to avoid position accumulation/explosion.
        left_arm = action["left_arm"].copy()
        right_arm = action["right_arm"].copy()

        # Apply safety limits to ABSOLUTE joint positions
        # Clamp arm joints (indices 0-5) and gripper (index 6) separately
        left_arm_clamped = left_arm.copy()
        right_arm_clamped = right_arm.copy()

        # Clamp arm joints (0-5)
        left_arm_clamped[:6] = np.clip(left_arm[:6], -self.MAX_JOINT_POSITION, self.MAX_JOINT_POSITION)
        right_arm_clamped[:6] = np.clip(right_arm[:6], -self.MAX_JOINT_POSITION, self.MAX_JOINT_POSITION)

        # Clamp gripper (index 6) with different limits - gripper seems to have ~10x scaling
        left_arm_clamped[6] = np.clip(left_arm[6], 0, self.MAX_GRIPPER_POSITION)
        right_arm_clamped[6] = np.clip(right_arm[6], 0, self.MAX_GRIPPER_POSITION)

        # Log if any values were clamped
        if np.any(left_arm != left_arm_clamped) or np.any(right_arm != right_arm_clamped):
            if np.any(left_arm != left_arm_clamped):
                logger.warning(f"Left arm clamped: {left_arm} -> {left_arm_clamped}")
            if np.any(right_arm != right_arm_clamped):
                logger.warning(f"Right arm clamped: {right_arm} -> {right_arm_clamped}")

        left_arm = left_arm_clamped
        right_arm = right_arm_clamped

        # Update tracked positions (will be synced with actual robot state on next observation)
        self._current_left_arm = left_arm.copy()
        self._current_right_arm = right_arm.copy()

        # ALWAYS log first few actions to help debug issues
        if not hasattr(self, '_action_count'):
            self._action_count = 0
        self._action_count += 1

        if self._action_count <= 5 or self.config.verbose:
            logger.info(f"Action #{self._action_count}:")
            logger.info(f"  robot_left:    {self._current_left_arm}")
            logger.info(f"  robot_right:   {self._current_right_arm}")
            logger.info(f"  model_left:    {action['left_arm']}")
            logger.info(f"  model_right:   {action['right_arm']}")
            logger.info(f"  target_left:   {left_arm}")
            logger.info(f"  target_right:  {right_arm}")
            logger.info(f"  base_vel:      {base_vel}")

        if self.config.dry_run or self._use_mock:
            return

        # Convert GR00T format to LeRobot format
        # GR00T: [base_vel(2), left_arm(7), right_arm(7)]
        # LeRobot: [left_arm(7), right_arm(7), linear_vel, angular_vel]
        import torch

        # If base is disabled or arms_only mode, zero out base velocity
        if self.config.disable_base or self.config.arms_only:
            base_vel = np.zeros(2, dtype=np.float32)

        # FINAL SAFETY CHECK - clamp again right before sending
        # This catches any bugs in the logic above
        HARD_LIMIT_JOINTS = 1.0  # Very conservative for arm joints
        HARD_LIMIT_GRIPPER = 0.1  # Even more conservative for gripper (has ~10x scaling)

        # Clamp joints and gripper separately
        left_arm[:6] = np.clip(left_arm[:6], -HARD_LIMIT_JOINTS, HARD_LIMIT_JOINTS)
        right_arm[:6] = np.clip(right_arm[:6], -HARD_LIMIT_JOINTS, HARD_LIMIT_JOINTS)
        left_arm[6] = np.clip(left_arm[6], 0, HARD_LIMIT_GRIPPER)
        right_arm[6] = np.clip(right_arm[6], 0, HARD_LIMIT_GRIPPER)

        # Log what we're sending after all clamping
        logger.info(f"After clamping: left={left_arm}, right={right_arm}")

        lerobot_action = torch.tensor(
            np.concatenate([
                left_arm,    # 7 joints (clamped)
                right_arm,   # 7 joints (clamped)
                base_vel,    # [linear_vel, angular_vel]
            ]),
            dtype=torch.float32
        )

        # Log what we're actually sending
        if self._action_count <= 3:
            logger.info(f"Sending to robot: {lerobot_action.numpy()}")

        # Send to robot
        try:
            self.robot.send_action(lerobot_action)
        except Exception as e:
            logger.error(f"Failed to send action: {e}")


def run_control_loop(
    policy_client: PolicyClient,
    adapter: TrossenMobileAdapter,
    robot: TrossenRobotInterface,
    config: ClientConfig,
):
    """
    Main real-time control loop.

    This loop:
    1. Reads observation from robot
    2. Sends observation to policy server
    3. Receives action chunk from server
    4. Executes actions on robot
    """
    logger.info(f"Starting control loop with task: '{config.task_instruction}'")
    logger.info(f"Action horizon: {config.action_horizon} steps")
    logger.info(f"Control frequency: {config.control_frequency} Hz")
    logger.info("Press Ctrl+C to stop")

    step_count = 0
    inference_times = []

    try:
        while True:
            loop_start = time.time()

            # 1. Get observation from robot
            obs = robot.get_observation()

            # 2. Query policy server for actions
            inference_start = time.time()
            try:
                actions = adapter.get_action(
                    cameras=obs["cameras"],
                    state=obs["state"],
                    language=config.task_instruction,
                )
            except Exception as e:
                logger.error(f"Policy server error: {e}")
                logger.info("Retrying connection...")
                time.sleep(1.0)
                continue

            inference_time = time.time() - inference_start
            inference_times.append(inference_time)

            # 3. Execute action chunk on robot
            for i, action in enumerate(actions[: config.action_horizon]):
                action_start = time.time()

                if config.verbose:
                    logger.debug(f"Action[{i}]: base_vel={action['base_vel']}")

                robot.send_action(action)
                step_count += 1

                # Maintain control frequency
                action_elapsed = time.time() - action_start
                sleep_time = (1.0 / config.control_frequency) - action_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Log statistics periodically
            if step_count % 100 == 0:
                avg_inference = np.mean(inference_times[-100:]) * 1000
                logger.info(f"Step {step_count}, Avg inference: {avg_inference:.1f}ms")

    except KeyboardInterrupt:
        logger.info("\nControl loop stopped by user")
    finally:
        # Print final statistics
        if inference_times:
            logger.info(f"Total steps: {step_count}")
            logger.info(f"Avg inference time: {np.mean(inference_times)*1000:.1f}ms")
            logger.info(f"Max inference time: {np.max(inference_times)*1000:.1f}ms")


def test_connection(client: PolicyClient) -> bool:
    """Test connection to the policy server."""
    logger.info("Testing connection to policy server...")

    try:
        if client.ping():
            logger.info("Server connection: OK")
            return True
        else:
            logger.error("Server connection: FAILED (no response)")
            return False
    except Exception as e:
        logger.error(f"Server connection: FAILED ({e})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Trossen AI Mobile Robot Client for GR00T Policy Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python deployment/trossen_client.py

  # Specify server and task
  python deployment/trossen_client.py \\
      --server-ip 130.199.95.27 \\
      --server-port 5559 \\
      --task "Stack the lego blocks"

  # Dry run mode (no robot commands)
  python deployment/trossen_client.py --dry-run --verbose

  # Test connection only
  python deployment/trossen_client.py --test-only
        """,
    )

    # Server connection
    parser.add_argument(
        "--server-ip",
        type=str,
        default="130.199.95.27",
        help="IP address of the GR00T policy server (default: 130.199.95.27)",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=5559,
        help="Port of the GR00T policy server (default: 5559)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15000,
        help="Connection timeout in milliseconds (default: 15000)",
    )

    # Task configuration
    parser.add_argument(
        "--task",
        type=str,
        default="Stack the lego blocks",
        help="Task instruction for the robot",
    )

    # Control parameters
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=8,
        help="Number of action steps to execute per inference (default: 8)",
    )
    parser.add_argument(
        "--control-freq",
        type=float,
        default=30.0,
        help="Control loop frequency in Hz (default: 30.0)",
    )

    # Debug options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending commands to robot",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock robot (no hardware required)",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test server connection, then exit",
    )

    # Robot hardware configuration
    parser.add_argument(
        "--left-arm-ip",
        type=str,
        default="192.168.1.5",
        help="IP address of the left arm (default: 192.168.1.5)",
    )
    parser.add_argument(
        "--right-arm-ip",
        type=str,
        default="192.168.1.4",
        help="IP address of the right arm (default: 192.168.1.4)",
    )
    parser.add_argument(
        "--cam-high-serial",
        type=str,
        default="130322274102",
        help="Serial number of the high camera",
    )
    parser.add_argument(
        "--cam-left-wrist-serial",
        type=str,
        default="130322271087",
        help="Serial number of the left wrist camera",
    )
    parser.add_argument(
        "--cam-right-wrist-serial",
        type=str,
        default="130322270184",
        help="Serial number of the right wrist camera",
    )

    # Hardware options
    parser.add_argument(
        "--arms-only",
        action="store_true",
        help="Only connect follower arms (no leader arms, no base)",
    )
    parser.add_argument(
        "--enable-leader-arms",
        action="store_true",
        help="Enable leader arms (disabled by default for policy deployment)",
    )
    parser.add_argument(
        "--disable-base",
        action="store_true",
        help="Disable mobile base",
    )

    args = parser.parse_args()

    # Create configuration
    config = ClientConfig(
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout_ms=args.timeout,
        task_instruction=args.task,
        action_horizon=args.action_horizon,
        control_frequency=args.control_freq,
        verbose=args.verbose,
        dry_run=args.dry_run,
        mock_robot=args.mock,
        left_arm_ip=args.left_arm_ip,
        right_arm_ip=args.right_arm_ip,
        cam_high_serial=args.cam_high_serial,
        cam_left_wrist_serial=args.cam_left_wrist_serial,
        cam_right_wrist_serial=args.cam_right_wrist_serial,
        arms_only=args.arms_only,
        disable_leader_arms=not args.enable_leader_arms,  # Disabled by default
        disable_base=args.disable_base,
    )

    if config.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print configuration
    logger.info("=" * 50)
    logger.info("Trossen AI Mobile - GR00T Policy Client")
    logger.info("VERSION: 2.3 - Very conservative limits (joints ±1.0, gripper 0-0.1)")
    logger.info("=" * 50)
    logger.info(f"Server: tcp://{config.server_ip}:{config.server_port}")
    logger.info(f"Task: {config.task_instruction}")
    logger.info(f"Dry run: {config.dry_run}")
    logger.info(f"Mock mode: {config.mock_robot}")
    if not config.mock_robot:
        logger.info(f"Left arm IP: {config.left_arm_ip}")
        logger.info(f"Right arm IP: {config.right_arm_ip}")
        logger.info(f"Cameras: cam_high={config.cam_high_serial}, "
                    f"cam_left_wrist={config.cam_left_wrist_serial}, "
                    f"cam_right_wrist={config.cam_right_wrist_serial}")
    logger.info("=" * 50)

    # Create policy client
    logger.info(f"Connecting to server at {config.server_ip}:{config.server_port}...")
    policy_client = PolicyClient(
        host=config.server_ip,
        port=config.server_port,
        timeout_ms=config.timeout_ms,
    )

    # Test connection
    if not test_connection(policy_client):
        logger.error("Failed to connect to policy server")
        logger.error(f"Make sure the server is running at {config.server_ip}:{config.server_port}")
        sys.exit(1)

    if args.test_only:
        logger.info("Connection test successful. Exiting.")
        sys.exit(0)

    # Create adapter and robot interface
    adapter = TrossenMobileAdapter(policy_client)
    robot = TrossenRobotInterface(config)

    # Connect to robot
    if not config.dry_run and not config.mock_robot:
        if not robot.connect():
            logger.error("Failed to connect to robot")
            sys.exit(1)
    elif config.mock_robot:
        logger.info("Running in MOCK mode - no robot hardware required")
        robot.connect()  # Sets connected=True for mock

    try:
        # Run the control loop
        run_control_loop(policy_client, adapter, robot, config)
    finally:
        # Cleanup
        robot.disconnect()
        logger.info("Client shutdown complete")


if __name__ == "__main__":
    main()
