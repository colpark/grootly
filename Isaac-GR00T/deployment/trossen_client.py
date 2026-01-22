#!/usr/bin/env python3
"""
Trossen AI Mobile Robot Client for GR00T Policy Server

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
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

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

    # Debug
    verbose: bool = False
    dry_run: bool = False  # If True, don't send commands to robot


class TrossenRobotInterface:
    """
    Interface to the physical Trossen AI Mobile robot.

    This is a placeholder implementation. Replace with actual
    Trossen robot SDK calls for your specific setup.
    """

    def __init__(self, config: ClientConfig):
        self.config = config
        self.connected = False

    def connect(self) -> bool:
        """
        Connect to the robot hardware.

        Replace this with your actual robot connection code.
        For example, using ROS, LeRobot, or Trossen's SDK.
        """
        logger.info("Connecting to Trossen AI Mobile robot...")

        # TODO: Replace with actual robot connection
        # Example for LeRobot:
        # from lerobot.robots import make_robot_from_config
        # self.robot = make_robot_from_config(your_config)
        # self.robot.connect()

        self.connected = True
        logger.info("Robot connected successfully")
        return True

    def disconnect(self):
        """Disconnect from the robot."""
        if self.connected:
            logger.info("Disconnecting from robot...")
            # TODO: Add actual disconnect code
            self.connected = False

    def get_observation(self) -> Dict:
        """
        Get current observation from robot sensors.

        Returns:
            Dict containing:
                - cameras: Dict[str, np.ndarray] - camera images
                - state: np.ndarray - 19-DOF state vector
        """
        # TODO: Replace with actual sensor reading code
        # Example structure:
        # obs = self.robot.get_observation()
        # cameras = {
        #     "cam_high": obs["cam_high"],
        #     "cam_left_wrist": obs["cam_left_wrist"],
        #     "cam_right_wrist": obs["cam_right_wrist"],
        # }
        # state = obs["state"]  # 19-DOF vector

        # Placeholder - replace with actual implementation
        cameras = {
            "cam_high": np.zeros((256, 256, 3), dtype=np.uint8),
            "cam_left_wrist": np.zeros((256, 256, 3), dtype=np.uint8),
            "cam_right_wrist": np.zeros((256, 256, 3), dtype=np.uint8),
        }
        state = np.zeros(19, dtype=np.float32)

        return {"cameras": cameras, "state": state}

    def send_action(self, action: Dict[str, np.ndarray]):
        """
        Send action command to robot actuators.

        Args:
            action: Dict with:
                - base_vel: np.ndarray (2,) - [linear_vel, angular_vel]
                - left_arm: np.ndarray (7,) - joint commands
                - right_arm: np.ndarray (7,) - joint commands
        """
        # Apply safety limits
        base_vel = action["base_vel"].copy()
        base_vel[0] = np.clip(base_vel[0], -self.config.max_base_linear_vel, self.config.max_base_linear_vel)
        base_vel[1] = np.clip(base_vel[1], -self.config.max_base_angular_vel, self.config.max_base_angular_vel)

        if self.config.verbose:
            logger.debug(f"Sending action - base_vel: {base_vel}")

        if self.config.dry_run:
            return

        # TODO: Replace with actual robot command code
        # Example:
        # self.robot.send_action({
        #     "base_linear_vel": float(base_vel[0]),
        #     "base_angular_vel": float(base_vel[1]),
        #     "left_arm_joints": action["left_arm"].tolist(),
        #     "right_arm_joints": action["right_arm"].tolist(),
        # })
        pass


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
        "--test-only",
        action="store_true",
        help="Only test server connection, then exit",
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
    )

    if config.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print configuration
    logger.info("=" * 50)
    logger.info("Trossen AI Mobile - GR00T Policy Client")
    logger.info("=" * 50)
    logger.info(f"Server: tcp://{config.server_ip}:{config.server_port}")
    logger.info(f"Task: {config.task_instruction}")
    logger.info(f"Dry run: {config.dry_run}")
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
    if not config.dry_run:
        if not robot.connect():
            logger.error("Failed to connect to robot")
            sys.exit(1)

    try:
        # Run the control loop
        run_control_loop(policy_client, adapter, robot, config)
    finally:
        # Cleanup
        robot.disconnect()
        logger.info("Client shutdown complete")


if __name__ == "__main__":
    main()
