"""
Trossen AI Mobile modality configuration for finetuning.

This registers the Trossen AI Mobile embodiment config for finetuning
on the ball2_groot dataset with 3-camera setup.

Robot Configuration:
- 3 cameras: cam_high, cam_left_wrist, cam_right_wrist
- State: 19 DOF (base_odom[3], base_vel[2], left_arm[7], right_arm[7])
- Action: 16 DOF (base_vel[2], left_arm[7], right_arm[7])
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# Trossen AI Mobile modality configuration
trossen_mobile_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "cam_high",        # Central overhead camera
            "cam_left_wrist",  # Left arm wrist camera
            "cam_right_wrist", # Right arm wrist camera
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "base_odom",   # 3 DOF: odom_x, odom_y, odom_theta
            "base_vel",    # 2 DOF: linear_vel, angular_vel
            "left_arm",    # 7 DOF: joint positions
            "right_arm",   # 7 DOF: joint positions
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),  # 16-step action horizon
        modality_keys=[
            "base_vel",    # 2 DOF: linear_vel, angular_vel
            "left_arm",    # 7 DOF: joint commands
            "right_arm",   # 7 DOF: joint commands
        ],
        action_configs=[
            # base_vel - velocity commands are absolute
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_arm - relative joint positions
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_arm - relative joint positions
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}

# Register config with NEW_EMBODIMENT tag (required for custom robots)
register_modality_config(trossen_mobile_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

print("Registered Trossen AI Mobile modality config for finetuning (as NEW_EMBODIMENT)")
