"""
Modality configuration for ALOHA Transfer Cube dataset mapped to GR1-compatible format.

This config registers the ALOHA embodiment with GR00T's training system.
The ALOHA 14-DOF bimanual robot is mapped to GR1's 44-DOF format with appropriate
dimension mapping and action representations.

Usage:
    python -m gr00t.experiment.launch_finetune \
        --modality-config-path ./finetuning/aloha_modality_config.py \
        --embodiment-tag NEW_EMBODIMENT \
        ...
"""

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.configs.data.embodiment_configs import register_modality_config


# ALOHA Transfer Cube modality configuration
# Mapped to GR1-compatible 44-DOF format
ALOHA_TRANSFER_CUBE_CONFIG = {
    # Video configuration
    # Original: observation.images.top (480x640)
    # Converted: ego_view (256x256)
    "video": ModalityConfig(
        delta_indices=[0],  # Current frame only
        modality_keys=["ego_view"],
    ),

    # State configuration
    # Original ALOHA: 14 DOF (7 left + 7 right, including grippers)
    # Mapped to GR1: 44 DOF with zeros for unused joints
    "state": ModalityConfig(
        delta_indices=[0],  # Current state only
        modality_keys=[
            "left_arm",    # 7 DOF: ALOHA left arm [0:6] + 1 padded zero
            "left_hand",   # 6 DOF: ALOHA left gripper [6] expanded
            "right_arm",   # 7 DOF: ALOHA right arm [7:13] + 1 padded zero
            "right_hand",  # 6 DOF: ALOHA right gripper [13] expanded
            "waist",       # 3 DOF: zeros (unused in ALOHA)
        ],
        # Note: left_leg, right_leg, neck are not included as they're zeros
    ),

    # Action configuration
    # Same mapping as state, with action representations defined
    "action": ModalityConfig(
        delta_indices=list(range(16)),  # Predict 16 future action steps
        modality_keys=[
            "left_arm",
            "left_hand",
            "right_arm",
            "right_hand",
            "waist",
        ],
        action_configs=[
            # left_arm: 7 DOF joint control
            # ALOHA uses joint position control, mapped to relative actions
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_hand: 6 DOF expanded from gripper
            # Gripper uses absolute control (open/close)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_arm: 7 DOF joint control
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_hand: 6 DOF expanded from gripper
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # waist: 3 DOF (zeros, unused but required for GR1 compatibility)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),

    # Language configuration
    # Task description for the bimanual handover task
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}


# Register the configuration with GR00T's system
# This makes the config available when --embodiment-tag NEW_EMBODIMENT is used
register_modality_config(ALOHA_TRANSFER_CUBE_CONFIG, EmbodimentTag.NEW_EMBODIMENT)

print("=" * 60)
print("Registered ALOHA Transfer Cube modality config")
print("  Embodiment tag: NEW_EMBODIMENT")
print("  State dimensions: 29 active (44 total with zeros)")
print("  Action dimensions: 29 active (44 total with zeros)")
print("  Video key: ego_view (256x256)")
print("=" * 60)
