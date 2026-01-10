"""
GR1 Modality configuration for finetuning.

This registers the GR1 embodiment config needed for finetuning on GR1-format data.
"""

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS


# GR1 modality configuration (matches pretrained model expectations)
GR1_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view_bg_crop_pad_res256_freq20"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",    # 7 DOF
            "left_hand",   # 6 DOF
            "right_arm",   # 7 DOF
            "right_hand",  # 6 DOF
            "waist",       # 3 DOF
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
            "left_arm",
            "left_hand",
            "right_arm",
            "right_hand",
            "waist",
        ],
        action_configs=[
            # left_arm
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_hand
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_arm
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_hand
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # waist
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
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

# Register GR1 config
MODALITY_CONFIGS["gr1"] = GR1_CONFIG

print("Registered GR1 modality config for finetuning")
