# SPDX-License-Identifier: MIT

"""CaptureTheFlag POMDP Environment Package.

Team capture-the-flag on a grid field split by a midline. The planner drives
the blue team jointly; the red team belongs to the transition model. Both the
red players and the red flag's cell are hidden and inferred from noisy ranges
and a per-player flag detector.
"""

from POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_pomdp import (
    CaptureTheFlagMetrics,
    CaptureTheFlagPOMDP,
    CaptureTheFlagStepChannel,
)
from POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_pomdp_utils import (
    RedRole,
    decode_joint_action,
    encode_joint_action,
)

__all__ = [
    "CaptureTheFlagPOMDP",
    "CaptureTheFlagMetrics",
    "CaptureTheFlagStepChannel",
    "RedRole",
    "decode_joint_action",
    "encode_joint_action",
]
