# SPDX-License-Identifier: MIT

"""Field geometry, red-team policy roles and state layout for CaptureTheFlag.

The state is a flat ``float64`` array so it can be batched, hashed by bytes and
serialized without a dataclass. This module owns the layout so the environment,
the visualizer and any future vectorized copy read the same offsets instead of
each hard-coding index arithmetic.

Classes:
    RedRole: Which objective a red player pursues.
    StateLayout: Index offsets into the flat state vector.

Functions:
    manhattan: Grid L1 distance.
    decode_joint_action: Split a joint action id into per-blue-player actions.
    encode_joint_action: Inverse of :func:`decode_joint_action`.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Per-player action ids. 0..3 move, 4 holds position, 5 holds position and
# widens that player's flag detector for the step.
ACTION_NORTH, ACTION_EAST, ACTION_SOUTH, ACTION_WEST = 0, 1, 2, 3
ACTION_STAY, ACTION_SCAN = 4, 5
PLAYER_ACTIONS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
N_PLAYER_ACTIONS = len(PLAYER_ACTIONS)

# (dx, dy) for each move action; stay and scan do not displace.
ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    ACTION_NORTH: (0, 1),
    ACTION_EAST: (1, 0),
    ACTION_SOUTH: (0, -1),
    ACTION_WEST: (-1, 0),
    ACTION_STAY: (0, 0),
    ACTION_SCAN: (0, 0),
}

# The two directions a slipped move can land in, per intended move action.
PERPENDICULAR: Dict[int, Tuple[int, int]] = {
    ACTION_NORTH: (ACTION_EAST, ACTION_WEST),
    ACTION_SOUTH: (ACTION_EAST, ACTION_WEST),
    ACTION_EAST: (ACTION_NORTH, ACTION_SOUTH),
    ACTION_WEST: (ACTION_NORTH, ACTION_SOUTH),
}


class RedRole(Enum):
    """What a red player is trying to do.

    Attributes:
        DEFEND: Guard the red flag cell; chase blue intruders in the red half.
        ATTACK: Run for the blue flag, then carry it back to the red base.
    """

    DEFEND = "defend"
    ATTACK = "attack"


def manhattan(a: Sequence[int], b: Sequence[int]) -> int:
    """Return the L1 distance between two grid cells.

    Args:
        a: First cell as ``(x, y)``.
        b: Second cell as ``(x, y)``.

    Returns:
        ``|a_x - b_x| + |a_y - b_y|``.
    """
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def decode_joint_action(action: int, n_blue: int) -> Tuple[int, ...]:
    """Split a joint action id into one per-player action per blue player.

    The joint action space is the base-``6`` odometer over the blue team, least
    significant digit first, so player 0's action is ``action % 6``.

    Args:
        action: Joint action id in ``[0, 6 ** n_blue)``.
        n_blue: Number of blue players.

    Returns:
        Per-player action ids, ordered by player index.
    """
    digits: List[int] = []
    remaining = int(action)
    for _ in range(n_blue):
        digits.append(remaining % N_PLAYER_ACTIONS)
        remaining //= N_PLAYER_ACTIONS
    return tuple(digits)


def encode_joint_action(player_actions: Sequence[int]) -> int:
    """Combine per-player actions into a joint action id.

    Args:
        player_actions: One action id per blue player, ordered by player index.

    Returns:
        The joint action id that :func:`decode_joint_action` inverts.
    """
    joint = 0
    for index, player_action in enumerate(player_actions):
        joint += int(player_action) * (N_PLAYER_ACTIONS**index)
    return joint


@dataclass(frozen=True)
class StateLayout:
    """Index offsets into the flat CaptureTheFlag state vector.

    The vector is laid out as blue positions, red positions, then the scalar
    bookkeeping fields. Sizes depend only on the team sizes, so one layout
    instance serves every state of one environment configuration.

    Attributes:
        n_blue: Number of blue players.
        n_red: Number of red players.
    """

    n_blue: int
    n_red: int

    @property
    def blue_pos(self) -> int:
        """Index of the first blue coordinate."""
        return 0

    @property
    def red_pos(self) -> int:
        """Index of the first red coordinate."""
        return 2 * self.n_blue

    @property
    def flag_cell(self) -> int:
        """Index of the red flag's candidate index."""
        return 2 * self.n_blue + 2 * self.n_red

    @property
    def carrier_red_flag(self) -> int:
        """Index of the blue carrier id for the red flag (0 = at home)."""
        return self.flag_cell + 1

    @property
    def carrier_blue_flag(self) -> int:
        """Index of the red carrier id for the blue flag (0 = at home)."""
        return self.flag_cell + 2

    @property
    def freeze_blue(self) -> int:
        """Index of the first blue respawn-freeze counter."""
        return self.flag_cell + 3

    @property
    def freeze_red(self) -> int:
        """Index of the first red respawn-freeze counter."""
        return self.freeze_blue + self.n_blue

    @property
    def cooldown_blue(self) -> int:
        """Index of the first blue tagger-cooldown counter."""
        return self.freeze_red + self.n_red

    @property
    def cooldown_red(self) -> int:
        """Index of the first red tagger-cooldown counter."""
        return self.cooldown_blue + self.n_blue

    @property
    def score_blue(self) -> int:
        """Index of the blue score."""
        return self.cooldown_red + self.n_red

    @property
    def score_red(self) -> int:
        """Index of the red score."""
        return self.score_blue + 1

    @property
    def size(self) -> int:
        """Total length of the state vector."""
        return self.score_red + 1

    def blue_cells(self, state: np.ndarray) -> List[Tuple[int, int]]:
        """Read every blue player's cell out of a state vector.

        Args:
            state: A state vector of length :attr:`size`.

        Returns:
            One ``(x, y)`` cell per blue player.
        """
        base = self.blue_pos
        return [
            (int(state[base + 2 * i]), int(state[base + 2 * i + 1])) for i in range(self.n_blue)
        ]

    def red_cells(self, state: np.ndarray) -> List[Tuple[int, int]]:
        """Read every red player's cell out of a state vector.

        Args:
            state: A state vector of length :attr:`size`.

        Returns:
            One ``(x, y)`` cell per red player.
        """
        base = self.red_pos
        return [(int(state[base + 2 * j]), int(state[base + 2 * j + 1])) for j in range(self.n_red)]

    def write_blue_cells(self, state: np.ndarray, cells: Sequence[Tuple[int, int]]) -> None:
        """Write blue player cells into a state vector in place.

        Args:
            state: The state vector to modify.
            cells: One ``(x, y)`` cell per blue player.
        """
        base = self.blue_pos
        for i, (cell_x, cell_y) in enumerate(cells):
            state[base + 2 * i] = float(cell_x)
            state[base + 2 * i + 1] = float(cell_y)

    def write_red_cells(self, state: np.ndarray, cells: Sequence[Tuple[int, int]]) -> None:
        """Write red player cells into a state vector in place.

        Args:
            state: The state vector to modify.
            cells: One ``(x, y)`` cell per red player.
        """
        base = self.red_pos
        for j, (cell_x, cell_y) in enumerate(cells):
            state[base + 2 * j] = float(cell_x)
            state[base + 2 * j + 1] = float(cell_y)
