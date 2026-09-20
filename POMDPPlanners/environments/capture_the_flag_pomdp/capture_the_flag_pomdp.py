# SPDX-License-Identifier: MIT

# pylint: disable=too-many-lines
"""CaptureTheFlag POMDP Environment Implementation.

Two teams share a grid field split by a midline. The planner controls the blue
team as one joint controller; the red team is part of the transition model and
follows a stochastic role policy. Each team has a flag. Blue wins by carrying
the red flag to the blue base while its own flag is home.

Blue cannot see the red players, and does not know which of ``K`` candidate
cells holds the red flag. Both are inferred from the same observation: a noisy
Manhattan range from every blue player to every red player, plus a per-player
binary detector for the red flag cell whose reliability decays with distance.
Because every blue player measures every red player, the range readings
multiply -- team size buys localisation quickly, while a single flag scan is
weak evidence. That asymmetry is the planning problem.

The task is complete when blue scores. Being tagged is a setback, not a
failure: a tagged player respawns at its own base, drops the flag it carried,
and is frozen for a few steps.

Classes:
    CaptureTheFlagStepChannel: Per-step measurement channels.
    CaptureTheFlagMetrics: Metric names reported by this environment.
    CaptureTheFlagPOMDP: The environment.
"""

from collections.abc import Hashable
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.step_info_metrics import (
    EpisodeReduction,
    StepInfoMetric,
)
from POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_pomdp_utils import (
    ACTION_DELTAS,
    ACTION_SCAN,
    N_PLAYER_ACTIONS,
    PERPENDICULAR,
    RedRole,
    StateLayout,
    decode_joint_action,
    manhattan,
)

# Sentinel emitted from, and only from, terminal states.
_TERMINAL_OBSERVATION_VALUE = -1.0

# Above this, the opening observation distribution is served as its single most
# likely member instead of enumerated. The support grows as
# ``K * 3 ** (n_blue * n_red) * 2 ** n_blue``, so a large team would otherwise
# spend real time building a distribution nothing reads in full.
_MAX_ENUMERATED_OBSERVATIONS = 8192

DEFAULT_TREES: Tuple[Tuple[int, int], ...] = (
    (2, 1),
    (2, 5),
    (3, 3),
    (4, 0),
    (4, 6),
    (6, 1),
    (6, 5),
)
DEFAULT_RED_FLAG_CANDIDATES: Tuple[Tuple[int, int], ...] = ((7, 1), (7, 5), (6, 3), (8, 2))


class CaptureTheFlagStepChannel(Enum):
    """Per-step channels reported by :meth:`CaptureTheFlagPOMDP.step_info`."""

    CAPTURED = "captured"
    RECORDED_STEP = "recorded_step"
    ENDED_BY_GOAL = "ended_by_goal"
    ENDED_BY_FAILURE = "ended_by_failure"
    ENDED_BY_TIMEOUT = "ended_by_timeout"
    TAGS_SUFFERED = "tags_suffered"
    TAGS_INFLICTED = "tags_inflicted"
    HOLDING_ENEMY_FLAG = "holding_enemy_flag"
    BLUE_PLAYERS_IN_ENEMY_HALF = "blue_players_in_enemy_half"


class CaptureTheFlagMetrics(Enum):
    """Metric names for the CaptureTheFlag POMDP environment."""

    TASK_COMPLETION_RATE = "task_completion_rate"
    ENDED_BY_GOAL_RATE = "ended_by_goal_rate"
    ENDED_BY_FAILURE_RATE = "ended_by_failure_rate"
    ENDED_BY_TIMEOUT_RATE = "ended_by_timeout_rate"
    AVERAGE_EPISODE_LENGTH = "average_episode_length"
    AVERAGE_TAGS_SUFFERED = "average_tags_suffered"
    AVERAGE_TAGS_INFLICTED = "average_tags_inflicted"
    AVERAGE_STEPS_HOLDING_ENEMY_FLAG = "average_steps_holding_enemy_flag"
    AVERAGE_BLUE_PLAYER_STEPS_IN_ENEMY_HALF = "average_blue_player_steps_in_enemy_half"
    MAX_BLUE_PLAYERS_IN_ENEMY_HALF = "max_blue_players_in_enemy_half"


class CaptureTheFlagPOMDP(DiscreteActionsEnvironment):  # pylint: disable=too-many-public-methods
    """Team capture-the-flag on a grid, with hidden red players and a hidden flag.

    Attributes:
        grid_size: Field size as ``(width, height)``.
        midline: Column index of the neutral midline. Blue owns columns below
            it, red owns columns above it.
        trees: Impassable cells, sorted.
        n_blue: Number of blue players, controlled jointly by the planner.
        n_red: Number of red players, driven by the environment's role policy.
        red_flag_candidates: Cells the red flag may occupy; which one is hidden.
        layout: Index offsets into the flat state vector.
    """

    # pylint: disable-next=too-many-arguments,too-many-locals,too-many-statements,dangerous-default-value
    def __init__(
        self,
        grid_size: Tuple[int, int] = (9, 7),
        midline: int = 4,
        trees: Optional[Iterable[Tuple[int, int]]] = None,
        n_blue: int = 2,
        n_red: int = 2,
        n_red_defenders: int = 1,
        blue_base: Tuple[int, int] = (0, 3),
        red_base: Tuple[int, int] = (8, 3),
        blue_flag_cell: Tuple[int, int] = (1, 3),
        red_flag_candidates: Optional[Sequence[Tuple[int, int]]] = None,
        slip_probability: float = 0.1,
        range_error_probability: float = 0.2,
        red_pursuit_probability: float = 0.7,
        red_alert_radius: int = 3,
        freeze_steps: int = 3,
        tagger_cooldown_steps: int = 2,
        detector_half_distance_move: float = 1.5,
        detector_half_distance_scan: float = 4.0,
        score_to_win: int = 1,
        capture_reward: float = 100.0,
        concede_penalty: float = 100.0,
        tagged_penalty: float = 25.0,
        tag_reward: float = 10.0,
        pickup_reward: float = 20.0,
        move_cost: float = 1.0,
        scan_cost: float = 2.0,
        discount_factor: float = 0.98,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the CaptureTheFlag environment.

        Args:
            grid_size: Field size as ``(width, height)``.
            midline: Neutral column splitting the two halves.
            trees: Impassable cells, sorted. ``None`` selects :data:`DEFAULT_TREES`;
                pass an empty collection for an open field.
            n_blue: Number of blue players.
            n_red: Number of red players.
            n_red_defenders: How many red players defend rather than attack.
            blue_base: Blue spawn and scoring cell.
            red_base: Red spawn and scoring cell.
            blue_flag_cell: Blue flag's home cell, known to both sides.
            red_flag_candidates: Cells the red flag may sit on. ``None``
                selects :data:`DEFAULT_RED_FLAG_CANDIDATES`.
            slip_probability: Chance a blue move deflects perpendicular.
            range_error_probability: Chance a range badge reads off by one.
            red_pursuit_probability: Chance a red player steps toward its target.
            red_alert_radius: Distance at which a defender switches to chasing.
            freeze_steps: Steps a tagged player is frozen after respawning.
            tagger_cooldown_steps: Steps before a tagger may tag again.
            detector_half_distance_move: Flag-detector half-distance without scan.
            detector_half_distance_scan: Flag-detector half-distance after scan.
            score_to_win: Captures needed to end the episode.
            capture_reward: Reward when blue scores.
            concede_penalty: Penalty when red scores.
            tagged_penalty: Penalty per blue player tagged.
            tag_reward: Reward per red player tagged by blue.
            pickup_reward: Reward when blue first picks up the red flag.
            move_cost: Per-player cost of a move, stay or blocked action.
            scan_cost: Per-player cost of a scan.
            discount_factor: Discount factor for future rewards.
            output_dir: Optional directory for logging output.
            debug: Enable debug logging.
            use_queue_logger: Whether to use queue-based logging.

        Raises:
            ValueError: If the field, teams, bases or flag cells are
                inconsistent -- for example a base standing on a tree, a
                candidate cell outside the red half, or more defenders than
                red players.
        """
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.midline = int(midline)
        # `trees is None` rather than `if not trees`: an explicitly empty
        # collection is a valid open field and must not fall back to the
        # default set.
        #
        # Stored sorted, not as a set: `config_id` serializes an unrecognised
        # type through `str()`, and a set's text depends on its iteration
        # order, so two environments built from the same cells in a different
        # order would hash to different ids. Membership goes through the
        # private mirror below, which `config_id` skips.
        cells = DEFAULT_TREES if trees is None else trees
        self.trees: Tuple[Tuple[int, int], ...] = tuple(sorted((int(x), int(y)) for x, y in cells))
        self._tree_cells = set(self.trees)
        self.n_blue = int(n_blue)
        self.n_red = int(n_red)
        self.n_red_defenders = int(n_red_defenders)
        self.blue_base = (int(blue_base[0]), int(blue_base[1]))
        self.red_base = (int(red_base[0]), int(red_base[1]))
        self.blue_flag_cell = (int(blue_flag_cell[0]), int(blue_flag_cell[1]))
        candidates = (
            DEFAULT_RED_FLAG_CANDIDATES if red_flag_candidates is None else red_flag_candidates
        )
        self.red_flag_candidates = tuple((int(x), int(y)) for x, y in candidates)
        self.slip_probability = float(slip_probability)
        self.range_error_probability = float(range_error_probability)
        self.red_pursuit_probability = float(red_pursuit_probability)
        self.red_alert_radius = int(red_alert_radius)
        self.freeze_steps = int(freeze_steps)
        self.tagger_cooldown_steps = int(tagger_cooldown_steps)
        self.detector_half_distance_move = float(detector_half_distance_move)
        self.detector_half_distance_scan = float(detector_half_distance_scan)
        self.score_to_win = int(score_to_win)
        self.capture_reward = float(capture_reward)
        self.concede_penalty = float(concede_penalty)
        self.tagged_penalty = float(tagged_penalty)
        self.tag_reward = float(tag_reward)
        self.pickup_reward = float(pickup_reward)
        self.move_cost = float(move_cost)
        self.scan_cost = float(scan_cost)

        self._validate_configuration()

        self.layout = StateLayout(n_blue=self.n_blue, n_red=self.n_red)
        self.actions = list(range(N_PLAYER_ACTIONS**self.n_blue))
        self.red_roles = tuple(
            RedRole.DEFEND if j < self.n_red_defenders else RedRole.ATTACK
            for j in range(self.n_red)
        )
        self.observation_size = (
            2 * self.n_blue + self.n_blue * self.n_red + self.n_blue + 2 + self.n_blue + 2
        )
        self.max_range = self.grid_size[0] + self.grid_size[1] - 2

        super().__init__(
            discount_factor=discount_factor,
            name="capture_the_flag",
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE,
                observation_space=SpaceType.CONTINUOUS,
            ),
            reward_range=self._compute_reward_range(),
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

    def _validate_configuration(self) -> None:
        """Reject configurations whose meaning is undefined.

        Raises:
            ValueError: If the field, teams, bases or flag cells are inconsistent.
        """
        width, height = self.grid_size
        if width < 3 or height < 1:
            raise ValueError(f"grid_size must be at least (3, 1), got {self.grid_size}")
        if not 0 < self.midline < width - 1:
            raise ValueError(
                f"midline must leave a column on each side, got {self.midline} for width {width}"
            )
        if self.n_blue < 1 or self.n_red < 1:
            raise ValueError("each team needs at least one player")
        if not 0 <= self.n_red_defenders <= self.n_red:
            raise ValueError(
                f"n_red_defenders must be in [0, n_red], got {self.n_red_defenders}/{self.n_red}"
            )
        if not 0.0 <= self.slip_probability <= 1.0:
            raise ValueError(f"slip_probability must be in [0, 1], got {self.slip_probability}")
        if not 0.0 <= self.range_error_probability <= 1.0:
            raise ValueError(
                f"range_error_probability must be in [0, 1], got {self.range_error_probability}"
            )
        if not 0.0 <= self.red_pursuit_probability <= 1.0:
            raise ValueError(
                f"red_pursuit_probability must be in [0, 1], got {self.red_pursuit_probability}"
            )
        if self.detector_half_distance_move <= 0 or self.detector_half_distance_scan <= 0:
            raise ValueError("detector half-distances must be positive")
        if self.freeze_steps < 1:
            # Not merely "non-negative". A tag is recovered after the fact from
            # the freeze counter it sets, by both the reward and the metrics, so
            # a zero-length freeze would make every tag invisible: free for the
            # planner and absent from both tag metrics, with nothing raised. A
            # tag that costs no time is also not a tag.
            raise ValueError(f"freeze_steps must be at least 1, got {self.freeze_steps}")
        if self.tagger_cooldown_steps < 0:
            raise ValueError("tagger_cooldown_steps cannot be negative")
        if self.score_to_win < 1:
            raise ValueError(f"score_to_win must be at least 1, got {self.score_to_win}")
        if not self.red_flag_candidates:
            raise ValueError("at least one red flag candidate is required")

        named = {
            "blue_base": self.blue_base,
            "red_base": self.red_base,
            "blue_flag_cell": self.blue_flag_cell,
        }
        for label, cell in named.items():
            if not self._in_bounds(cell):
                raise ValueError(f"{label} {cell} is outside the field")
            if cell in self._tree_cells:
                raise ValueError(f"{label} {cell} stands on a tree")
        if not self.is_blue_half(self.blue_base) or not self.is_blue_half(self.blue_flag_cell):
            raise ValueError("blue base and flag must lie in the blue half")
        if not self.is_red_half(self.red_base):
            raise ValueError("red base must lie in the red half")
        for cell in self.red_flag_candidates:
            if not self._in_bounds(cell) or cell in self._tree_cells:
                raise ValueError(f"red flag candidate {cell} is not a free cell")
            if not self.is_red_half(cell):
                raise ValueError(f"red flag candidate {cell} is not in the red half")
        if len(set(self.red_flag_candidates)) != len(self.red_flag_candidates):
            raise ValueError("red flag candidates must be distinct")

    def _compute_reward_range(self) -> Tuple[float, float]:
        """Bound every reward this configuration can produce.

        The reward terms are not mutually exclusive, so the bound is the joint
        worst case rather than the largest single term. Blue scoring and red
        scoring *are* mutually exclusive -- each requires the other side's flag
        to be home -- which is why only one of them appears in each end.

        The worst step for blue: red scores, every blue player is tagged, and
        every player pays the most expensive action cost. The best step: blue
        scores, picks the flag up on the same step (possible when a candidate
        cell coincides with the blue base), tags every red player, and every
        player pays the cheapest action cost. Terminal states score zero, which
        the bound also has to contain.

        Returns:
            The ``(minimum, maximum)`` reward bound.
        """
        # Each term is folded in by sign rather than assumed positive: nothing
        # stops a caller passing a negative bonus or a negative penalty, and a
        # bound that assumed the signs would then exclude ordinary rewards.
        terms = (
            self.capture_reward,
            -self.concede_penalty,
            -self.tagged_penalty * self.n_blue,
            self.tag_reward * self.n_red,
            self.pickup_reward,
        )
        action_costs = (
            -self.move_cost * self.n_blue,
            -self.scan_cost * self.n_blue,
        )
        minimum = sum(min(term, 0.0) for term in terms) + min(action_costs)
        maximum = sum(max(term, 0.0) for term in terms) + max(action_costs)
        # A terminal state scores 0.0 whatever the costs are, so the bound has
        # to admit it even when every action is expensive.
        return (float(min(minimum, 0.0)), float(max(maximum, 0.0)))

    # ------------------------------------------------------------------ field

    def _in_bounds(self, cell: Tuple[int, int]) -> bool:
        """Return whether a cell lies inside the field rectangle."""
        return 0 <= cell[0] < self.grid_size[0] and 0 <= cell[1] < self.grid_size[1]

    def is_free(self, cell: Tuple[int, int]) -> bool:
        """Return whether a cell is inside the field and not a tree.

        Args:
            cell: The ``(x, y)`` cell to test.

        Returns:
            ``True`` if a player may stand on the cell.
        """
        return self._in_bounds(cell) and cell not in self._tree_cells

    def is_blue_half(self, cell: Tuple[int, int]) -> bool:
        """Return whether a cell belongs to the blue half."""
        return int(cell[0]) < self.midline

    def is_red_half(self, cell: Tuple[int, int]) -> bool:
        """Return whether a cell belongs to the red half."""
        return int(cell[0]) > self.midline

    def free_cells(self) -> List[Tuple[int, int]]:
        """List every cell a player may stand on.

        Returns:
            All free cells, in column-major order.
        """
        return [
            (x, y)
            for x in range(self.grid_size[0])
            for y in range(self.grid_size[1])
            if (x, y) not in self._tree_cells
        ]

    def _neighbours(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return the free 4-neighbours of a cell, plus the cell itself."""
        options = [cell]
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            candidate = (cell[0] + dx, cell[1] + dy)
            if self.is_free(candidate):
                options.append(candidate)
        return options

    def red_flag_cell(self, state: np.ndarray) -> Tuple[int, int]:
        """Return the red flag's home cell for a state.

        Args:
            state: A state vector.

        Returns:
            The candidate cell the red flag belongs to in this state.
        """
        return self.red_flag_candidates[int(state[self.layout.flag_cell])]

    # ------------------------------------------------------- transition model

    def _blue_move_distribution(
        self, cell: Tuple[int, int], player_action: int, frozen: bool
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Distribution over one blue player's next cell.

        A move slips to either perpendicular direction with probability
        ``slip_probability / 2``; a move into a tree or off the field leaves the
        player where it stood. Two different slips can be blocked into the same
        cell, so outcomes are accumulated rather than listed.

        Args:
            cell: The player's current cell.
            player_action: That player's action id.
            frozen: Whether the player is serving a respawn freeze.

        Returns:
            ``(cell, probability)`` pairs summing to one.
        """
        if frozen or player_action not in PERPENDICULAR:
            return [(cell, 1.0)]
        weights = [(player_action, 1.0 - self.slip_probability)]
        for slipped in PERPENDICULAR[player_action]:
            weights.append((slipped, self.slip_probability / 2.0))
        outcomes: Dict[Tuple[int, int], float] = {}
        for candidate_action, weight in weights:
            if weight == 0.0:
                continue
            dx, dy = ACTION_DELTAS[candidate_action]
            target = (cell[0] + dx, cell[1] + dy)
            landing = target if self.is_free(target) else cell
            outcomes[landing] = outcomes.get(landing, 0.0) + weight
        return sorted(outcomes.items())

    def _red_target(
        self,
        role: RedRole,
        red_index: int,
        blue_cells: Sequence[Tuple[int, int]],
        red_cell: Tuple[int, int],
        flag_cell: Tuple[int, int],
        carrier_blue_flag: int,
    ) -> Tuple[int, int]:
        """Pick the cell one red player is heading for this step.

        A defender guards the red flag, switching to the nearest blue intruder
        inside its own half once that intruder is within the alert radius. An
        attacker runs for the blue flag, or for its own base once carrying it.

        Args:
            role: The player's role.
            red_index: Its index within the red team, one-based in the state.
            blue_cells: Blue player cells after the blue move.
            red_cell: The red player's current cell.
            flag_cell: The red flag's home cell.
            carrier_blue_flag: Red carrier id for the blue flag, 0 if none.

        Returns:
            The cell the player wants to reduce its distance to.
        """
        # Carrying beats the role. A defender that happens to pick the blue
        # flag up -- which it can, standing on its own half's flag cell while
        # chasing -- must still run it home, or the flag sits in the red half
        # forever and red can never score.
        if carrier_blue_flag == red_index + 1:
            return self.red_base
        if role is RedRole.ATTACK:
            return self.blue_flag_cell
        intruders = [cell for cell in blue_cells if self.is_red_half(cell)]
        if intruders:
            nearest = min(intruders, key=lambda cell: (manhattan(red_cell, cell), cell))
            if manhattan(red_cell, nearest) <= self.red_alert_radius:
                return nearest
        return self.guard_post(flag_cell)

    def guard_post(self, flag_cell: Tuple[int, int]) -> Tuple[int, int]:
        """Return the cell an idle defender stands on to watch a flag.

        Beside the flag, never on it. A defender parked on the objective makes
        the objective unreachable: pick-up resolves before tagging, so an
        attacker arriving on the cell takes the flag and is tagged on the same
        step, every time, and the flag returns home. Standing one cell away
        leaves the approach contestable, which is both how the real game is
        played and what makes the defender's alert radius mean anything.

        Args:
            flag_cell: The flag being watched.

        Returns:
            The free neighbour nearest the defender's own base, or the flag
            cell itself when it has no free neighbour to stand on.
        """
        neighbours = [cell for cell in self._neighbours(flag_cell) if cell != flag_cell]
        if not neighbours:
            return flag_cell
        return min(neighbours, key=lambda cell: (manhattan(cell, self.red_base), cell))

    def _red_move_distribution(
        self, red_cell: Tuple[int, int], target: Tuple[int, int], frozen: bool
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Distribution over one red player's next cell.

        Args:
            red_cell: The player's current cell.
            target: The cell it is heading for.
            frozen: Whether the player is serving a respawn freeze.

        Returns:
            ``(cell, probability)`` pairs summing to one.
        """
        if frozen:
            return [(red_cell, 1.0)]
        options = self._neighbours(red_cell)
        best_distance = min(manhattan(option, target) for option in options)
        closing = [option for option in options if manhattan(option, target) == best_distance]
        outcomes: Dict[Tuple[int, int], float] = {}
        for option in options:
            outcomes[option] = outcomes.get(option, 0.0) + (
                1.0 - self.red_pursuit_probability
            ) / len(options)
        for option in closing:
            outcomes[option] += self.red_pursuit_probability / len(closing)
        return sorted(outcomes.items())

    # The stages are one ordered procedure whose order is the semantics;
    # splitting them into helpers would scatter the very sequence the
    # docstring exists to pin down.
    # pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
    def _apply_deterministic_stages(
        self,
        state: np.ndarray,
        blue_cells: List[Tuple[int, int]],
        red_cells: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Resolve pick-up, tagging, scoring and counters after both teams moved.

        The stage order is semantics, not style. Pick-up runs *before* tagging,
        so a player tagged on the flag cell has already taken the flag and
        therefore drops it. Both scoring conditions are evaluated against the
        pre-scoring carrier ids, which keeps blue scoring and red scoring
        mutually exclusive rather than letting whichever is resolved first
        enable the other.

        Counters are decremented against the values carried in from ``state``
        before any stage sets a fresh one, so a freeze set this step lasts its
        full length instead of losing a step immediately.

        Args:
            state: The state the step was taken from.
            blue_cells: Blue cells after the blue move.
            red_cells: Red cells after the red move.

        Returns:
            The successor state vector.
        """
        layout = self.layout
        next_state = np.zeros(layout.size, dtype=np.float64)
        next_state[layout.flag_cell] = state[layout.flag_cell]

        freeze_blue = [max(0, int(state[layout.freeze_blue + i]) - 1) for i in range(self.n_blue)]
        freeze_red = [max(0, int(state[layout.freeze_red + j]) - 1) for j in range(self.n_red)]
        cooldown_blue = [
            max(0, int(state[layout.cooldown_blue + i]) - 1) for i in range(self.n_blue)
        ]
        cooldown_red = [max(0, int(state[layout.cooldown_red + j]) - 1) for j in range(self.n_red)]

        was_frozen_blue = [int(state[layout.freeze_blue + i]) > 0 for i in range(self.n_blue)]
        was_frozen_red = [int(state[layout.freeze_red + j]) > 0 for j in range(self.n_red)]

        carrier_red_flag = int(state[layout.carrier_red_flag])
        carrier_blue_flag = int(state[layout.carrier_blue_flag])
        flag_cell = self.red_flag_cell(state)

        # Stage 3: pick-up. Lowest index wins a tie.
        if carrier_red_flag == 0:
            for i, cell in enumerate(blue_cells):
                if cell == flag_cell and not was_frozen_blue[i]:
                    carrier_red_flag = i + 1
                    break
        if carrier_blue_flag == 0:
            for j, cell in enumerate(red_cells):
                if cell == self.blue_flag_cell and not was_frozen_red[j]:
                    carrier_blue_flag = j + 1
                    break

        # Stage 4: tagging. Only in the enemy half, only by a free tagger.
        #
        # ``tagged_blue``/``tagged_red`` track who was sent home *this* step.
        # The tagger guards have to consult them rather than the incoming
        # freeze: a player tagged earlier in this stage has already been
        # teleported to its own base, and without this it would tag an
        # opponent standing on that base from a cell it never walked through.
        tagged_blue: Set[int] = set()
        tagged_red: Set[int] = set()
        for j, red_cell in enumerate(red_cells):
            if was_frozen_red[j] or int(state[layout.cooldown_red + j]) > 0:
                continue
            for i, blue_cell in enumerate(blue_cells):
                if blue_cell != red_cell or not self.is_red_half(blue_cell):
                    continue
                if was_frozen_blue[i] or i in tagged_blue:
                    continue
                blue_cells[i] = self.blue_base
                freeze_blue[i] = self.freeze_steps
                cooldown_red[j] = self.tagger_cooldown_steps
                if carrier_red_flag == i + 1:
                    carrier_red_flag = 0
                tagged_blue.add(i)
                break
        for i, blue_cell in enumerate(blue_cells):
            if was_frozen_blue[i] or i in tagged_blue:
                continue
            if int(state[layout.cooldown_blue + i]) > 0:
                continue
            for j, red_cell in enumerate(red_cells):
                if red_cell != blue_cell or not self.is_blue_half(red_cell):
                    continue
                if was_frozen_red[j] or j in tagged_red:
                    continue
                red_cells[j] = self.red_base
                freeze_red[j] = self.freeze_steps
                cooldown_blue[i] = self.tagger_cooldown_steps
                if carrier_blue_flag == j + 1:
                    carrier_blue_flag = 0
                tagged_red.add(j)
                break

        # Stage 5: scoring. Both sides are judged against the same carrier ids.
        score_blue = int(state[layout.score_blue])
        score_red = int(state[layout.score_red])
        blue_scores = (
            carrier_red_flag != 0
            and blue_cells[carrier_red_flag - 1] == self.blue_base
            and carrier_blue_flag == 0
        )
        red_scores = (
            carrier_blue_flag != 0
            and red_cells[carrier_blue_flag - 1] == self.red_base
            and carrier_red_flag == 0
        )
        if blue_scores:
            score_blue += 1
            carrier_red_flag = 0
        if red_scores:
            score_red += 1
            carrier_blue_flag = 0

        layout.write_blue_cells(next_state, blue_cells)
        layout.write_red_cells(next_state, red_cells)
        next_state[layout.carrier_red_flag] = float(carrier_red_flag)
        next_state[layout.carrier_blue_flag] = float(carrier_blue_flag)
        for i in range(self.n_blue):
            next_state[layout.freeze_blue + i] = float(freeze_blue[i])
            next_state[layout.cooldown_blue + i] = float(cooldown_blue[i])
        for j in range(self.n_red):
            next_state[layout.freeze_red + j] = float(freeze_red[j])
            next_state[layout.cooldown_red + j] = float(cooldown_red[j])
        next_state[layout.score_blue] = float(score_blue)
        next_state[layout.score_red] = float(score_red)
        return next_state

    def _successor_distribution(
        self, state: np.ndarray, action: int
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Enumerate every successor of one ``(state, action)`` with its probability.

        The support is the product of the per-player move outcomes -- at most
        three per blue player and five per red player -- so it stays small
        enough to enumerate exactly. Successors that coincide after the
        deterministic stages are merged, which is what makes this a
        distribution rather than a list of branches.

        Args:
            state: The state the step is taken from.
            action: The joint action id.

        Returns:
            The distinct successor states and their probabilities.
        """
        layout = self.layout
        player_actions = decode_joint_action(action, self.n_blue)
        blue_cells = layout.blue_cells(state)
        red_cells = layout.red_cells(state)
        flag_cell = self.red_flag_cell(state)
        carrier_blue_flag = int(state[layout.carrier_blue_flag])

        blue_options = [
            self._blue_move_distribution(
                blue_cells[i], player_actions[i], int(state[layout.freeze_blue + i]) > 0
            )
            for i in range(self.n_blue)
        ]

        merged: Dict[bytes, float] = {}
        arrays: Dict[bytes, np.ndarray] = {}
        for blue_combo, blue_prob in _cartesian(blue_options):
            red_options = []
            for j in range(self.n_red):
                target = self._red_target(
                    self.red_roles[j], j, blue_combo, red_cells[j], flag_cell, carrier_blue_flag
                )
                red_options.append(
                    self._red_move_distribution(
                        red_cells[j], target, int(state[layout.freeze_red + j]) > 0
                    )
                )
            for red_combo, red_prob in _cartesian(red_options):
                successor = self._apply_deterministic_stages(
                    state, list(blue_combo), list(red_combo)
                )
                key = successor.tobytes()
                merged[key] = merged.get(key, 0.0) + blue_prob * red_prob
                arrays.setdefault(key, successor)
        keys = list(merged)
        return [arrays[key] for key in keys], np.array([merged[key] for key in keys])

    def sample_next_state(self, state: np.ndarray, action: int, n_samples: int = 1) -> Any:
        """Sample the successor state.

        Args:
            state: Current state vector.
            action: Joint action id.
            n_samples: Number of independent samples to draw.

        Returns:
            One state vector when ``n_samples`` is 1, otherwise a list of them.
        """
        if self.is_terminal(state):
            frozen = np.asarray(state, dtype=np.float64).copy()
            return frozen if n_samples == 1 else [frozen.copy() for _ in range(n_samples)]
        samples = [self._draw_one_successor(state, action) for _ in range(n_samples)]
        return samples[0] if n_samples == 1 else samples

    def _draw_one_successor(self, state: np.ndarray, action: int) -> np.ndarray:
        """Draw one successor by sampling each player, not the joint outcome.

        Enumerating the joint distribution to draw a single sample costs
        ``3 ** n_blue * 5 ** n_red`` applications of the deterministic stages --
        225 at the reference team size, for one transition. Planners sample
        this in their innermost loop, so drawing per player and resolving the
        stages once is the difference between a usable environment and an
        unusable one. Both paths share
        :meth:`_apply_deterministic_stages` and the same per-player
        distributions, which is what keeps them the same model; the parity is
        asserted in ``test_sampling_matches_the_enumerated_distribution``.

        Args:
            state: The state the step is taken from.
            action: The joint action id.

        Returns:
            One successor state.
        """
        layout = self.layout
        player_actions = decode_joint_action(action, self.n_blue)
        blue_cells = layout.blue_cells(state)
        red_cells = layout.red_cells(state)
        flag_cell = self.red_flag_cell(state)
        carrier_blue_flag = int(state[layout.carrier_blue_flag])

        drawn_blue: List[Tuple[int, int]] = []
        for i in range(self.n_blue):
            options = self._blue_move_distribution(
                blue_cells[i], player_actions[i], int(state[layout.freeze_blue + i]) > 0
            )
            drawn_blue.append(_draw_cell(options))
        drawn_red: List[Tuple[int, int]] = []
        for j in range(self.n_red):
            target = self._red_target(
                self.red_roles[j], j, drawn_blue, red_cells[j], flag_cell, carrier_blue_flag
            )
            options = self._red_move_distribution(
                red_cells[j], target, int(state[layout.freeze_red + j]) > 0
            )
            drawn_red.append(_draw_cell(options))
        return self._apply_deterministic_stages(state, drawn_blue, drawn_red)

    def sample_next_state_batch(self, states: Any, action: int) -> np.ndarray:
        """Sample one successor for each state in a batch.

        Args:
            states: Array of state vectors, shape ``(N, state_size)``.
            action: Joint action id applied to every state.

        Returns:
            Successor states as a ``(N, state_size)`` float64 array, matching
            the dtype of the single-state path.
        """
        state_array = np.asarray(states, dtype=np.float64)
        if state_array.ndim == 1:
            state_array = state_array.reshape(1, -1)
        return np.asarray(
            [self.sample_next_state(row, action, 1) for row in state_array], dtype=np.float64
        )

    def transition_log_probability(
        self, state: np.ndarray, action: int, next_states: Any
    ) -> np.ndarray:
        """Log-probability of each candidate successor.

        Args:
            state: The state the step was taken from.
            action: The joint action id.
            next_states: Candidate successor state vectors.

        Returns:
            One log-probability per candidate; ``-inf`` for unreachable ones.
        """
        if self.is_terminal(state):
            frozen = np.asarray(state, dtype=np.float64)
            probabilities = np.array(
                [1.0 if _same_state(candidate, frozen) else 0.0 for candidate in next_states]
            )
        else:
            successors, weights = self._successor_distribution(state, action)
            lookup = {successor.tobytes(): weight for successor, weight in zip(successors, weights)}
            probabilities = np.array(
                [lookup.get(_state_key(candidate), 0.0) for candidate in next_states]
            )
        with np.errstate(divide="ignore"):
            return np.log(probabilities)

    # ------------------------------------------------------ observation model

    def _detector_half_distance(self, player_action: Optional[int]) -> float:
        """Return the flag detector's half-distance for one player's action."""
        if player_action == ACTION_SCAN:
            return self.detector_half_distance_scan
        return self.detector_half_distance_move

    def _detection_probability(self, distance: int, player_action: Optional[int]) -> float:
        """Probability the flag detector fires at a given distance."""
        return 0.5 * (1.0 + 2.0 ** (-distance / self._detector_half_distance(player_action)))

    def _range_probabilities(self, true_distance: int) -> Dict[int, float]:
        """Distribution over one range reading.

        Readings are clipped to ``[0, max_range]`` and the mass that would fall
        outside is folded back onto the end point, so the distribution sums to
        one for every true distance including at the field's extremes.

        Args:
            true_distance: The exact Manhattan distance.

        Returns:
            Reading value to probability.
        """
        outcomes: Dict[int, float] = {}
        candidates = [
            (true_distance, 1.0 - self.range_error_probability),
            (true_distance - 1, self.range_error_probability / 2.0),
            (true_distance + 1, self.range_error_probability / 2.0),
        ]
        for value, weight in candidates:
            if weight == 0.0:
                continue
            clipped = int(min(max(value, 0), self.max_range))
            outcomes[clipped] = outcomes.get(clipped, 0.0) + weight
        return outcomes

    def _true_distances(self, next_state: np.ndarray) -> Tuple[List[List[int]], List[int]]:
        """Return the exact range and flag distances a state implies.

        Args:
            next_state: The realised successor state.

        Returns:
            The ``n_blue x n_red`` range distances and the per-blue-player
            distance to the red flag cell.
        """
        blue_cells = self.layout.blue_cells(next_state)
        red_cells = self.layout.red_cells(next_state)
        flag_cell = self.red_flag_cell(next_state)
        ranges = [[manhattan(blue, red) for red in red_cells] for blue in blue_cells]
        flag_distances = [manhattan(blue, flag_cell) for blue in blue_cells]
        return ranges, flag_distances

    def _terminal_observation(self) -> Tuple[float, ...]:
        """Return the sentinel observation emitted by terminal states."""
        return tuple([_TERMINAL_OBSERVATION_VALUE] * self.observation_size)

    def _observed_prefix(self, next_state: np.ndarray) -> List[float]:
        """Return the exactly-observed part of an observation."""
        layout = self.layout
        prefix: List[float] = []
        for cell in layout.blue_cells(next_state):
            prefix.extend((float(cell[0]), float(cell[1])))
        return prefix

    def _observed_suffix(self, next_state: np.ndarray) -> List[float]:
        """Return the exactly-observed tail of an observation."""
        layout = self.layout
        suffix = [
            float(next_state[layout.carrier_red_flag]),
            float(next_state[layout.carrier_blue_flag] != 0.0),
        ]
        suffix.extend(float(next_state[layout.freeze_blue + i]) for i in range(self.n_blue))
        suffix.extend((float(next_state[layout.score_blue]), float(next_state[layout.score_red])))
        return suffix

    def sample_observation(self, next_state: np.ndarray, action: int, n_samples: int = 1) -> Any:
        """Sample the blue team's observation of a successor state.

        Args:
            next_state: The realised successor state.
            action: The joint action taken to reach it, which sets each
                player's detector range.
            n_samples: Number of independent samples to draw.

        Returns:
            One observation tuple when ``n_samples`` is 1, otherwise a list.
        """
        if self.is_terminal(next_state):
            terminal = self._terminal_observation()
            return terminal if n_samples == 1 else [terminal] * n_samples

        player_actions = decode_joint_action(action, self.n_blue)
        ranges, flag_distances = self._true_distances(next_state)
        prefix = self._observed_prefix(next_state)
        suffix = self._observed_suffix(next_state)

        range_draws = np.random.random(size=(n_samples, self.n_blue * self.n_red))
        detector_draws = np.random.random(size=(n_samples, self.n_blue))
        samples: List[Tuple[float, ...]] = []
        for sample_index in range(n_samples):
            observation = list(prefix)
            pair = 0
            for i in range(self.n_blue):
                for j in range(self.n_red):
                    observation.append(
                        float(
                            _inverse_cdf(
                                self._range_probabilities(ranges[i][j]),
                                range_draws[sample_index, pair],
                            )
                        )
                    )
                    pair += 1
            for i in range(self.n_blue):
                probability = self._detection_probability(flag_distances[i], player_actions[i])
                observation.append(float(detector_draws[sample_index, i] < probability))
            observation.extend(suffix)
            samples.append(tuple(observation))
        return samples[0] if n_samples == 1 else samples

    def observation_log_probability(
        self, next_state: np.ndarray, action: int, observations: Any
    ) -> np.ndarray:
        """Log-likelihood of each observation under a successor state.

        The exactly-observed components act as a delta factor: an observation
        that disagrees with them is impossible, not merely unlikely.

        Args:
            next_state: The realised successor state.
            action: The joint action taken to reach it.
            observations: Candidate observations.

        Returns:
            One log-likelihood per candidate.
        """
        result = np.zeros(len(observations))
        if self.is_terminal(next_state):
            terminal = self._terminal_observation()
            for index, observation in enumerate(observations):
                result[index] = float(self.is_equal_observation(observation, terminal))
            with np.errstate(divide="ignore"):
                return np.log(result)

        player_actions = decode_joint_action(action, self.n_blue)
        ranges, flag_distances = self._true_distances(next_state)
        prefix = self._observed_prefix(next_state)
        suffix = self._observed_suffix(next_state)
        range_tables = [
            self._range_probabilities(ranges[i][j])
            for i in range(self.n_blue)
            for j in range(self.n_red)
        ]
        detection = [
            self._detection_probability(flag_distances[i], player_actions[i])
            for i in range(self.n_blue)
        ]
        n_pairs = self.n_blue * self.n_red

        for index, observation in enumerate(observations):
            values = np.asarray(observation, dtype=np.float64).ravel()
            if values.size != self.observation_size:
                continue
            if not np.array_equal(values[: len(prefix)], np.asarray(prefix)):
                continue
            if not np.array_equal(values[-len(suffix) :], np.asarray(suffix)):
                continue
            probability = 1.0
            for pair in range(n_pairs):
                reading = values[len(prefix) + pair]
                # Rounding here would hand a fractional reading the likelihood
                # of the nearest integer, and the sampler can never emit one.
                # The observation space is declared continuous, so nothing stops
                # a caller passing one; it is impossible, not approximate.
                if reading != np.floor(reading):
                    probability = 0.0
                    break
                probability *= range_tables[pair].get(int(reading), 0.0)
            if probability == 0.0:
                continue
            for i in range(self.n_blue):
                bit = values[len(prefix) + n_pairs + i]
                if bit not in (0.0, 1.0):
                    probability = 0.0
                    break
                probability *= detection[i] if bit == 1.0 else 1.0 - detection[i]
            result[index] = probability
        with np.errstate(divide="ignore"):
            return np.log(result)

    # ------------------------------------------------------------ reward/term

    @property
    def reward_requires_next_state(self) -> bool:
        """Whether :meth:`reward` needs the realised successor.

        Returns:
            Always ``True``: scoring, tagging and pick-up are all properties of
            the realised transition, not of ``(state, action)`` alone.
        """
        return True

    def reward(self, state: np.ndarray, action: int, next_state: Any = None) -> float:
        """Reward for one transition.

        Args:
            state: The state the step was taken from.
            action: The joint action id.
            next_state: The realised successor. Sampled here when ``None``,
                which callers should avoid: it consumes randomness and returns
                the reward of a different draw than the trajectory took.

        Returns:
            The immediate reward.
        """
        if self.is_terminal(state):
            return 0.0
        if next_state is None:
            next_state = self.sample_next_state(state, action, 1)
        layout = self.layout
        counts = self._transition_counts(state, next_state)
        score_delta_blue = int(next_state[layout.score_blue]) - int(state[layout.score_blue])
        score_delta_red = int(next_state[layout.score_red]) - int(state[layout.score_red])
        picked_up = (
            int(state[layout.carrier_red_flag]) == 0
            and int(next_state[layout.carrier_red_flag]) != 0
        )
        action_cost = sum(
            self.scan_cost if player_action == ACTION_SCAN else self.move_cost
            for player_action in decode_joint_action(action, self.n_blue)
        )
        return (
            self.capture_reward * score_delta_blue
            - self.concede_penalty * score_delta_red
            - self.tagged_penalty * counts["tags_suffered"]
            + self.tag_reward * counts["tags_inflicted"]
            + self.pickup_reward * float(picked_up)
            - action_cost
        )

    def reward_batch(self, states: Any, action: int, next_states: Any = None) -> np.ndarray:
        """Rewards for a batch of transitions.

        Args:
            states: Array of state vectors.
            action: The joint action id applied to every state.
            next_states: Realised successors, one per state.

        Returns:
            One reward per transition.
        """
        state_array = np.asarray(states, dtype=np.float64)
        if state_array.ndim == 1:
            state_array = state_array.reshape(1, -1)
        if next_states is None:
            return np.array(
                [self.reward(row, action, None) for row in state_array], dtype=np.float64
            )
        next_array = np.asarray(next_states, dtype=np.float64)
        if next_array.ndim == 1:
            next_array = next_array.reshape(1, -1)
        return np.array(
            [self.reward(row, action, next_row) for row, next_row in zip(state_array, next_array)],
            dtype=np.float64,
        )

    def _transition_counts(self, state: Any, next_state: Any) -> Dict[str, int]:
        """Count tags in both directions from the two states.

        A freshly set freeze counter is the signature of a tag: a player that
        was free and is now frozen for the full ``freeze_steps`` was tagged
        this step, and a frozen player cannot be tagged again.

        Args:
            state: The state the step was taken from.
            next_state: The realised successor.

        Returns:
            ``tags_suffered`` and ``tags_inflicted`` for this transition.
        """
        layout = self.layout
        if not _is_state(state, layout.size) or not _is_state(next_state, layout.size):
            return {"tags_suffered": 0, "tags_inflicted": 0}
        suffered = sum(
            1
            for i in range(self.n_blue)
            if int(state[layout.freeze_blue + i]) == 0
            and int(next_state[layout.freeze_blue + i]) == self.freeze_steps
        )
        inflicted = sum(
            1
            for j in range(self.n_red)
            if int(state[layout.freeze_red + j]) == 0
            and int(next_state[layout.freeze_red + j]) == self.freeze_steps
        )
        return {"tags_suffered": suffered, "tags_inflicted": inflicted}

    def is_terminal(self, state: np.ndarray) -> bool:
        """Return whether either side has reached the winning score."""
        layout = self.layout
        return (
            int(state[layout.score_blue]) >= self.score_to_win
            or int(state[layout.score_red]) >= self.score_to_win
        )

    # --------------------------------------------------------------- initial

    def _spawn_state(self, flag_index: int) -> np.ndarray:
        """Build the opening state for one red flag placement."""
        layout = self.layout
        state = np.zeros(layout.size, dtype=np.float64)
        layout.write_blue_cells(state, [self.blue_base] * self.n_blue)
        layout.write_red_cells(state, [self.red_base] * self.n_red)
        state[layout.flag_cell] = float(flag_index)
        return state

    def initial_state_dist(self) -> Distribution:
        """Opening distribution: everything known but the red flag's cell.

        Returns:
            A uniform distribution over the red flag candidates.
        """
        states = [self._spawn_state(index) for index in range(len(self.red_flag_candidates))]
        probabilities = np.ones(len(states)) / len(states)
        return DiscreteDistribution(values=states, probs=probabilities)

    def initial_observation_dist(self) -> Distribution:
        """Observation distribution before the first action.

        This is a real draw from the observation model, not a representative
        reading. Everything but the red flag's cell is known at spawn, so the
        distribution is the mixture over the candidates of the sensor noise
        each one implies -- and a filter that weights the opening observation
        through :meth:`observation_log_probability` therefore stays uniform
        over the candidates instead of favouring the nearer ones.

        Returns:
            The exact opening observation distribution, or the single most
            likely observation when the support is larger than
            :data:`_MAX_ENUMERATED_OBSERVATIONS` and enumerating it would cost
            more than it informs.
        """
        prefix = self._observed_prefix(self._spawn_state(0))
        suffix = self._observed_suffix(self._spawn_state(0))
        per_candidate = []
        for index in range(len(self.red_flag_candidates)):
            state = self._spawn_state(index)
            ranges, flag_distances = self._true_distances(state)
            range_tables = [
                sorted(self._range_probabilities(ranges[i][j]).items())
                for i in range(self.n_blue)
                for j in range(self.n_red)
            ]
            detector_tables = [
                sorted(
                    {
                        1.0: self._detection_probability(flag_distances[i], None),
                        0.0: 1.0 - self._detection_probability(flag_distances[i], None),
                    }.items()
                )
                for i in range(self.n_blue)
            ]
            per_candidate.append((range_tables, detector_tables))

        support = sum(
            _support_size(range_tables) * _support_size(detector_tables)
            for range_tables, detector_tables in per_candidate
        )
        weight_per_candidate = 1.0 / len(self.red_flag_candidates)
        if support > _MAX_ENUMERATED_OBSERVATIONS:
            observation = list(prefix)
            range_tables, detector_tables = per_candidate[0]
            observation.extend(
                float(max(table, key=lambda row: row[1])[0]) for table in range_tables
            )
            observation.extend(
                float(max(table, key=lambda row: row[1])[0]) for table in detector_tables
            )
            observation.extend(suffix)
            return DiscreteDistribution(values=[tuple(observation)], probs=np.array([1.0]))

        merged: Dict[Tuple[float, ...], float] = {}
        for range_tables, detector_tables in per_candidate:
            for values, probability in _mixture(list(range_tables) + list(detector_tables)):
                observation = (
                    tuple(prefix) + tuple(float(value) for value in values) + tuple(suffix)
                )
                merged[observation] = (
                    merged.get(observation, 0.0) + probability * weight_per_candidate
                )
        keys = sorted(merged)
        return DiscreteDistribution(
            values=list(keys), probs=np.array([merged[key] for key in keys])
        )

    def get_actions(self) -> List[int]:
        """Return every joint action id."""
        return self.actions

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        """Return whether two observations are identical."""
        return np.array_equal(
            np.asarray(observation1, dtype=np.float64).ravel(),
            np.asarray(observation2, dtype=np.float64).ravel(),
        )

    def hash_action(self, action: Any) -> Hashable:
        """Return a hashable key for a joint action id."""
        return int(action)

    def hash_observation(self, observation: Any) -> Hashable:
        """Return a hashable key agreeing with :meth:`is_equal_observation`."""
        return tuple(np.asarray(observation, dtype=np.float64).ravel().tolist())

    # --------------------------------------------------------------- metrics

    def step_info(self, state: Any, action: Any, next_state: Any) -> Dict[str, float]:
        """Report per-step channels for this transition.

        Args:
            state: The state the step was taken from, or the final state on the
                terminal bookkeeping step.
            action: The joint action, or ``None`` on the terminal step.
            next_state: The realised successor, or ``None`` on the terminal step.

        Returns:
            The per-step channels. Transition-derived channels report zero on
            the terminal step; state-derived ones still report, because the
            episode-end channels are read from the final state.
        """
        del action  # every channel here is read from the states
        layout = self.layout
        if not _is_state(state, layout.size):
            return {}
        # The episode-end channels describe the state the step *landed* in, not
        # the one it left. The terminal bookkeeping record would normally carry
        # the final state, but the episode runner checks its step budget before
        # it checks termination, so an episode that scores on its last budgeted
        # action never gets that record -- and reading `state` would file a real
        # capture as a timeout. Scoring late is exactly when this environment
        # scores, so the bias would run one way.
        landed = next_state if _is_state(next_state, layout.size) else state
        blue_scored = int(landed[layout.score_blue]) >= self.score_to_win
        red_scored = int(landed[layout.score_red]) >= self.score_to_win
        counts = self._transition_counts(state, next_state)
        in_enemy_half = sum(1 for cell in layout.blue_cells(state) if self.is_red_half(cell))
        return {
            CaptureTheFlagStepChannel.CAPTURED.value: float(blue_scored),
            CaptureTheFlagStepChannel.RECORDED_STEP.value: 1.0,
            CaptureTheFlagStepChannel.ENDED_BY_GOAL.value: float(blue_scored),
            CaptureTheFlagStepChannel.ENDED_BY_FAILURE.value: float(red_scored),
            CaptureTheFlagStepChannel.ENDED_BY_TIMEOUT.value: float(
                not blue_scored and not red_scored
            ),
            CaptureTheFlagStepChannel.TAGS_SUFFERED.value: float(counts["tags_suffered"]),
            CaptureTheFlagStepChannel.TAGS_INFLICTED.value: float(counts["tags_inflicted"]),
            CaptureTheFlagStepChannel.HOLDING_ENEMY_FLAG.value: float(
                int(state[layout.carrier_red_flag]) != 0
            ),
            CaptureTheFlagStepChannel.BLUE_PLAYERS_IN_ENEMY_HALF.value: float(in_enemy_half),
        }

    def get_metric_specs(self) -> List[StepInfoMetric]:
        """Declare every metric this environment reports.

        Returns:
            One spec per channel, in the order :class:`CaptureTheFlagMetrics`
            declares the names.
        """
        return [
            StepInfoMetric(
                name=CaptureTheFlagMetrics.TASK_COMPLETION_RATE.value,
                channel=CaptureTheFlagStepChannel.CAPTURED.value,
                # ANY, not ALL: scoring is a goal reached once, not a condition
                # that has to hold on every step.
                per_episode=EpisodeReduction.ANY,
            ),
            StepInfoMetric(
                name=CaptureTheFlagMetrics.ENDED_BY_GOAL_RATE.value,
                channel=CaptureTheFlagStepChannel.ENDED_BY_GOAL.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=CaptureTheFlagMetrics.ENDED_BY_FAILURE_RATE.value,
                channel=CaptureTheFlagStepChannel.ENDED_BY_FAILURE.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=CaptureTheFlagMetrics.ENDED_BY_TIMEOUT_RATE.value,
                channel=CaptureTheFlagStepChannel.ENDED_BY_TIMEOUT.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=CaptureTheFlagMetrics.AVERAGE_EPISODE_LENGTH.value,
                channel=CaptureTheFlagStepChannel.RECORDED_STEP.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=CaptureTheFlagMetrics.AVERAGE_TAGS_SUFFERED.value,
                channel=CaptureTheFlagStepChannel.TAGS_SUFFERED.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=CaptureTheFlagMetrics.AVERAGE_TAGS_INFLICTED.value,
                channel=CaptureTheFlagStepChannel.TAGS_INFLICTED.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=CaptureTheFlagMetrics.AVERAGE_STEPS_HOLDING_ENEMY_FLAG.value,
                channel=CaptureTheFlagStepChannel.HOLDING_ENEMY_FLAG.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=CaptureTheFlagMetrics.AVERAGE_BLUE_PLAYER_STEPS_IN_ENEMY_HALF.value,
                channel=CaptureTheFlagStepChannel.BLUE_PLAYERS_IN_ENEMY_HALF.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=CaptureTheFlagMetrics.MAX_BLUE_PLAYERS_IN_ENEMY_HALF.value,
                channel=CaptureTheFlagStepChannel.BLUE_PLAYERS_IN_ENEMY_HALF.value,
                # Severity alongside the count: one player exposed for twenty
                # steps and two exposed for ten give the same sum.
                per_episode=EpisodeReduction.MAX,
            ),
        ]

    def get_metric_names(self) -> List[str]:
        """Return every metric name, in declaration order."""
        return [metric.value for metric in CaptureTheFlagMetrics]

    def cache_visualization(
        self, history: List[StepData], output_dir: Path, episode_index: int
    ) -> None:
        """Render the episode as an animated GIF.

        Args:
            history: The episode's recorded steps.
            output_dir: Directory the ``.gif`` is written into.
            episode_index: Zero-based episode index, used to name the file.
        """
        # Imported here so the environment stays importable without Pillow's
        # drawing stack, matching how the other environments defer visualizers.
        from POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_visualizer import (  # pylint: disable=import-outside-toplevel
            CaptureTheFlagVisualizer,
        )

        CaptureTheFlagVisualizer(self).render_episode(
            history, output_dir / f"capture_the_flag_{episode_index}.gif"
        )


def _cartesian(
    per_player: Sequence[Sequence[Tuple[Tuple[int, int], float]]],
) -> List[Tuple[Tuple[Tuple[int, int], ...], float]]:
    """Expand per-player outcome lists into joint outcomes with probabilities.

    Args:
        per_player: One ``(cell, probability)`` list per player.

    Returns:
        ``(cells, probability)`` pairs over the whole team.
    """
    combos: List[Tuple[Tuple[Tuple[int, int], ...], float]] = [((), 1.0)]
    for options in per_player:
        expanded: List[Tuple[Tuple[Tuple[int, int], ...], float]] = []
        for cells, probability in combos:
            for cell, weight in options:
                if weight > 0.0:
                    expanded.append((cells + (cell,), probability * weight))
        combos = expanded
    return combos


def _support_size(tables: Sequence[Sequence[Tuple[Any, float]]]) -> int:
    """Return how many outcomes a product of small tables has.

    Args:
        tables: One ``(value, probability)`` list per factor.

    Returns:
        The product of the factor sizes.
    """
    size = 1
    for table in tables:
        size *= len(table)
    return size


def _mixture(
    tables: Sequence[Sequence[Tuple[Any, float]]],
) -> List[Tuple[Tuple[Any, ...], float]]:
    """Expand independent factors into their joint outcomes.

    Args:
        tables: One ``(value, probability)`` list per factor.

    Returns:
        ``(values, probability)`` pairs over all factors.
    """
    combos: List[Tuple[Tuple[Any, ...], float]] = [((), 1.0)]
    for table in tables:
        combos = [
            (values + (value,), probability * weight)
            for values, probability in combos
            for value, weight in table
            if weight > 0.0
        ]
    return combos


def _draw_cell(options: Sequence[Tuple[Tuple[int, int], float]]) -> Tuple[int, int]:
    """Draw one cell from a small ``(cell, probability)`` list.

    Args:
        options: Outcomes and their probabilities, summing to one.

    Returns:
        The sampled cell.
    """
    if len(options) == 1:
        return options[0][0]
    draw = float(np.random.random())
    cumulative = 0.0
    cell = options[0][0]
    for cell, probability in options:
        cumulative += probability
        if draw < cumulative:
            return cell
    return cell


def _inverse_cdf(table: Dict[int, float], draw: float) -> int:
    """Map a uniform draw to a value of a small discrete distribution.

    Args:
        table: Value to probability; need not be sorted.
        draw: A uniform sample in ``[0, 1)``.

    Returns:
        The sampled value.
    """
    cumulative = 0.0
    value = 0
    for value, probability in sorted(table.items()):
        cumulative += probability
        if draw < cumulative:
            return value
    return value


def _is_state(candidate: Any, size: int) -> bool:
    """Return whether a candidate is a well-formed state vector."""
    return isinstance(candidate, np.ndarray) and candidate.shape == (size,)


def _state_key(candidate: Any) -> bytes:
    """Return the byte key a state vector is merged under."""
    return np.ascontiguousarray(np.asarray(candidate, dtype=np.float64)).tobytes()


def _same_state(candidate: Any, reference: np.ndarray) -> bool:
    """Return whether a candidate equals a reference state vector."""
    array = np.asarray(candidate, dtype=np.float64)
    return array.shape == reference.shape and bool(np.array_equal(array, reference))
