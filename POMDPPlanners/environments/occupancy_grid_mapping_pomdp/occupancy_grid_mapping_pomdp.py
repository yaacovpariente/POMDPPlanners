# SPDX-License-Identifier: MIT

"""Occupancy-grid exploration with known integer pose and noisy range scans.

State: [step, row, col, heading, hidden occupancy(C), inverse-map log-odds(C),
last noisy ranges(B)]. The transition samples motion and a noisy scan, stores
that scan and applies its inverse sensor update. Observation reveals the exact
pose and stored ranges. Hidden occupancy is used only by the forward
motion/sensor model, never by the inverse map update.

The per-beam range law is selected by ``range_noise_model``: the unbounded
Gaussian (the default, and the law every earlier result used), or the same
normal truncated to ``[0, +inf)`` so that no reading can be negative. The
truncated law is a renormalised density, not a clamp, and its normaliser
depends on each beam's nominal range, so it enters every likelihood per beam.
Neither law truncates above the maximum range.

A range at or above the maximum is interpreted as a miss. A shorter range
selects the nearest ray cell (negative readings select the first cell). Thus a
hit exactly at maximum range and a miss have identical observation laws and
identical updates for equal readings. This ambiguity is intentional; no hidden
hit flag is supplied to the robot.

The map is an approximate independent-cell inverse estimate. Its entropy is
not the entropy of the posterior over whole maps. Realised reward and completion use the observed inverse map. Planning
without a successor uses a fixed antithetic quadrature, mapped through the selected range law, of its expected
reduction. Repeated correlated beams can create false confidence. The separate whole-map particle belief predicts scans and motion.
Use OccupancyGridMappingBelief, or its batched twin in the
occupancy_grid_mapping_beliefs package, to condition the augmented state correctly.

Integer pose and three actions are a chosen simplification. Particle-based
MCTS also supports continuous states; it does not require this discretization.
"""

# pylint: disable=too-many-lines  # one environment, its sensor plumbing and its metrics

from enum import Enum, IntEnum
from pathlib import Path
from collections.abc import Hashable
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_maps import (
    OccupancyGridInitialStateDistribution,
    default_start_cell,
    optional_int,
    resolve_keep_free,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_update_rules import (
    OccupancyUpdateRule,
    default_update_rule,
    non_default_update_rule_id,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_sensor import (
    HEADING_STEPS,
    NUM_HEADINGS,
    RangeNoiseModel,
    build_ray_templates,
    cast_scan,
    grid_entropy_bits,
    log_odds_from_probability,
    quadrature_ranges,
    resolve_range_noise_model,
    sample_ranges,
    scan_log_density,
)


#: Index of the step counter inside a state vector.
STEP_INDEX = 0
#: Index of the robot row inside a state vector.
ROW_INDEX = 1
#: Index of the robot column inside a state vector.
COL_INDEX = 2
#: Index of the robot heading inside a state vector.
HEADING_INDEX = 3
#: Number of scalar pose fields preceding the two grid blocks.
POSE_WIDTH = 4

#: A cell counts as resolved once its occupancy log-odds leave this band. The
#: value is the log-odds of ``p = 0.95``, so "resolved" means the robot is at
#: least 95% sure either way.
RESOLVED_LOG_ODDS = 2.9444389791664403


class OccupancyGridAction(IntEnum):
    """The robot's three actions.

    Attributes:
        FORWARD: Attempt to move one cell along the current heading.
        TURN_LEFT: Rotate 90 degrees anticlockwise, staying in place.
        TURN_RIGHT: Rotate 90 degrees clockwise, staying in place.
    """

    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


class OccupancyGridStepChannel(Enum):
    """Per-step channels reported by :meth:`OccupancyGridMappingPOMDP.step_info`."""

    MAP_RESOLVED = "map_resolved"
    MAP_UNRESOLVED = "map_unresolved"
    EPISODE_FAILURE = "episode_failure"
    RECORDED_STEP = "recorded_step"
    RESIDUAL_ENTROPY_BITS = "residual_entropy_bits"
    RESOLVED_CELL_FRACTION = "resolved_cell_fraction"
    OBSTACLE_COLLISION = "obstacle_collision"
    SUCCESSFUL_TRANSLATION = "successful_translation"


class OccupancyGridMappingMetrics(Enum):
    """Metric names for the occupancy-grid mapping environment."""

    TASK_COMPLETION_RATE = "task_completion_rate"
    ENDED_BY_GOAL = "ended_by_goal"
    ENDED_BY_FAILURE = "ended_by_failure"
    ENDED_BY_TIMEOUT = "ended_by_timeout"
    AVERAGE_EPISODE_LENGTH = "average_episode_length"
    FINAL_RESIDUAL_ENTROPY_BITS = "final_residual_entropy_bits"
    MAX_RESOLVED_CELL_FRACTION = "max_resolved_cell_fraction"
    AVERAGE_OBSTACLE_COLLISIONS = "average_obstacle_collisions"
    AVERAGE_SUCCESSFUL_TRANSLATIONS = "average_successful_translations"


#: Type alias for an occupancy-grid mapping state.
OccupancyGridState = np.ndarray


# pylint: disable-next=too-many-public-methods
class OccupancyGridMappingPOMDP(DiscreteActionsEnvironment):
    """Explore until observed inverse-map entropy reaches its threshold.

    Motion checks the hidden static grid. Each transition draws a fresh noisy
    scan. Realised reward is inverse-map entropy reduction minus step cost;
    planning uses a numerical expectation. Either can be negative. Timeout is max_steps.
    """

    # pylint: disable-next=too-many-arguments,too-many-branches,too-many-locals,too-many-statements
    def __init__(
        self,
        num_rows: int = 10,
        num_cols: int = 10,
        num_beams: int = 24,
        field_of_view_degrees: float = 360.0,
        max_range_cells: float = 3.5,
        range_noise_std_cells: float = 0.35,
        range_noise_model: Union[RangeNoiseModel, str] = RangeNoiseModel.GAUSSIAN,
        hit_probability: float = 0.85,
        miss_probability: float = 0.15,
        log_odds_clamp: float = 6.0,
        update_rule: Optional[OccupancyUpdateRule] = None,
        num_obstacles: int = 3,
        max_obstacle_size: int = 2,
        has_boundary_wall: bool = True,
        start_row: Optional[int] = None,
        start_col: Optional[int] = None,
        start_heading: int = 0,
        move_failure_probability: float = 0.0,
        max_steps: int = 40,
        entropy_threshold_fraction: float = 0.25,
        step_cost: float = 0.0,
        discount_factor: float = 0.95,
        name: str = "OccupancyGridMapping",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the occupancy-grid mapping POMDP.

        Args:
            num_rows: Grid rows. Defaults to 10.
            num_cols: Grid columns. Defaults to 10. With the default wall the
                interior is 8x8, which is small enough that a particle-based
                tree search can carry whole maps as particles and large enough
                that the sensor cannot see all of it from one place.
            num_beams: Beams per scan. Defaults to 24. At the default range the
                gap between neighbouring beams at their far end is under one
                cell, so a full scan leaves no unswept wedge; halving it would
                leave cells that no beam ever crosses and an entropy floor the
                robot could not get under.
            field_of_view_degrees: Angular width of the fan, centred on the
                heading. Defaults to 360 -- a surround scanner, so heading
                affects where the robot can drive but not what it can see.
                A narrower cone makes turning informative in its own right.
            max_range_cells: Sensor range in cell widths. Defaults to 3.5, a
                little over a third of the grid, so the robot must travel to
                finish the map rather than resolving it from the start cell.
            range_noise_std_cells: Per-beam Gaussian range noise. Defaults to
                0.35 cells -- a tenth of the sensor's range, used by the forward sensor model. Finite map particles can still collapse.
            range_noise_model: Which per-beam range law that standard deviation
                parametrises. Defaults to ``RangeNoiseModel.GAUSSIAN``, the
                unbounded normal, which is what every result before this option
                existed was produced under. ``RangeNoiseModel.TRUNCATED_NORMAL``
                truncates it to ``[0, +inf)`` and renormalises, so no beam can
                report a negative distance -- a reading a real range finder
                cannot produce, and one the untruncated law assigns real
                probability to whenever a nominal range is within a few standard
                deviations of zero. Accepts the member or its string value.
                Readings above the maximum range are not truncated either way.
            hit_probability: Inverse sensor model probability for the cell a
                beam stops in. Defaults to 0.85.
            miss_probability: Inverse sensor model probability for a cell a beam
                passes through. Defaults to 0.15. The pair is deliberately
                stronger than the textbook 0.7/0.4: at 0.7/0.4 a cell needs
                roughly eight sightings to leave the unknown band, which is more
                than a short exploration episode ever gives it.
            log_odds_clamp: Symmetric bound on accumulated log-odds. Defaults to
                6.0 (``p ~= 0.9975``). Clamping is what stops a cell swept by
                many beams from becoming unrevisable.
            update_rule: How an observed scan changes the map. Defaults to
                ``None``, which builds the original rule
                (``NearestCellLogOddsUpdateRule``) from ``hit_probability``,
                ``miss_probability`` and ``log_odds_clamp`` -- so every result
                and every cached episode from before this argument existed is
                reproduced exactly. Supply an ``OccupancyUpdateRule`` to map
                with a different law; the scalar transition, the batched
                particle kernels and both rewards all use the one you supply.
                A rule other than the default changes the environment's
                ``config_id``, so its results cannot be served from a cache
                filled under another rule.
            num_obstacles: Rectangular blocks placed per episode. Defaults to 3.
            max_obstacle_size: Largest block side length. Defaults to 2.
            has_boundary_wall: Whether the outer ring is occupied. Defaults to
                ``True``, which closes the world.
            start_row: Robot's start row. Defaults to the grid centre.
            start_col: Robot's start column. Defaults to the grid centre.
            start_heading: Start heading index. Defaults to 0 (north).
            move_failure_probability: Chance an otherwise-valid forward move
                fails. Defaults to 0.0.
            max_steps: Transitions allowed per episode. Defaults to 40.
            entropy_threshold_fraction: Fraction of the initial entropy at or
                below which the map counts as resolved. Defaults to 0.25 --
                an average of a quarter of a bit per cell, which corresponds to
                roughly 96% certainty per cell and is reachable in two or three
                sightings at the default inverse sensor model.
            step_cost: Constant charge per step. Defaults to 0.0, leaving the
                objective pure information gain.
            discount_factor: Discount factor. Defaults to 0.95.
            name: Environment name. Defaults to ``"OccupancyGridMapping"``.
            output_dir: Output directory for logging. Defaults to ``None``.
            debug: Enable debug logging. Defaults to ``False``.
            use_queue_logger: Whether to use queue-based logging.

        Raises:
            ValueError: If the geometry, the sensor model or the episode limits
                are outside their valid ranges.
        """
        if num_rows < 1 or num_cols < 1:
            raise ValueError(f"grid must be at least 1x1, got {num_rows}x{num_cols}")
        if not 0 <= int(start_heading) < NUM_HEADINGS:
            raise ValueError(
                f"start_heading must be one of 0..{NUM_HEADINGS - 1}, got {start_heading}"
            )
        if not 0.0 <= move_failure_probability <= 1.0:
            raise ValueError(
                f"move_failure_probability must be in [0, 1], got {move_failure_probability}"
            )
        if range_noise_std_cells <= 0.0:
            raise ValueError(
                "range_noise_std_cells must be positive: a zero-width Gaussian has no "
                f"density for the particle filter to weight with, got {range_noise_std_cells}"
            )
        # Raises on an unknown name, so a typo in a config file fails at
        # construction rather than silently selecting the default law.
        range_noise_model = resolve_range_noise_model(range_noise_model)
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        if not 0.0 <= entropy_threshold_fraction <= 1.0:
            raise ValueError(
                "entropy_threshold_fraction must be in [0, 1], got " f"{entropy_threshold_fraction}"
            )
        if log_odds_clamp <= 0.0:
            raise ValueError(f"log_odds_clamp must be positive, got {log_odds_clamp}")
        if num_obstacles < 0:
            raise ValueError(f"num_obstacles must be non-negative, got {num_obstacles}")
        if max_obstacle_size < 1:
            raise ValueError(f"max_obstacle_size must be at least 1, got {max_obstacle_size}")

        num_cells = int(num_rows) * int(num_cols)

        # One cell holds at most one bit, so the whole grid holds at most
        # ``num_cells`` bits and a step can neither gain nor lose more than that.
        # The bound holds for every sensor setting, every map and every clamp:
        # both entropies in the difference already live in [0, num_cells]. The
        # loss side is not hypothetical -- a cell believed occupied that later
        # takes free-space evidence moves back towards p = 0.5, which raises the
        # grid's entropy and makes the step's reward negative. ``step_cost`` is
        # charged on every step, so it shifts both ends by the same amount.
        max_reward = float(num_cells) - float(step_cost)
        min_reward = -float(num_cells) - float(step_cost)

        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE, observation_space=SpaceType.CONTINUOUS
            ),
            reward_range=(min_reward, max_reward),
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        self.num_rows = int(num_rows)
        self.num_cols = int(num_cols)
        self.num_beams = int(num_beams)
        self.field_of_view_degrees = float(field_of_view_degrees)
        self.max_range_cells = float(max_range_cells)
        self.range_noise_std_cells = float(range_noise_std_cells)
        self.range_noise_model = range_noise_model
        self.hit_probability = float(hit_probability)
        self.miss_probability = float(miss_probability)
        self.log_odds_clamp = float(log_odds_clamp)
        self.num_obstacles = int(num_obstacles)
        self.max_obstacle_size = int(max_obstacle_size)
        self.has_boundary_wall = bool(has_boundary_wall)
        centre_row, centre_col = default_start_cell(self.num_rows, self.num_cols)
        self.start_row = optional_int(start_row, centre_row)
        self.start_col = optional_int(start_col, centre_col)
        self.start_heading = int(start_heading)
        self.move_failure_probability = float(move_failure_probability)
        self.max_steps = int(max_steps)
        self.entropy_threshold_fraction = float(entropy_threshold_fraction)
        self.step_cost = float(step_cost)

        if not 0 <= self.start_row < self.num_rows or not 0 <= self.start_col < self.num_cols:
            raise ValueError(
                f"start cell ({self.start_row}, {self.start_col}) is outside the "
                f"{self.num_rows}x{self.num_cols} grid"
            )
        if self.has_boundary_wall and not (
            0 < self.start_row < self.num_rows - 1 and 0 < self.start_col < self.num_cols - 1
        ):
            raise ValueError(
                f"start cell ({self.start_row}, {self.start_col}) is on the boundary wall"
            )

        self.num_cells = num_cells
        self.scan_offset = POSE_WIDTH + 2 * num_cells
        self.state_size = self.scan_offset + self.num_beams
        # Bumped from 2 when range_noise_model was added: the range law is part
        # of the sensor contract, so a cached episode from before the option
        # existed must not be reused for one after it.
        self.sensor_contract_version = 3
        # Fixed antithetic integration points leave the simulation RNG untouched.
        # They are unit normals; ``quadrature_ranges`` maps them into the
        # selected range law at use time.
        points = np.random.RandomState(0).normal(size=(4, self.num_beams))
        self._reward_noise = np.concatenate([points, -points])
        self.map_offset = POSE_WIDTH
        self.log_odds_offset = POSE_WIDTH + num_cells
        #: Entropy of the all-unknown grid, in bits: one bit per cell.
        self.initial_entropy_bits = float(num_cells)
        self.entropy_threshold_bits = self.entropy_threshold_fraction * self.initial_entropy_bits

        # Raises for a probability of exactly 0 or 1, which would be infinite
        # evidence no later reading could revise. Validated here rather than in
        # the argument block above so there is one implementation of the rule.
        self.occupied_log_odds = log_odds_from_probability(self.hit_probability)
        self.free_log_odds = log_odds_from_probability(self.miss_probability)
        if self.occupied_log_odds <= 0.0:
            raise ValueError(
                f"hit_probability must exceed 0.5 to be evidence of occupancy, "
                f"got {self.hit_probability}"
            )
        if self.free_log_odds >= 0.0:
            raise ValueError(
                f"miss_probability must be below 0.5 to be evidence of free space, "
                f"got {self.miss_probability}"
            )

        # Underscored so the rule object itself stays out of ``config_id`` and
        # ``__eq__``: the *default* rule is nothing but the three settings
        # above restated, so letting it into the identity would change every
        # existing config_id for no change in behaviour. What does go into the
        # identity is the line below -- an attribute that exists only when the
        # rule is not the default one, so a non-default rule can never be
        # served a cached result produced under the original update.
        self._update_rule = (
            default_update_rule(self.free_log_odds, self.occupied_log_odds, self.log_odds_clamp)
            if update_rule is None
            else update_rule
        )
        if not isinstance(self._update_rule, OccupancyUpdateRule):
            raise TypeError(
                "update_rule must be an OccupancyUpdateRule, got "
                f"{type(self._update_rule).__name__}"
            )
        rule_id = non_default_update_rule_id(
            self._update_rule, self.free_log_odds, self.occupied_log_odds, self.log_odds_clamp
        )
        if rule_id is not None:
            self.update_rule_config_id = rule_id

        # Derived read-only ray geometry. Underscored so it stays out of
        # ``config_id`` and ``__eq__``: it is a pure function of num_beams,
        # field_of_view_degrees and max_range_cells, all of which are already in
        # the identity, and putting several thousand offsets into a cache key
        # would be slow and redundant.
        self._ray_templates = build_ray_templates(
            num_beams=self.num_beams,
            field_of_view_degrees=self.field_of_view_degrees,
            max_range_cells=self.max_range_cells,
        )

    @property
    def update_rule(self) -> OccupancyUpdateRule:
        """The rule that turns an observed scan into the next map."""
        return self._update_rule

    # -- state accessors ------------------------------------------------

    def pose(self, state: OccupancyGridState) -> Tuple[int, int, int]:
        """Return the robot's ``(row, col, heading)`` from ``state``.

        Args:
            state: A state vector.

        Returns:
            The pose as three ints.
        """
        values = np.asarray(state, dtype=np.float64)
        return (
            int(round(values[ROW_INDEX])),
            int(round(values[COL_INDEX])),
            int(round(values[HEADING_INDEX])) % NUM_HEADINGS,
        )

    def true_map(self, state: OccupancyGridState) -> np.ndarray:
        """Return the episode's hidden true map as a ``(num_rows, num_cols)`` view.

        Args:
            state: A state vector.

        Returns:
            ``float64`` occupancy grid, non-zero where occupied.
        """
        values = np.asarray(state, dtype=np.float64)
        return values[self.map_offset : self.map_offset + self.num_cells].reshape(
            self.num_rows, self.num_cols
        )

    def log_odds(self, state: OccupancyGridState) -> np.ndarray:
        """Return the robot's occupancy grid, in log-odds, as a 2-D view.

        Args:
            state: A state vector.

        Returns:
            ``float64`` array of occupancy log-odds, shape ``(num_rows, num_cols)``.
        """
        values = np.asarray(state, dtype=np.float64)
        return values[self.log_odds_offset : self.log_odds_offset + self.num_cells].reshape(
            self.num_rows, self.num_cols
        )

    def occupancy_probabilities(self, state: OccupancyGridState) -> np.ndarray:
        """Return the robot's occupancy grid as probabilities.

        Provided for display and for reading a result; the environment itself
        never leaves log-odds space except to compute entropy.

        Args:
            state: A state vector.

        Returns:
            ``float64`` array in ``[0, 1]``, shape ``(num_rows, num_cols)``.
        """
        return 1.0 / (1.0 + np.exp(-self.log_odds(state)))

    def entropy_bits(self, state: OccupancyGridState) -> float:
        """Total occupancy-grid entropy of ``state``, in bits.

        Args:
            state: A state vector.

        Returns:
            Entropy in ``[0, num_cells]``.
        """
        return grid_entropy_bits(self.log_odds(state))

    # -- dynamics -------------------------------------------------------

    def get_actions(self) -> List[int]:
        """Return the three actions: forward, turn left, turn right."""
        return [int(action) for action in OccupancyGridAction]

    def _next_pose(
        self, row: int, col: int, heading: int, action: int, occupancy: np.ndarray
    ) -> Tuple[int, int, int, bool]:
        """Where the robot ends up, and whether a forward move was blocked.

        Args:
            row: Current row.
            col: Current column.
            heading: Current heading index.
            action: The action taken.
            occupancy: The episode's true map.

        Returns:
            ``(row, col, heading, blocked)``. ``blocked`` is ``True`` only for a
            forward move that ran into an occupied cell or off the grid, which
            is what the collision metric counts.
        """
        action = int(action)
        if action == int(OccupancyGridAction.TURN_LEFT):
            return row, col, (heading - 1) % NUM_HEADINGS, False
        if action == int(OccupancyGridAction.TURN_RIGHT):
            return row, col, (heading + 1) % NUM_HEADINGS, False

        delta_row, delta_col = HEADING_STEPS[heading]
        target_row, target_col = row + delta_row, col + delta_col
        inside = 0 <= target_row < self.num_rows and 0 <= target_col < self.num_cols
        if not inside or occupancy[target_row, target_col] != 0:
            return row, col, heading, True
        return target_row, target_col, heading, False

    def state_from_observation(self, state, observation):
        """Advance using only prior map and observed pose/ranges for mapping."""
        observation = np.asarray(observation, dtype=float)
        if observation.shape != (3 + self.num_beams,) or not np.all(np.isfinite(observation)):
            raise ValueError("observation must contain finite pose and ranges")
        row, col, heading = observation[:3]
        if (
            not np.array_equal(observation[:3], np.round(observation[:3]))
            or not 0 <= row < self.num_rows
            or not 0 <= col < self.num_cols
            or not 0 <= heading < NUM_HEADINGS
        ):
            raise ValueError("observation pose must be an in-grid integer pose")
        row, col, heading = int(row), int(col), int(heading)
        next_state = np.array(state, dtype=float, copy=True)
        next_state[STEP_INDEX] += 1
        next_state[1:4] = observation[:3]
        next_state[self.scan_offset :] = observation[3:]
        end = self.log_odds_offset + self.num_cells
        next_state[self.log_odds_offset : end] = self._update_rule.update_log_odds(
            next_state[self.log_odds_offset : end],
            row,
            col,
            heading,
            observation[3:],
            self._ray_templates[heading],
            self.num_rows,
            self.num_cols,
            self.max_range_cells,
        )
        return next_state

    def _state_after_pose(self, state, row, col, heading):
        """Motion-only successor, before drawing the scan."""
        result = np.array(state, dtype=float, copy=True)
        result[1:4] = row, col, heading
        return result

    def _motion_outcomes(
        self, state: OccupancyGridState, action: int
    ) -> Tuple[List[OccupancyGridState], np.ndarray]:
        """Enumerate moved poses and probabilities before advancing step or scan."""
        row, col, heading = self.pose(state)
        occupancy = self.true_map(state)
        next_row, next_col, next_heading, blocked = self._next_pose(
            row, col, heading, action, occupancy
        )
        moved = (next_row, next_col) != (row, col)
        if not moved or self.move_failure_probability <= 0.0:
            del blocked
            return [self._state_after_pose(state, next_row, next_col, next_heading)], np.array(
                [1.0]
            )
        return (
            [
                self._state_after_pose(state, next_row, next_col, next_heading),
                self._state_after_pose(state, row, col, heading),
            ],
            np.array([1.0 - self.move_failure_probability, self.move_failure_probability]),
        )

    def sample_next_state(self, state, action, n_samples=1):
        """Sample motion and noisy ranges, then apply the observed scan."""
        successors, probabilities = self._motion_outcomes(state, action)
        samples = []
        for _ in range(int(n_samples)):
            index = (
                0 if len(successors) == 1 else np.random.choice(len(successors), p=probabilities)
            )
            moved = successors[index]
            ranges = sample_ranges(
                self.nominal_scan(moved), self.range_noise_std_cells, self.range_noise_model
            )
            samples.append(self.state_from_observation(state, np.r_[moved[1:4], ranges]))
        return samples[0] if n_samples == 1 else np.asarray(samples)

    def predictive_observation_log_probability(self, state, action, observation):
        """Integrate discrete motion; score the continuous scan before it is stored.

        This is p(o | s,a), not p(o | augmented next_state,a). The conditional
        filter uses it as its importance weight and installs o deterministically.
        """
        observation = np.asarray(observation, dtype=float)
        if observation.shape != (3 + self.num_beams,) or not np.all(np.isfinite(observation)):
            return -np.inf
        successors, probabilities = self._motion_outcomes(state, action)
        score = -np.inf
        for moved, probability in zip(successors, probabilities):
            if probability <= 0 or not np.array_equal(moved[1:4], observation[:3]):
                continue
            # Under truncation the normaliser is Phi(nominal / sigma), which
            # differs from beam to beam and from particle to particle, so the
            # helper scores the scan under whichever law is selected.
            log_density = float(
                scan_log_density(
                    observation[3:],
                    self.nominal_scan(moved),
                    self.range_noise_std_cells,
                    self.range_noise_model,
                )
            )
            score = np.logaddexp(score, np.log(probability) + log_density)
        return float(score)

    def transition_log_probability(self, state, action, next_states):
        """Density on the scan coordinates with deterministic map/pose constraints."""
        candidates = np.atleast_2d(np.asarray(next_states, dtype=float))
        scores = np.full(len(candidates), -np.inf)
        for index, candidate in enumerate(candidates):
            observation = np.r_[candidate[1:4], candidate[self.scan_offset :]]
            score = self.predictive_observation_log_probability(state, action, observation)
            if np.isfinite(score) and np.array_equal(
                candidate, self.state_from_observation(state, observation)
            ):
                scores[index] = score
        return scores

    # -- observations ---------------------------------------------------

    def nominal_scan(self, state: OccupancyGridState) -> np.ndarray:
        """Noise-free ranges a scan from ``state``'s pose would return.

        Args:
            state: A state vector; its pose and its true map are both read.

        Returns:
            ``(num_beams,)`` ``float64`` ranges in cell widths.
        """
        row, col, heading = self.pose(state)
        ranges, _, _ = cast_scan(
            occupancy=self.true_map(state),
            row=row,
            col=col,
            template=self._ray_templates[heading],
            max_range_cells=self.max_range_cells,
        )
        return ranges

    def sample_observation(self, next_state, action, n_samples=1):
        """Reveal the scan already drawn in the transition; do not draw twice."""
        del action
        observation = np.r_[
            np.asarray(self.pose(next_state), dtype=float), next_state[self.scan_offset :]
        ]
        return observation if n_samples == 1 else np.tile(observation, (int(n_samples), 1))

    def observation_log_probability(self, next_state, action, observations):
        """Point mass on the exact pose and stored scan of the augmented state."""
        expected = self.sample_observation(next_state, action)
        candidates = np.atleast_2d(np.asarray(observations, dtype=float))
        return np.where(np.all(candidates == expected, axis=1), 0.0, -np.inf)

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        """Check whether two readings are the same array of numbers."""
        return bool(
            np.array_equal(
                np.asarray(observation1, dtype=np.float64),
                np.asarray(observation2, dtype=np.float64),
            )
        )

    def hash_observation(self, observation: Any) -> Hashable:
        """Return a hashable key agreeing with :meth:`is_equal_observation`.

        Args:
            observation: A reading.

        Returns:
            The reading's raw ``float64`` bytes, which is the standard surrogate
            here for an ndarray observation and matches ``array_equal``.
        """
        return np.ascontiguousarray(observation, dtype=np.float64).tobytes()

    def hash_action(self, action: Any) -> Hashable:
        """Return a hashable key for an action (already an int)."""
        return int(action)

    # -- reward ---------------------------------------------------------

    @property
    def reward_requires_next_state(self):
        """Simulation reward measures the realised observed map change."""
        return True

    def reward(self, state, action, next_state=None):
        """Score realised map change, or estimate its expectation for planning.

        Simulation supplies next_state and receives the observed inverse-map
        entropy difference. PFT_DPW's belief reward calls without a successor;
        that explicit fallback uses eight fixed antithetic unit-normal points
        mapped through the selected range law, so under truncation it never
        integrates over a scan the law cannot produce. The numerical
        expectation is approximate and uses no global randomness. Neither
        quantity is posterior whole-map information gain.
        """
        if next_state is not None:
            return self.entropy_bits(state) - self.entropy_bits(next_state) - self.step_cost
        successors, probabilities = self._motion_outcomes(state, action)
        after = 0.0
        for moved, probability in zip(successors, probabilities):
            nominal = self.nominal_scan(moved)
            for noise in self._reward_noise:
                observation = np.r_[
                    moved[1:4],
                    quadrature_ranges(
                        nominal, noise, self.range_noise_std_cells, self.range_noise_model
                    ),
                ]
                after += probability * self.entropy_bits(
                    self.state_from_observation(state, observation)
                )
        return self.entropy_bits(state) - after / len(self._reward_noise) - self.step_cost

    def reward_batch(self, states, action, next_states=None):
        """Batched :meth:`reward`: one array pass instead of one call per state.

        A belief-space planner asks for the expected reward of every particle
        at every node it expands, and without a successor each scalar call
        integrates eight hypothetical scans. Doing all particles and all
        integration points in one batched inverse-sensor call is what makes
        the planner's decision time drop; the values are the scalar ones.

        Args:
            states: Sequence of ``N`` states.
            action: Action executed from each state.
            next_states: Optional realised successors, length ``N``.

        Returns:
            Rewards of shape ``(N,)``.
        """
        states = np.asarray(states, dtype=np.float64).reshape(-1, self.state_size)
        kernels = self._batched_kernels
        if next_states is not None:
            successors = np.asarray(next_states, dtype=np.float64).reshape(-1, self.state_size)
            end = self.log_odds_offset + self.num_cells
            return (
                kernels.entropy_bits_rows(states[:, self.log_odds_offset : end])
                - kernels.entropy_bits_rows(successors[:, self.log_odds_offset : end])
                - self.step_cost
            )
        return kernels.batch_expected_reward(states, action, self._reward_noise, self.step_cost)

    @property
    def _batched_kernels(self):
        """The batched sensor and motion kernels, built on first use.

        Underscored and lazily built for the same reason as the ray templates:
        a pure function of settings already in the identity, and not worth
        shipping to every worker.
        """
        # pylint: disable-next=import-outside-toplevel
        from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_beliefs.occupancy_grid_mapping_vectorized_updater import (  # noqa: E501
            OccupancyGridMappingVectorizedUpdater,
        )

        # Keyed on the settings so a test that edits one after construction
        # gets kernels that match, not a stale copy.
        key = (
            self.num_rows,
            self.num_cols,
            self.num_beams,
            self.field_of_view_degrees,
            self.max_range_cells,
            self.range_noise_std_cells,
            self.range_noise_model,
            self.free_log_odds,
            self.occupied_log_odds,
            self.log_odds_clamp,
            self.move_failure_probability,
            self.sensor_contract_version,
            self._update_rule.config_id,
        )
        cached = vars(self).get("_batched_kernels_cache")
        if cached is None or cached[0] != key:
            cached = (key, OccupancyGridMappingVectorizedUpdater.from_environment(self))
            vars(self)["_batched_kernels_cache"] = cached
        return cached[1]

    # -- terminal / initial ---------------------------------------------

    def is_terminal(self, state: OccupancyGridState) -> bool:
        """Whether the map is resolved or the step budget is spent.

        Args:
            state: A state vector.

        Returns:
            ``True`` when the occupancy grid's entropy has fallen to the
            threshold, or when ``max_steps`` transitions have been taken.
        """
        values = np.asarray(state, dtype=np.float64)
        if int(round(values[STEP_INDEX])) >= self.max_steps:
            return True
        return self.entropy_bits(state) <= self.entropy_threshold_bits

    def initial_state_dist(self) -> Distribution:
        """A fresh map per episode, the robot at the known start pose.

        Returns:
            The map prior lifted into this environment's state layout.
        """
        return OccupancyGridInitialStateDistribution(
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            start_row=self.start_row,
            start_col=self.start_col,
            start_heading=self.start_heading,
            num_obstacles=self.num_obstacles,
            max_obstacle_size=self.max_obstacle_size,
            has_boundary_wall=self.has_boundary_wall,
            keep_free=resolve_keep_free(
                self.start_row, self.start_col, self.num_rows, self.num_cols
            ),
            state_size=self.state_size,
            map_offset=self.map_offset,
        )

    def initial_observation_dist(self) -> DiscreteDistribution:
        """The pre-scan reading: the known start pose and no ranges yet.

        Returns:
            A point mass on the start pose with zero range placeholders. This initial sentinel is never
            passed to the inverse map update and carries no map information.
        """
        observation = np.empty(3 + self.num_beams, dtype=np.float64)
        observation[0] = float(self.start_row)
        observation[1] = float(self.start_col)
        observation[2] = float(self.start_heading)
        observation[3:] = 0.0
        return DiscreteDistribution(values=[observation], probs=np.array([1.0]))

    # -- metrics --------------------------------------------------------

    def step_info(self, state: Any, action: Any, next_state: Any) -> Dict[str, float]:
        """Report the per-step channels this environment's metrics are built on.

        Draws no randomness: every channel is read off ``state``, ``action`` and
        ``next_state``, and the collision channel is decided by re-running the
        pose rule, which is deterministic.

        Map-progress channels are read from ``next_state`` when there is one,
        for the reason Battleship documents: the episode runner checks its step
        budget before it checks terminality, so an episode whose final allowed
        step resolves the map records no terminal bookkeeping step, and reading
        progress from ``state`` alone would score it as an unresolved timeout.

        Args:
            state: The state the step was taken from, or the final state on the
                terminal bookkeeping step.
            action: The action taken, or ``None`` on the terminal step.
            next_state: The realised successor, or ``None`` on the terminal step.

        Returns:
            The channels named by :class:`OccupancyGridStepChannel`. The two
            transition channels report ``0.0`` on the terminal step, where no
            action was taken.
        """
        collision = 0.0
        successful_translation = 0.0
        if action is not None and next_state is not None:
            row, col, heading = self.pose(state)
            _, _, _, blocked = self._next_pose(row, col, heading, int(action), self.true_map(state))
            next_row, next_col, _ = self.pose(next_state)
            collision = float(blocked)
            successful_translation = float((next_row, next_col) != (row, col))

        grid_state = state if next_state is None else next_state
        residual_entropy = self.entropy_bits(grid_state)
        resolved = float(residual_entropy <= self.entropy_threshold_bits)
        resolved_fraction = float(
            np.count_nonzero(np.abs(self.log_odds(grid_state)) >= RESOLVED_LOG_ODDS)
        ) / float(self.num_cells)

        return {
            OccupancyGridStepChannel.MAP_RESOLVED.value: resolved,
            OccupancyGridStepChannel.MAP_UNRESOLVED.value: 1.0 - resolved,
            # Nothing here can fail: the robot cannot be destroyed, and a wall it
            # drives into simply stops it. The channel is still emitted, as a
            # constant, because a declared-but-unreported channel is silently
            # dropped and a reader comparing the three end-reason rates needs
            # all three to sum to one.
            OccupancyGridStepChannel.EPISODE_FAILURE.value: 0.0,
            OccupancyGridStepChannel.RECORDED_STEP.value: 1.0,
            OccupancyGridStepChannel.RESIDUAL_ENTROPY_BITS.value: residual_entropy,
            OccupancyGridStepChannel.RESOLVED_CELL_FRACTION.value: resolved_fraction,
            OccupancyGridStepChannel.OBSTACLE_COLLISION.value: collision,
            OccupancyGridStepChannel.SUCCESSFUL_TRANSLATION.value: successful_translation,
        }

    def get_metric_specs(self) -> List[StepInfoMetric]:
        """Declare the occupancy-grid mapping metrics.

        Completion reduces with ``ANY``: resolving the map is something that
        happens once and stays true, since the entropy threshold, once crossed,
        also ends the episode. ``ended_by_*`` reduce with ``LAST``.

        The danger here is driving into something. It is reported as a count and
        not also as a severity: every collision is the same event -- the robot
        stops and the step buys nothing -- so a per-episode maximum would be the
        constant 1 for any episode with a collision and add nothing to the count.

        Returns:
            One spec per metric named in :class:`OccupancyGridMappingMetrics`.
        """
        return [
            StepInfoMetric(
                name=OccupancyGridMappingMetrics.TASK_COMPLETION_RATE.value,
                channel=OccupancyGridStepChannel.MAP_RESOLVED.value,
                per_episode=EpisodeReduction.ANY,
            ),
            StepInfoMetric(
                name=OccupancyGridMappingMetrics.ENDED_BY_GOAL.value,
                channel=OccupancyGridStepChannel.MAP_RESOLVED.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=OccupancyGridMappingMetrics.ENDED_BY_FAILURE.value,
                channel=OccupancyGridStepChannel.EPISODE_FAILURE.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=OccupancyGridMappingMetrics.ENDED_BY_TIMEOUT.value,
                channel=OccupancyGridStepChannel.MAP_UNRESOLVED.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=OccupancyGridMappingMetrics.AVERAGE_EPISODE_LENGTH.value,
                channel=OccupancyGridStepChannel.RECORDED_STEP.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=OccupancyGridMappingMetrics.FINAL_RESIDUAL_ENTROPY_BITS.value,
                channel=OccupancyGridStepChannel.RESIDUAL_ENTROPY_BITS.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=OccupancyGridMappingMetrics.MAX_RESOLVED_CELL_FRACTION.value,
                channel=OccupancyGridStepChannel.RESOLVED_CELL_FRACTION.value,
                per_episode=EpisodeReduction.MAX,
            ),
            StepInfoMetric(
                name=OccupancyGridMappingMetrics.AVERAGE_OBSTACLE_COLLISIONS.value,
                channel=OccupancyGridStepChannel.OBSTACLE_COLLISION.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=OccupancyGridMappingMetrics.AVERAGE_SUCCESSFUL_TRANSLATIONS.value,
                channel=OccupancyGridStepChannel.SUCCESSFUL_TRANSLATION.value,
                per_episode=EpisodeReduction.SUM,
            ),
        ]

    # -- pickling -------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """Drop the ray templates and the batched kernels before pickling.

        Both are derived read-only artifacts rebuilt in milliseconds, and this
        environment is shipped to every parallel worker once per task.
        """
        state = self.__dict__.copy()
        state["_ray_templates"] = None
        state.pop("_batched_kernels_cache", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore, rebuilding the ray templates."""
        vars(self).update(state)
        self._ray_templates = build_ray_templates(
            num_beams=self.num_beams,
            field_of_view_degrees=self.field_of_view_degrees,
            max_range_cells=self.max_range_cells,
        )

    # -- visualization --------------------------------------------------

    def cache_visualization(
        self, history: List[StepData], output_dir: Path, episode_index: int
    ) -> None:
        """Write the episode's animated GIF into ``output_dir``.

        Args:
            history: Episode history.
            output_dir: Directory to write into.
            episode_index: Zero-based episode index, used to name the file.
        """
        # Imported lazily: every parallel worker imports this module while
        # almost none of them render anything, so the renderer's textures,
        # sprites and palette tables stay out of a planning run's memory.
        # pylint: disable-next=import-outside-toplevel
        from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_visualizer import (  # noqa: E501
            OccupancyGridMappingVisualizer,
        )

        cache_path = output_dir / f"occupancy_grid_mapping_{episode_index}.gif"
        OccupancyGridMappingVisualizer(self).create_visualization(history, cache_path)


def create_occupancy_grid_state(
    environment: OccupancyGridMappingPOMDP,
    occupancy: Union[Sequence[float], np.ndarray],
    row: Optional[int] = None,
    col: Optional[int] = None,
    heading: Optional[int] = None,
    step: int = 0,
    log_odds: Optional[Union[Sequence[float], np.ndarray]] = None,
) -> OccupancyGridState:
    """Build a state vector for ``environment`` from its parts.

    Exists so tests and pinned scenarios can construct a specific world without
    reproducing the layout arithmetic, which is the kind of duplication that
    goes stale silently when the layout changes.

    Args:
        environment: The environment whose layout the state must match.
        occupancy: The true map, row-major, non-zero where occupied.
        row: Robot row. Defaults to the environment's start row.
        col: Robot column. Defaults to the environment's start column.
        heading: Heading index. Defaults to the environment's start heading.
        step: Step counter. Defaults to 0.
        log_odds: The occupancy grid, row-major. Defaults to all-unknown.

    Returns:
        A ``float64`` state vector of length ``environment.state_size``.
    """
    state = np.zeros(environment.state_size, dtype=np.float64)
    state[STEP_INDEX] = float(step)
    state[ROW_INDEX] = float(environment.start_row if row is None else row)
    state[COL_INDEX] = float(environment.start_col if col is None else col)
    state[HEADING_INDEX] = float(environment.start_heading if heading is None else heading)
    state[environment.map_offset : environment.map_offset + environment.num_cells] = np.asarray(
        occupancy, dtype=np.float64
    ).ravel()
    if log_odds is not None:
        end = environment.log_odds_offset + environment.num_cells
        state[environment.log_odds_offset : end] = np.asarray(log_odds, dtype=np.float64).ravel()
    return state
