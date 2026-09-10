# SPDX-License-Identifier: MIT

"""Occupancy-grid mapping and exploration as a POMDP.

A robot is dropped into a 2-D world it has never seen, carrying a range sensor
and nothing else. Its job is to *find out what the world looks like*: to drive
the uncertainty out of an occupancy grid over the world's cells. There is no
goal cell to reach and no object to find. The task is the map.

That makes the reward **belief-dependent** rather than state-dependent, which
is what distinguishes this environment from every other grid world here. The
agent is paid the *entropy reduction* of its occupancy grid, in bits -- the
information-gain exploration objective of Bourgault et al. (2002). Nothing about
the world changes when the robot drives; only what it knows does.

How that is expressed inside a state-based API
----------------------------------------------
The occupancy grid the robot accumulates is carried **inside the state**,
alongside the true map:

* the **true map** is hidden and constant for the whole episode -- this is the
  uncertainty a belief is actually over;
* the **occupancy grid**, held as log-odds, is a deterministic accumulation of
  the scans taken from the poses visited, so it is a legitimate state variable
  rather than bookkeeping the runner keeps on the side.

A belief particle therefore carries one hypothesised world *and* the map the
robot would have built if that world were the real one. Averaging the per-step
entropy reduction over the particles -- which every planner here does when it
averages rewards over a belief -- recovers exactly the expected information gain
the exploration literature maximises. No special belief-reward hook is needed
and no other environment's behaviour changes.

Log-odds, not probabilities, is the internal representation, for the standard
reason: the update is additive there, so a cell observed a hundred times is a
sum rather than a hundred multiplications of small numbers, and nothing
underflows. Probabilities appear only where entropy is computed and where the
visualizer draws.

State
-----
One ``float64`` vector, ``4 + 2 * num_cells`` long:

``[step, row, col, heading, true_occupancy(num_cells), log_odds(num_cells)]``

``heading`` is an index into ``0=N, 1=E, 2=S, 3=W``; ``row`` grows downwards.

Actions
-------
Three, discrete: ``0`` move forward one cell, ``1`` turn left, ``2`` turn right.
A forward move into an occupied cell or off the grid leaves the robot where it
was. This is the discrete formulation, chosen because every solver in this
repository that can consume a generative environment is a particle-based tree
search over a discrete action set; see the module's documentation page.

Observation
-----------
``[row, col, heading, range(num_beams)]``. The pose is reported exactly:
occupancy grid mapping is classically posed as *mapping with known poses*
(Moravec and Elfes 1985; Thrun, Burgard and Fox, ch. 9), and localisation is a
different problem that would change what this environment measures. The ranges
are the nominal ray-cast ranges plus zero-mean Gaussian noise.

The noisy ranges are **not** clipped back into ``[0, max_range_cells]``.
Clipping would put a point mass at each end that a Gaussian density cannot
represent, so the likelihood used to weight particles would stop matching the
sampler. ContinuousLightDark leaves its sampler unclipped for the same reason.

Range noise reaches the *likelihood* but never the *map update*: the log-odds
accumulated in the state are built from the nominal scan. Keeping the two apart
is what lets the transition stay a function of ``(state, action)`` while the
observation model stays a plain Gaussian; the approximation it buys is small,
because the inverse sensor model applies a fixed increment per cell and noise of
well under a cell rarely moves which cell that is.

Termination
-----------
Two ways, and the metrics tell them apart:

* the occupancy grid's total entropy falls to ``entropy_threshold_fraction`` of
  its initial value -- the map is resolved, the task is complete;
* ``max_steps`` transitions have been taken -- the budget ran out.

Classes:
    OccupancyGridAction: The three action indices.
    OccupancyGridStepChannel: Per-step measurement channels.
    OccupancyGridMappingMetrics: Metric names.
    OccupancyGridMappingPOMDP: The environment.
"""

from enum import Enum, IntEnum
from pathlib import Path
from collections.abc import Hashable
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_sensor import (
    HEADING_STEPS,
    NUM_HEADINGS,
    build_ray_templates,
    cast_scan,
    grid_entropy_bits,
    log_odds_from_probability,
    scan_log_odds_delta,
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
    VISITED_NEW_CELL = "visited_new_cell"


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
    AVERAGE_NEW_CELLS_VISITED = "average_new_cells_visited"


#: Type alias for an occupancy-grid mapping state.
OccupancyGridState = np.ndarray


class OccupancyGridMappingPOMDP(DiscreteActionsEnvironment):
    """Explore an unknown 2-D world by driving the entropy out of its map.

    Dynamics:
        Deterministic by default. ``FORWARD`` advances one cell along the
        heading unless the target cell is occupied in the episode's true map or
        lies off the grid, in which case the robot stays put. Turning is always
        possible. Setting ``move_failure_probability`` above zero makes an
        otherwise-successful forward move fail with that probability, which is
        the only source of transition noise.

    Observation model:
        A ray-cast scan, sampled from the true map, plus the robot's exact pose.
        Per beam the nominal range is the centre-to-centre distance to the first
        occupied cell the beam meets, or ``max_range_cells`` if it meets none
        before running out of range or leaving the grid. Gaussian noise of
        ``range_noise_std_cells`` is added to each nominal range.

    Reward:
        The expected reduction in the occupancy grid's binary entropy, in bits
        -- the information-gain exploration objective of Bourgault, Makarenko,
        Williams, Grocholsky and Durrant-Whyte, *Information based adaptive
        robotic exploration* (IROS 2002). Expected rather than realised, so the
        reward stays a function of ``(state, action)`` and
        :attr:`reward_requires_next_state` stays ``False``; with deterministic
        motion the two coincide exactly. ``step_cost`` is subtracted on every
        step and defaults to zero, leaving the pure information-gain objective.

        The gain can be negative: a cell already believed occupied that then
        receives free-space evidence moves back towards ``p = 0.5``, and the
        grid's entropy rises. The declared range covers that.

    Terminal:
        Entropy at or below ``entropy_threshold_fraction`` of the initial
        all-unknown entropy (the map is resolved), or ``max_steps`` transitions
        taken. There is no failure terminal -- bumping into a wall wastes a step
        and is counted, but does not end anything.

    Attributes:
        num_rows: Grid rows.
        num_cols: Grid columns.
        num_cells: ``num_rows * num_cols``.
        num_beams: Beams per scan.
        field_of_view_degrees: Angular width of the beam fan.
        max_range_cells: Sensor range, in cell widths.
        range_noise_std_cells: Standard deviation of the per-beam range noise.
        hit_probability: Inverse sensor model's ``p(occupied | beam stopped here)``.
        miss_probability: Inverse sensor model's ``p(occupied | beam passed through)``.
        log_odds_clamp: Symmetric bound the accumulated log-odds are held inside.
        num_obstacles: Rectangular blocks the map prior places per episode.
        max_obstacle_size: Largest block side length.
        has_boundary_wall: Whether the outer ring of cells is walled.
        start_row: Robot's start row.
        start_col: Robot's start column.
        start_heading: Robot's start heading index.
        move_failure_probability: Chance an otherwise-valid forward move fails.
        max_steps: Transitions allowed before the episode is out of budget.
        entropy_threshold_fraction: Fraction of the initial entropy at or below
            which the map counts as resolved.
        step_cost: Constant charge per step.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> env = OccupancyGridMappingPOMDP()
        >>> state = env.initial_state_dist().sample()[0]
        >>> env.is_terminal(state)
        False
        >>> next_state, observation, reward = env.sample_next_step(
        ...     state, OccupancyGridAction.FORWARD
        ... )
        >>> observation.shape
        (27,)
        >>> bool(reward > 0.0)  # the first scan resolves cells, so it pays
        True
    """

    # pylint: disable-next=too-many-arguments,too-many-locals,too-many-statements
    def __init__(
        self,
        num_rows: int = 10,
        num_cols: int = 10,
        num_beams: int = 24,
        field_of_view_degrees: float = 360.0,
        max_range_cells: float = 3.5,
        range_noise_std_cells: float = 0.35,
        hit_probability: float = 0.85,
        miss_probability: float = 0.15,
        log_odds_clamp: float = 6.0,
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
                0.35 cells -- a tenth of the sensor's range, big enough that the
                likelihood separates hypotheses softly rather than killing
                every disagreeing particle outright.
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
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        if not 0.0 <= entropy_threshold_fraction <= 1.0:
            raise ValueError(
                "entropy_threshold_fraction must be in [0, 1], got "
                f"{entropy_threshold_fraction}"
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
        self.state_size = POSE_WIDTH + 2 * num_cells
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

    def _state_after_scan(
        self, state: OccupancyGridState, row: int, col: int, heading: int
    ) -> OccupancyGridState:
        """Return ``state`` advanced one step to the given pose, scan applied.

        Args:
            state: The state being stepped from.
            row: The pose's row.
            col: The pose's column.
            heading: The pose's heading index.

        Returns:
            A fresh state vector: step incremented, pose set, and the occupancy
            grid updated by the inverse sensor model for a scan taken from the
            new pose against this state's true map.
        """
        next_state = np.array(state, dtype=np.float64, copy=True)
        next_state[STEP_INDEX] += 1.0
        next_state[ROW_INDEX] = float(row)
        next_state[COL_INDEX] = float(col)
        next_state[HEADING_INDEX] = float(heading)

        occupancy = self.true_map(state)
        template = self._ray_templates[heading]
        _, hit, stop_slot = cast_scan(
            occupancy=occupancy,
            row=row,
            col=col,
            template=template,
            max_range_cells=self.max_range_cells,
        )
        delta = scan_log_odds_delta(
            hit=hit,
            stop_slot=stop_slot,
            template=template,
            row=row,
            col=col,
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            free_log_odds=self.free_log_odds,
            occupied_log_odds=self.occupied_log_odds,
        )
        # The robot occupies the cell it is standing in, so that cell is free by
        # direct evidence. No beam ever reports it -- every ray template starts
        # one cell out -- so without this the robot's own trail would stay at
        # p = 0.5 forever and the map could never be fully resolved.
        delta[row * self.num_cols + col] += self.free_log_odds

        end = self.log_odds_offset + self.num_cells
        np.clip(
            next_state[self.log_odds_offset : end] + delta,
            -self.log_odds_clamp,
            self.log_odds_clamp,
            out=next_state[self.log_odds_offset : end],
        )
        return next_state

    def _transition_outcomes(
        self, state: OccupancyGridState, action: int
    ) -> Tuple[List[OccupancyGridState], np.ndarray]:
        """Enumerate the successors of ``(state, action)`` with their probabilities.

        There are at most two: the move succeeds, or ``move_failure_probability``
        makes it fail and the robot stays where it is while still taking a scan.
        Turning and a blocked forward move have a single outcome.

        Args:
            state: The state being stepped from.
            action: The action taken.

        Returns:
            ``(successors, probabilities)`` with the probabilities summing to 1.
        """
        row, col, heading = self.pose(state)
        occupancy = self.true_map(state)
        next_row, next_col, next_heading, blocked = self._next_pose(
            row, col, heading, action, occupancy
        )
        moved = (next_row, next_col) != (row, col)
        if not moved or self.move_failure_probability <= 0.0:
            del blocked
            return [self._state_after_scan(state, next_row, next_col, next_heading)], np.array(
                [1.0]
            )
        return (
            [
                self._state_after_scan(state, next_row, next_col, next_heading),
                self._state_after_scan(state, row, col, heading),
            ],
            np.array([1.0 - self.move_failure_probability, self.move_failure_probability]),
        )

    def sample_next_state(
        self, state: OccupancyGridState, action: int, n_samples: int = 1
    ) -> Any:
        """Move the robot, take a scan and fold it into the occupancy grid.

        Args:
            state: Current state.
            action: One of :class:`OccupancyGridAction`.
            n_samples: How many successors to draw. Defaults to 1.

        Returns:
            A single ``float64`` state when ``n_samples == 1``, else an
            ``(n_samples, state_size)`` ``float64`` array.
        """
        successors, probabilities = self._transition_outcomes(state, action)
        if n_samples == 1:
            if len(successors) == 1:
                return successors[0]
            index = int(np.random.choice(len(successors), p=probabilities))
            return successors[index]
        count = int(n_samples)
        if len(successors) == 1:
            return np.tile(successors[0], (count, 1))
        indices = np.random.choice(len(successors), size=count, p=probabilities)
        return np.stack([successors[int(index)] for index in indices])

    def transition_log_probability(
        self, state: OccupancyGridState, action: int, next_states: Any
    ) -> np.ndarray:
        """Log-probability of each candidate successor.

        Args:
            state: Current state.
            action: The action taken.
            next_states: Candidate successors.

        Returns:
            ``(N,)`` ``float64`` array; ``-inf`` for anything the transition
            cannot produce.
        """
        successors, probabilities = self._transition_outcomes(state, action)
        candidates = np.asarray(next_states, dtype=np.float64)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
        log_probs = np.full(candidates.shape[0], -np.inf, dtype=np.float64)
        for successor, probability in zip(successors, probabilities):
            matches = np.all(np.isclose(candidates, successor, atol=1e-9), axis=1)
            log_probs[matches] = float(np.log(probability)) if probability > 0.0 else -np.inf
        return log_probs

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

    def sample_observation(
        self, next_state: OccupancyGridState, action: int, n_samples: int = 1
    ) -> Any:
        """Report the robot's pose exactly and its ranges with Gaussian noise.

        Args:
            next_state: The post-transition state.
            action: The action taken. Unused: the sensor reads the world, not
                the manoeuvre that reached it.
            n_samples: How many readings to draw. Defaults to 1.

        Returns:
            ``(3 + num_beams,)`` ``float64`` when ``n_samples == 1``, else
            ``(n_samples, 3 + num_beams)``.
        """
        del action
        row, col, heading = self.pose(next_state)
        nominal = self.nominal_scan(next_state)
        count = int(n_samples)
        noise = np.random.normal(0.0, self.range_noise_std_cells, size=(count, self.num_beams))
        observations = np.empty((count, 3 + self.num_beams), dtype=np.float64)
        observations[:, 0] = float(row)
        observations[:, 1] = float(col)
        observations[:, 2] = float(heading)
        observations[:, 3:] = nominal[None, :] + noise
        if n_samples == 1:
            return observations[0]
        return observations

    def observation_log_probability(
        self, next_state: OccupancyGridState, action: int, observations: Any
    ) -> np.ndarray:
        """Log-likelihood of each candidate reading under ``next_state``.

        The pose block is reported exactly, so a candidate whose pose disagrees
        with ``next_state`` has zero likelihood -- that is what lets a particle
        filter reject a hypothesised map under which the robot's forward move
        would have been blocked when in fact it was not. The range block is a
        product of independent Gaussians about the nominal scan.

        Args:
            next_state: The post-transition state.
            action: The action taken. Unused.
            observations: Candidate readings.

        Returns:
            ``(N,)`` ``float64`` array of log-densities.
        """
        del action
        row, col, heading = self.pose(next_state)
        nominal = self.nominal_scan(next_state)
        candidates = np.asarray(observations, dtype=np.float64)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)

        pose_matches = (
            (np.abs(candidates[:, 0] - float(row)) < 0.5)
            & (np.abs(candidates[:, 1] - float(col)) < 0.5)
            & (np.abs(candidates[:, 2] - float(heading)) < 0.5)
        )
        residuals = candidates[:, 3:] - nominal[None, :]
        variance = self.range_noise_std_cells**2
        log_density = -0.5 * np.sum(residuals**2, axis=1) / variance - 0.5 * self.num_beams * (
            np.log(2.0 * np.pi) + np.log(variance)
        )
        return np.where(pose_matches, log_density, -np.inf)

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

    def reward(
        self, state: OccupancyGridState, action: int, next_state: Any = None
    ) -> float:
        """Expected reduction in the occupancy grid's entropy, in bits.

        This is the information-gain exploration reward of Bourgault,
        Makarenko, Williams, Grocholsky and Durrant-Whyte, *Information based
        adaptive robotic exploration* (IROS 2002): the agent is paid for what it
        learns about the map, and for nothing else.

        The expectation is taken over the transition's outcomes rather than
        scored against the realised one, which keeps the reward a pure function
        of ``(state, action)`` and lets :attr:`reward_requires_next_state` stay
        ``False``. Under the default deterministic motion there is one outcome
        and the two definitions coincide exactly.

        Args:
            state: The state the action is taken from.
            action: The action taken.
            next_state: Unused; see above.

        Returns:
            Entropy reduction in bits, minus ``step_cost``. Negative when the
            scan pushes cells back towards ``p = 0.5``.
        """
        del next_state
        successors, probabilities = self._transition_outcomes(state, action)
        before = self.entropy_bits(state)
        gain = sum(
            float(probability) * (before - self.entropy_bits(successor))
            for successor, probability in zip(successors, probabilities)
        )
        return float(gain) - self.step_cost

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
            A point mass on the start pose with every range reported at the
            sensor's maximum, which is what "nothing detected" means here and
            carries no information about the map.
        """
        observation = np.empty(3 + self.num_beams, dtype=np.float64)
        observation[0] = float(self.start_row)
        observation[1] = float(self.start_col)
        observation[2] = float(self.start_heading)
        observation[3:] = self.max_range_cells
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
        visited_new_cell = 0.0
        if action is not None and next_state is not None:
            row, col, heading = self.pose(state)
            _, _, _, blocked = self._next_pose(
                row, col, heading, int(action), self.true_map(state)
            )
            next_row, next_col, _ = self.pose(next_state)
            collision = float(blocked)
            visited_new_cell = float((next_row, next_col) != (row, col))

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
            OccupancyGridStepChannel.VISITED_NEW_CELL.value: visited_new_cell,
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
                name=OccupancyGridMappingMetrics.AVERAGE_NEW_CELLS_VISITED.value,
                channel=OccupancyGridStepChannel.VISITED_NEW_CELL.value,
                per_episode=EpisodeReduction.SUM,
            ),
        ]

    # -- pickling -------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """Drop the ray templates before pickling.

        They are a derived read-only artifact rebuilt in milliseconds, and this
        environment is shipped to every parallel worker once per task.
        """
        state = self.__dict__.copy()
        state["_ray_templates"] = None
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
        # Imported lazily: matplotlib is heavy and every parallel worker imports
        # this module, while almost none of them render anything.
        # pylint: disable-next=import-outside-toplevel
        from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_visualizer import (  # noqa: E501
            OccupancyGridMappingVisualizer,
        )

        cache_path = output_dir / f"occupancy_grid_mapping_{episode_index}.gif"
        OccupancyGridMappingVisualizer(self).create_visualization(history, cache_path)


def create_occupancy_grid_state(
    environment: OccupancyGridMappingPOMDP,
    occupancy: Sequence[float],
    row: Optional[int] = None,
    col: Optional[int] = None,
    heading: Optional[int] = None,
    step: int = 0,
    log_odds: Optional[Sequence[float]] = None,
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
    state[HEADING_INDEX] = float(
        environment.start_heading if heading is None else heading
    )
    state[environment.map_offset : environment.map_offset + environment.num_cells] = np.asarray(
        occupancy, dtype=np.float64
    ).ravel()
    if log_odds is not None:
        end = environment.log_odds_offset + environment.num_cells
        state[environment.log_odds_offset : end] = np.asarray(
            log_odds, dtype=np.float64
        ).ravel()
    return state
