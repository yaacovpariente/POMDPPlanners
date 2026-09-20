# SPDX-License-Identifier: MIT

"""Several robots fight a wind-driven grid fire they can only see nearby.

``N`` firefighting robots stand on an ``R x C`` grid. Every cell is one of five
categories -- unburnt, smoldering, burning, burnt, wet -- and the fire spreads
from alight cells into their unburnt neighbours under a **hidden wind** that is
drawn uniformly from eight values at reset and is constant for the episode. The
wind is never observed. It has to be inferred from which neighbours catch.

The robots are driven by one centralized joint action: a single integer whose
base-5 digits are the per-robot actions, four moves and SUPPRESS. Suppression
soaks the robot's own cell and its four neighbours, so a careful planner fights
from an adjacent cell and takes no damage while a careless one stands in the
fire. Suppressant is finite and refilled by stepping on the depot; health is
finite and spent by standing in fire. A robot at zero health is disabled and
its digit of the joint action is ignored.

Each robot sees the exact poses, tanks and healths of all of them, plus a noisy
category for every cell within Chebyshev radius ``rho`` of any live robot.
Cells nobody is looking at report an unknown marker. The task is complete when
no cell is smoldering or burning. Burnt and wet are absorbing and no rule maps
either back into an alight category, so a fire-free map cannot be undone and
the goal is a genuine terminal state.

State layout, one ``float64`` vector of length ``3 + 4N + R*C``::

    [step, (row, col, tank, health) x N, wind_direction, wind_strength, cells]

This environment has no torch vectorized model and no C++ native model, so it
cannot be run under VOPP and is deliberately absent from the vectorized config
contract. ``PFT_DPW`` takes the scalar API directly and is what it is validated
with.

Classes:
    MultiAgentFirefightingPOMDP: The environment.
    MultiAgentFirefightingMetrics: Its metric names.
    MultiAgentFirefightingStepChannel: Its per-step channel names.
"""

# pylint: disable=too-many-lines  # one environment, its dynamics and its metrics

from enum import Enum
from pathlib import Path
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

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
from POMDPPlanners.environments.multiagent_firefighting_pomdp.multiagent_firefighting_world import (
    DIRECTION_OFFSETS,
    HEAT_DAMAGE,
    MAX_HEAT_DAMAGE_PER_STEP,
    NUM_CATEGORIES,
    NUM_ROBOT_ACTIONS,
    NUM_WIND_VALUES,
    ROBOT_FIELD_WIDTH,
    ROBOT_OFFSET,
    STEP_INDEX,
    FireCategory,
    FirefightingAction,
    MultiAgentFirefightingInitialStateDistribution,
    WindDirection,
    WindStrength,
    default_depot_cell,
    default_obstacle_cells,
    default_robot_start_cells,
    resolve_cells,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from POMDPPlanners.core.simulation.traces import EpisodeTrace

#: The category an unobserved cell reports. Encoded as a number because the
#: observation is one flat ``float64`` vector of fixed shape whatever the robots
#: do, which is what lets a particle filter compare two observations elementwise.
UNKNOWN_CATEGORY = -1.0

#: Type alias for a multi-agent firefighting state.
FirefightingState = np.ndarray


class MultiAgentFirefightingStepChannel(Enum):
    """Per-step channels reported by :meth:`MultiAgentFirefightingPOMDP.step_info`."""

    FIRE_EXTINGUISHED = "fire_extinguished"
    ALL_ROBOTS_DISABLED = "all_robots_disabled"
    TIMED_OUT_WITH_FIRE = "timed_out_with_fire"
    RECORDED_STEP = "recorded_step"
    ROBOT_IN_ALIGHT_CELL = "robot_in_alight_cell"
    HEALTH_LOST = "health_lost"
    SUPPRESS_ACTIONS = "suppress_actions"
    ALIGHT_CELLS = "alight_cells"
    BURNT_CELL_FRACTION = "burnt_cell_fraction"
    ROBOTS_DISABLED = "robots_disabled"


class MultiAgentFirefightingMetrics(Enum):
    """Metric names for the multi-agent firefighting environment."""

    TASK_COMPLETION_RATE = "task_completion_rate"
    ENDED_BY_GOAL = "ended_by_goal"
    ENDED_BY_FAILURE = "ended_by_failure"
    ENDED_BY_TIMEOUT = "ended_by_timeout"
    AVERAGE_EPISODE_LENGTH = "average_episode_length"
    ROBOT_STEPS_IN_FIRE = "robot_steps_in_fire"
    ROBOT_HEALTH_LOST = "robot_health_lost"
    SUPPRESSANT_UNITS_USED = "suppressant_units_used"
    MAX_SIMULTANEOUS_ALIGHT_CELLS = "max_simultaneous_alight_cells"
    MAX_BURNT_CELL_FRACTION = "max_burnt_cell_fraction"
    ROBOTS_DISABLED_AT_END = "robots_disabled_at_end"


# pylint: disable-next=too-many-public-methods,too-many-instance-attributes
class MultiAgentFirefightingPOMDP(DiscreteActionsEnvironment):
    """Put out a wind-driven grid fire with ``N`` partially sighted robots.

    The episode ends when no cell is alight (goal), when every robot is
    disabled while fire is still active (failure, when
    ``is_all_robots_disabled_terminal``), or when the step budget runs out
    (timeout). Goal wins over failure: a fire put out by robots that then
    burned out is still a success.
    """

    # pylint: disable-next=too-many-arguments,too-many-locals,too-many-statements,too-many-branches
    def __init__(
        self,
        num_rows: int = 10,
        num_cols: int = 10,
        num_robots: int = 2,
        obstacle_cells: Optional[List[Tuple[int, int]]] = None,
        depot_cell: Optional[Tuple[int, int]] = None,
        robot_start_cells: Optional[List[Tuple[int, int]]] = None,
        num_initial_fires: int = 1,
        max_tank: int = 6,
        max_health: int = 3,
        sensing_radius: int = 2,
        observation_error_probability: float = 0.1,
        slip_probability: float = 0.05,
        spread_probability: float = 0.10,
        wind_gain_low: float = 2.0,
        wind_gain_high: float = 3.5,
        crosswind_attenuation: float = 0.5,
        growth_probability: float = 0.35,
        burnout_probability: float = 0.03,
        suppression_probability_unburnt: float = 1.0,
        suppression_probability_smoldering: float = 0.9,
        suppression_probability_burning: float = 0.6,
        max_steps: int = 100,
        success_reward: float = 100.0,
        step_cost: float = 0.1,
        smoldering_cell_cost: float = 0.5,
        burning_cell_cost: float = 1.0,
        burnt_cell_cost: float = 5.0,
        damage_cost: float = 10.0,
        water_cost: float = 0.1,
        is_all_robots_disabled_terminal: bool = True,
        discount_factor: float = 0.95,
        name: str = "MultiAgentFirefighting",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the multi-agent firefighting POMDP.

        Args:
            num_rows: Grid rows. Defaults to 10.
            num_cols: Grid columns. Defaults to 10. At 10x10 the whole map is
                100 cells, small enough that a particle carries a whole world
                cheaply and large enough that two sensing footprints of 25
                cells each leave most of it unseen.
            num_robots: How many robots. Defaults to 2. This sets the size of
                the joint action space, ``5 ** num_robots``: 25 at the default,
                125 at three robots, which is where a tree search starts to
                feel the branching.
            obstacle_cells: Cells that are never enterable and never ignitable.
                Defaults to ``None``, which places one small square blob just
                past the middle of the grid. Pass ``[]`` for an open grid --
                that is a different thing from ``None``.
            depot_cell: The cell that refills a tank on entry. Defaults to
                ``None``, which puts it on the first non-obstacle cell in
                row-major order, i.e. the north-west corner.
            robot_start_cells: Where the robots begin. Defaults to ``None``,
                which seats them on the first free cells scanned from a quarter
                of the way into the grid -- ``(2, 2)`` and ``(2, 3)`` at the
                default size.
            num_initial_fires: Cells alight at reset, drawn uniformly without
                replacement from the non-obstacle cells and set to ``BURNING``.
                Defaults to 1. Raising it makes the fire harder to contain and
                makes splitting the robots up the better policy.
            max_tank: Tank capacity, and the tank every robot starts with.
                Defaults to 6: how many sprays a robot gets between depot
                trips, and therefore how often logistics interrupts
                firefighting.
            max_health: Health capacity, and the health every robot starts
                with. Defaults to 3, so a robot survives one burning step and
                is disabled by the second.
            sensing_radius: Chebyshev sensing radius. Defaults to 2, so each
                robot reports the 5x5 block centred on itself.
            observation_error_probability: Chance a sensed cell is reported as
                the wrong category, the wrong mass spread evenly over the other
                four. Defaults to 0.1. At 0 the fire map is exact inside the
                footprints and only the wind and the unseen cells stay hidden;
                at 0.8 a reading carries nothing. Both extremes are legal and
                both make a poor task.
            slip_probability: Chance an otherwise admissible move fails and the
                robot stays put, standing in for smoke and debris. Defaults to
                0.05.
            spread_probability: Base chance that one alight neighbour ignites a
                cell in one step, before wind. Defaults to 0.10. This is the
                single number that decides how fast an unattended fire grows,
                and together with ``burnout_probability`` it is what makes the
                completion metric mean anything. At the originally proposed
                0.06 with a burnout of 0.12, an unattended fire on the default
                world went out by itself in 93% of episodes: a planner that did
                nothing would have "completed the task" nine times in ten, and
                no margin over a random baseline would have been measurable.
                At 0.10 with a burnout of 0.03 the same unattended fire goes
                out 6% of the time, a random policy 27%, and a hand-written
                greedy firefighter 89% -- so the number now reports what the
                planner did.
            wind_gain_low: Downwind multiplier at low wind strength. Defaults
                to 2.0.
            wind_gain_high: Downwind multiplier at high wind strength. Defaults
                to 3.5. The gain applies only to the one neighbour sitting
                directly upwind of a cell.
            crosswind_attenuation: How much the base rate is cut in the other
                three directions; they spread at ``1 - crosswind_attenuation``
                of it. Defaults to 0.5. At 0 the wind helps downwind but
                hinders nowhere; near 1 the fire becomes a directed front.
                Together with the gains this is what makes the wind
                identifiable at all -- with no asymmetry the hidden wind would
                be unobservable noise rather than something to infer.
            growth_probability: Chance a smoldering cell grows to burning in
                one step. Defaults to 0.35. It sets how long the robots have to
                reach a new ignition while it is still cheap to put out.
            burnout_probability: Chance a burning cell burns out to ``BURNT``
                in one step. Defaults to 0.03, so a burning cell lives about
                thirty steps. This is the parameter that decides whether a fire
                ever dies on its own, and a value that lets it self-extinguish
                makes the completion rate meaningless -- see
                ``spread_probability`` for the measurements that moved this
                from the originally proposed 0.12.
            suppression_probability_unburnt: Chance one spray turns an unburnt
                cell wet. Defaults to 1.0 -- pre-wetting a firebreak always
                works, which is how a break gets built ahead of the front.
            suppression_probability_smoldering: Chance one spray turns a
                smoldering cell wet. Defaults to 0.9.
            suppression_probability_burning: Chance one spray turns a burning
                cell wet. Defaults to 0.6: a developed fire resists one hose,
                which is why two robots spraying the same cell is worth more
                than two robots dividing the work.
            max_steps: Transitions allowed per episode. Defaults to 100.
            success_reward: Paid once, on the transition into a fire-free
                state. Defaults to 100.0.
            step_cost: Charged on every transition, including the terminating
                one. Defaults to 0.1.
            smoldering_cell_cost: Cost per smoldering cell in the successor.
                Defaults to 0.5.
            burning_cell_cost: Cost per burning cell in the successor. Defaults
                to 1.0. Together with the line above, this is the pressure to
                shrink the fire rather than sit safely beside it.
            burnt_cell_cost: Cost per cell destroyed this step. Defaults to
                5.0. This is the property the robots are there to save.
            damage_cost: Cost per point of robot health lost. Defaults to 10.0,
                deliberately above ``burnt_cell_cost`` so a planner does not
                trade a robot for a cell.
            water_cost: Cost per spray. Defaults to 0.1: small, just enough to
                stop spraying at nothing.
            is_all_robots_disabled_terminal: Whether an episode ends when every
                robot is disabled and fire is still active. Defaults to
                ``True``. This changes only termination, never the reward, so
                it does not move either end of the declared reward range.
            discount_factor: Discount factor. Defaults to 0.95.
            name: Environment name. Defaults to ``"MultiAgentFirefighting"``.
            output_dir: Output directory for logging. Defaults to ``None``.
            debug: Enable debug logging. Defaults to ``False``.
            use_queue_logger: Whether to use queue-based logging.

        Raises:
            ValueError: If the geometry, the probabilities, the capacities or
                the episode limits are outside their valid ranges.
        """
        if num_rows < 1 or num_cols < 1:
            raise ValueError(f"grid must be at least 1x1, got {num_rows}x{num_cols}")
        if num_robots < 1:
            raise ValueError(f"num_robots must be at least 1, got {num_robots}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        if max_tank < 0:
            raise ValueError(f"max_tank must be non-negative, got {max_tank}")
        if max_health < 1:
            raise ValueError(
                f"max_health must be at least 1, or every robot starts disabled, got {max_health}"
            )
        if sensing_radius < 0:
            raise ValueError(f"sensing_radius must be non-negative, got {sensing_radius}")
        for label, probability in (
            ("observation_error_probability", observation_error_probability),
            ("slip_probability", slip_probability),
            ("spread_probability", spread_probability),
            ("crosswind_attenuation", crosswind_attenuation),
            ("growth_probability", growth_probability),
            ("burnout_probability", burnout_probability),
            ("suppression_probability_unburnt", suppression_probability_unburnt),
            ("suppression_probability_smoldering", suppression_probability_smoldering),
            ("suppression_probability_burning", suppression_probability_burning),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{label} must be in [0, 1], got {probability}")
        if wind_gain_low < 0.0 or wind_gain_high < 0.0:
            raise ValueError(
                "wind gains must be non-negative, got "
                f"low={wind_gain_low}, high={wind_gain_high}"
            )
        for label, cost in (
            ("success_reward", success_reward),
            ("step_cost", step_cost),
            ("smoldering_cell_cost", smoldering_cell_cost),
            ("burning_cell_cost", burning_cell_cost),
            ("burnt_cell_cost", burnt_cell_cost),
            ("damage_cost", damage_cost),
            ("water_cost", water_cost),
        ):
            if cost < 0.0:
                raise ValueError(f"{label} must be non-negative, got {cost}")

        num_cells = int(num_rows) * int(num_cols)
        obstacles = resolve_cells(obstacle_cells, default_obstacle_cells(num_rows, num_cols))
        for row, col in obstacles:
            if not 0 <= row < int(num_rows) or not 0 <= col < int(num_cols):
                raise ValueError(f"obstacle ({row}, {col}) is outside the grid")
        depot = (
            default_depot_cell(num_rows, num_cols, obstacles)
            if depot_cell is None
            else (int(depot_cell[0]), int(depot_cell[1]))
        )
        if not 0 <= depot[0] < int(num_rows) or not 0 <= depot[1] < int(num_cols):
            raise ValueError(f"depot {depot} is outside the grid")
        if depot in set(obstacles):
            raise ValueError(f"depot {depot} is an obstacle")
        # The default is built only when it is needed. It refuses to seat a
        # robot on the depot, which explicit starts are allowed to do, so
        # computing it eagerly would let a perfectly legal explicit layout be
        # rejected by a placement rule nothing was going to use.
        starts = (
            default_robot_start_cells(num_rows, num_cols, num_robots, obstacles, depot)
            if robot_start_cells is None
            else [(int(row), int(col)) for row, col in robot_start_cells]
        )
        if len(starts) != int(num_robots):
            raise ValueError(f"robot_start_cells has {len(starts)} cells for {num_robots} robots")
        for row, col in starts:
            if not 0 <= row < int(num_rows) or not 0 <= col < int(num_cols):
                raise ValueError(f"robot start ({row}, {col}) is outside the grid")
            if (row, col) in set(obstacles):
                raise ValueError(f"robot start ({row}, {col}) is an obstacle")
        num_ignitable = num_cells - len(set(obstacles))
        if not 1 <= int(num_initial_fires) <= num_ignitable:
            raise ValueError(
                f"num_initial_fires must be in [1, {num_ignitable}] so the goal is neither "
                f"free at reset nor impossible to set up, got {num_initial_fires}"
            )

        # -- Reward range, enumerated rather than estimated --------------
        #
        # Maximum: the only positive term is the success bonus, paid once on
        # the transition into a fire-free state, and the step cost is charged
        # on that transition too. Every other term is a non-negative quantity
        # entering with a minus sign, so nothing can push a reward above
        # ``success_reward - step_cost``. It is not ``success_reward``.
        #
        # Minimum: the step cost is always charged; the water cost is at most
        # one spray per robot; damage is capped per robot per step at the
        # burning-cell damage, and also by the health the robot has, hence the
        # ``min``. The three map terms -- smoldering cells, burning cells and
        # cells newly burnt this step -- look as though they stack, but they
        # cannot: in the successor a cell is smoldering, or burning, or newly
        # burnt, or none of those, never two at once. They therefore share one
        # budget of ``num_cells`` cells, and the worst case is every cell
        # charged at whichever of the three coefficients is largest. That is
        # what allows ``max(...)`` here instead of a sum, and getting this
        # wrong in the loose direction hides a real bug while getting it wrong
        # in the tight direction fails the conformance suite.
        #
        # ``is_all_robots_disabled_terminal`` changes only termination, never
        # any reward term, so it moves neither end of this bound.
        max_reward = float(success_reward) - float(step_cost)
        min_reward = -(
            float(step_cost)
            + max(float(smoldering_cell_cost), float(burning_cell_cost), float(burnt_cell_cost))
            * float(num_cells)
            + float(damage_cost)
            * float(num_robots)
            * float(min(int(max_health), MAX_HEAT_DAMAGE_PER_STEP))
            + float(water_cost) * float(num_robots)
        )

        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
            ),
            reward_range=(min_reward, max_reward),
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        self.num_rows = int(num_rows)
        self.num_cols = int(num_cols)
        self.num_robots = int(num_robots)
        self.obstacle_cells = obstacles
        self.depot_cell = depot
        self.robot_start_cells = starts
        self.num_initial_fires = int(num_initial_fires)
        self.max_tank = int(max_tank)
        self.max_health = int(max_health)
        self.sensing_radius = int(sensing_radius)
        self.observation_error_probability = float(observation_error_probability)
        self.slip_probability = float(slip_probability)
        self.spread_probability = float(spread_probability)
        self.wind_gain_low = float(wind_gain_low)
        self.wind_gain_high = float(wind_gain_high)
        self.crosswind_attenuation = float(crosswind_attenuation)
        self.growth_probability = float(growth_probability)
        self.burnout_probability = float(burnout_probability)
        self.suppression_probability_unburnt = float(suppression_probability_unburnt)
        self.suppression_probability_smoldering = float(suppression_probability_smoldering)
        self.suppression_probability_burning = float(suppression_probability_burning)
        self.max_steps = int(max_steps)
        self.success_reward = float(success_reward)
        self.step_cost = float(step_cost)
        self.smoldering_cell_cost = float(smoldering_cell_cost)
        self.burning_cell_cost = float(burning_cell_cost)
        self.burnt_cell_cost = float(burnt_cell_cost)
        self.damage_cost = float(damage_cost)
        self.water_cost = float(water_cost)
        self.is_all_robots_disabled_terminal = bool(is_all_robots_disabled_terminal)

        self.num_cells = num_cells
        self.num_actions = NUM_ROBOT_ACTIONS**self.num_robots
        self.wind_direction_index = ROBOT_OFFSET + ROBOT_FIELD_WIDTH * self.num_robots
        self.wind_strength_index = self.wind_direction_index + 1
        self.fire_offset = self.wind_strength_index + 1
        self.state_size = self.fire_offset + self.num_cells
        self.observation_size = ROBOT_FIELD_WIDTH * self.num_robots + self.num_cells

        # Derived read-only tables. Underscored so they stay out of
        # ``config_id`` and ``__eq__``: every one is a pure function of
        # settings already in the identity, and putting a 125x3 action table
        # or a whole boolean grid into a cache key would be slow and redundant.
        self._obstacle_mask = np.zeros((self.num_rows, self.num_cols), dtype=bool)
        for row, col in self.obstacle_cells:
            self._obstacle_mask[row, col] = True
        self._action_table = self._build_action_table()
        self._suppression_probabilities = np.array(
            [
                self.suppression_probability_unburnt,
                self.suppression_probability_smoldering,
                self.suppression_probability_burning,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        self._heat_damage = np.asarray(HEAT_DAMAGE, dtype=np.int64)
        self._ignitable_indices = np.flatnonzero(~self._obstacle_mask.ravel())

    # -- construction helpers -------------------------------------------

    def _build_action_table(self) -> np.ndarray:
        """Return the ``(num_actions, num_robots)`` base-5 decoding table.

        Robot ``i`` is the ``i``-th base-5 digit of the joint action, least
        significant first, which is the convention the formal definition uses.

        Built with array arithmetic rather than a nested Python loop, because
        the table has ``5 ** num_robots`` rows: at two robots either is
        instant, but the loop is quadratic in a number that is already
        exponential, and at six or seven robots it would add seconds to every
        construction -- including the one every parallel worker does.

        Returns:
            ``int64`` array whose row ``a`` holds the per-robot actions of
            joint action ``a``.
        """
        joint = np.arange(self.num_actions, dtype=np.int64)[:, None]
        place = NUM_ROBOT_ACTIONS ** np.arange(self.num_robots, dtype=np.int64)[None, :]
        return (joint // place) % NUM_ROBOT_ACTIONS

    # -- state accessors ------------------------------------------------

    def step_count(self, state: FirefightingState) -> int:
        """Return the step counter of ``state``.

        Args:
            state: A state vector.

        Returns:
            How many transitions have been taken.
        """
        return int(round(float(np.asarray(state, dtype=np.float64)[STEP_INDEX])))

    def robots(self, state: FirefightingState) -> np.ndarray:
        """Return the per-robot block of ``state`` as integers.

        Args:
            state: A state vector.

        Returns:
            ``(num_robots, 4)`` ``int64`` array of row, column, tank, health.
        """
        values = np.asarray(state, dtype=np.float64)
        block = values[ROBOT_OFFSET : ROBOT_OFFSET + ROBOT_FIELD_WIDTH * self.num_robots]
        return np.rint(block).astype(np.int64).reshape(self.num_robots, ROBOT_FIELD_WIDTH)

    def wind(self, state: FirefightingState) -> Tuple[int, int]:
        """Return the hidden wind of ``state``.

        Args:
            state: A state vector.

        Returns:
            ``(direction, strength)`` as the integer codes of
            :class:`~...multiagent_firefighting_world.WindDirection` and
            :class:`~...multiagent_firefighting_world.WindStrength`.
        """
        values = np.asarray(state, dtype=np.float64)
        return (
            int(round(float(values[self.wind_direction_index]))),
            int(round(float(values[self.wind_strength_index]))),
        )

    def fire_map(self, state: FirefightingState) -> np.ndarray:
        """Return the fire map of ``state`` as a grid of category codes.

        Args:
            state: A state vector.

        Returns:
            ``(num_rows, num_cols)`` ``int64`` array of :class:`FireCategory`
            codes. A fresh array, so a caller may mutate it.
        """
        values = np.asarray(state, dtype=np.float64)
        return (
            np.rint(values[self.fire_offset :])
            .astype(np.int64)
            .reshape(self.num_rows, self.num_cols)
        )

    @property
    def obstacle_mask(self) -> np.ndarray:
        """Read-only mask of the obstacle cells, ``True`` where blocked."""
        return self._obstacle_mask

    def decode_action(self, action: Any) -> np.ndarray:
        """Return the per-robot actions of a joint action.

        Args:
            action: The joint action, an integer in ``[0, 5 ** num_robots)``.

        Returns:
            ``(num_robots,)`` ``int64`` array of :class:`FirefightingAction`
            codes, robot 0 first.

        Raises:
            ValueError: If the joint action is outside the action space.
        """
        joint = int(action)
        if not 0 <= joint < self.num_actions:
            raise ValueError(f"joint action must be in [0, {self.num_actions}), got {joint}")
        return self._action_table[joint]

    @staticmethod
    def alight_mask(fire: np.ndarray) -> np.ndarray:
        """Return the mask of cells that are smoldering or burning.

        Args:
            fire: A fire map of category codes.

        Returns:
            Boolean array of the same shape.
        """
        return (fire == int(FireCategory.SMOLDERING)) | (fire == int(FireCategory.BURNING))

    def get_actions(self) -> List[int]:
        """Return every joint action, as plain integers.

        Integer actions also sidestep the ``id(action)`` caching trap the
        environment API contract documents: there is no action array whose
        memory address a cache could be keyed on, so the recycled-address bug
        that cost ContinuousPush 95% of its steps cannot arise here.

        Returns:
            ``[0, 1, ..., 5 ** num_robots - 1]``.
        """
        return list(range(self.num_actions))

    # -- transition -----------------------------------------------------

    def _is_admissible(self, row: int, col: int, fire: np.ndarray) -> bool:
        """Whether a robot may enter ``(row, col)`` given the pre-step fire map.

        Args:
            row: Target row.
            col: Target column.
            fire: The fire map the move is resolved against, i.e. the one in
                the state the step is taken from -- motion resolves before
                suppression and before spread.

        Returns:
            ``True`` when the cell is on the grid, is not an obstacle, and is
            not burnt.
        """
        if not 0 <= row < self.num_rows or not 0 <= col < self.num_cols:
            return False
        if self._obstacle_mask[row, col]:
            return False
        return fire[row, col] != int(FireCategory.BURNT)

    def _move_targets(
        self, robots: np.ndarray, actions: np.ndarray, fire: np.ndarray
    ) -> List[Optional[Tuple[int, int]]]:
        """Return each robot's admissible move target, or ``None``.

        Args:
            robots: ``(num_robots, 4)`` state block.
            actions: Per-robot actions.
            fire: The pre-step fire map.

        Returns:
            One entry per robot: the cell it would move into if the move does
            not slip, or ``None`` when the robot is disabled, is suppressing,
            or the target is inadmissible -- the three cases in which the
            motion draw is not taken at all.
        """
        targets: List[Optional[Tuple[int, int]]] = []
        for robot in range(self.num_robots):
            if robots[robot, 3] <= 0 or actions[robot] == int(FirefightingAction.SUPPRESS):
                targets.append(None)
                continue
            offset = DIRECTION_OFFSETS[int(actions[robot])]
            row = int(robots[robot, 0]) + offset[0]
            col = int(robots[robot, 1]) + offset[1]
            targets.append((row, col) if self._is_admissible(row, col, fire) else None)
        return targets

    def _spraying_robots(self, robots: np.ndarray, actions: np.ndarray) -> List[int]:
        """Return the robots that actually spray this step.

        Args:
            robots: ``(num_robots, 4)`` state block.
            actions: Per-robot actions.

        Returns:
            Indices of the live robots that chose SUPPRESS with a non-empty
            tank. A robot with an empty tank that chooses SUPPRESS does nothing
            and pays nothing.
        """
        return [
            robot
            for robot in range(self.num_robots)
            if robots[robot, 3] > 0
            and actions[robot] == int(FirefightingAction.SUPPRESS)
            and robots[robot, 2] > 0
        ]

    def _coverage_counts(self, positions: np.ndarray, sprayers: Sequence[int]) -> np.ndarray:
        """Return how many sprays cover each cell.

        Each spraying robot covers its own cell and its four neighbours, and
        the counts add: two robots covering the same cell each get an
        independent attempt at soaking it. That is the only place in the model
        where the robots genuinely cooperate rather than merely divide the
        work.

        Args:
            positions: ``(num_robots, 2)`` post-motion cells.
            sprayers: Indices of the robots that sprayed.

        Returns:
            ``(num_rows, num_cols)`` ``int64`` counts.
        """
        counts = np.zeros((self.num_rows, self.num_cols), dtype=np.int64)
        for robot in sprayers:
            row, col = int(positions[robot, 0]), int(positions[robot, 1])
            counts[row, col] += 1
            for offset_row, offset_col in DIRECTION_OFFSETS:
                target_row, target_col = row + offset_row, col + offset_col
                if 0 <= target_row < self.num_rows and 0 <= target_col < self.num_cols:
                    counts[target_row, target_col] += 1
        return counts

    def _suppression_probability(self, fire: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Return the chance each cell is soaked, given how often it is sprayed.

        Args:
            fire: The fire map after motion, before suppression.
            counts: Per-cell spray counts.

        Returns:
            ``(num_rows, num_cols)`` probabilities. Zero wherever the count is
            zero or the category absorbs spray.
        """
        per_spray = self._suppression_probabilities[fire]
        return 1.0 - np.power(1.0 - per_spray, counts)

    def _ignition_probability(self, fire: np.ndarray, wind: Tuple[int, int]) -> np.ndarray:
        """Return the chance each cell catches, given the post-suppression map.

        A cell catches unless *every* alight neighbour fails to ignite it, so
        the per-cell probability is one minus a product over the alight
        four-neighbours. The single neighbour sitting directly upwind under
        ``wind`` contributes the boosted rate; the other three contribute the
        attenuated one.

        Args:
            fire: The fire map after suppression.
            wind: ``(direction, strength)``.

        Returns:
            ``(num_rows, num_cols)`` probabilities. Meaningful only on cells
            that are unburnt and not obstacles; callers mask it.
        """
        direction, strength = wind
        gain = self.wind_gain_high if strength == int(WindStrength.HIGH) else self.wind_gain_low
        downwind_rate = min(1.0, self.spread_probability * gain)
        crosswind_rate = self.spread_probability * (1.0 - self.crosswind_attenuation)

        alight = self.alight_mask(fire)
        survive = np.ones((self.num_rows, self.num_cols), dtype=np.float64)
        for code, (offset_row, offset_col) in enumerate(DIRECTION_OFFSETS):
            # ``shifted[k]`` is ``alight[k - offset]``: the neighbour that would
            # push fire into ``k`` along this offset. The offset equal to the
            # wind direction is the one the wind carries, so that neighbour is
            # the upwind one and gets the gain.
            shifted = np.zeros_like(alight)
            rows = slice(max(0, offset_row), self.num_rows + min(0, offset_row))
            cols = slice(max(0, offset_col), self.num_cols + min(0, offset_col))
            source_rows = slice(max(0, -offset_row), self.num_rows + min(0, -offset_row))
            source_cols = slice(max(0, -offset_col), self.num_cols + min(0, -offset_col))
            shifted[rows, cols] = alight[source_rows, source_cols]
            rate = downwind_rate if code == direction else crosswind_rate
            survive = np.where(shifted, survive * (1.0 - rate), survive)
        return 1.0 - survive

    # pylint: disable-next=too-many-locals
    def _transition(self, state: FirefightingState, action: Any) -> np.ndarray:
        """Draw one successor of ``state`` under ``action``.

        The six stages resolve in the order the formal definition fixes:
        motion, suppression, spread, growth and burnout, heat damage,
        bookkeeping. The order is not cosmetic -- resolving suppression before
        spread is what lets a robot stop a front by soaking the cell ahead of
        it in the same step, and resolving growth only on cells that were
        already alight *before* the spread is what stops a cell igniting and
        growing to burning within one step.

        Args:
            state: The state to step from.
            action: The joint action.

        Returns:
            A fresh ``float64`` successor state vector.
        """
        values = np.asarray(state, dtype=np.float64)
        successor = values.copy()
        actions = self.decode_action(action)
        robots = self.robots(values)
        fire = self.fire_map(values)
        wind = self.wind(values)

        # 1. Motion. The slip draw is taken only where a move could succeed,
        #    so a robot walking into a wall consumes no randomness.
        positions = robots[:, :2].copy()
        for robot, target in enumerate(self._move_targets(robots, actions, fire)):
            if target is None:
                continue
            if np.random.random() >= self.slip_probability:
                positions[robot] = target

        # 2. Suppression, then the tank, with a refill overriding the cost: a
        #    robot that sprays and steps into the depot on the same step ends
        #    the step full.
        sprayers = self._spraying_robots(robots, actions)
        tanks = robots[:, 2].copy()
        if sprayers:
            counts = self._coverage_counts(positions, sprayers)
            probabilities = self._suppression_probability(fire, counts)
            covered = np.flatnonzero((counts > 0).ravel())
            draws = np.random.random(covered.size)
            soaked = covered[draws < probabilities.ravel()[covered]]
            fire.flat[soaked] = int(FireCategory.WET)
            for robot in sprayers:
                tanks[robot] -= 1
        for robot in range(self.num_robots):
            if (int(positions[robot, 0]), int(positions[robot, 1])) == self.depot_cell:
                tanks[robot] = self.max_tank

        # 3. Spread. ``fire`` is the post-suppression map; the alight cells it
        #    is read from are the ones that survived the hoses.
        fire_after_suppression = fire.copy()
        ignitable = (fire == int(FireCategory.UNBURNT)) & ~self._obstacle_mask
        ignition = self._ignition_probability(fire, wind)
        candidates = np.flatnonzero((ignitable & (ignition > 0.0)).ravel())
        if candidates.size:
            draws = np.random.random(candidates.size)
            ignited = candidates[draws < ignition.ravel()[candidates]]
            fire.flat[ignited] = int(FireCategory.SMOLDERING)

        # 4. Growth and burnout, on the cells that were already alight before
        #    the spread stage.
        smoldering = np.flatnonzero(
            (fire_after_suppression == int(FireCategory.SMOLDERING)).ravel()
        )
        if smoldering.size:
            grown = smoldering[np.random.random(smoldering.size) < self.growth_probability]
            fire.flat[grown] = int(FireCategory.BURNING)
        burning = np.flatnonzero((fire_after_suppression == int(FireCategory.BURNING)).ravel())
        if burning.size:
            burnt = burning[np.random.random(burning.size) < self.burnout_probability]
            fire.flat[burnt] = int(FireCategory.BURNT)

        # 5. Heat damage, read off the final map at each robot's final cell.
        healths = robots[:, 3].copy()
        for robot in range(self.num_robots):
            damage = self._heat_damage[fire[int(positions[robot, 0]), int(positions[robot, 1])]]
            healths[robot] = max(0, int(healths[robot]) - int(damage))

        # 6. Bookkeeping. The wind is copied unchanged, which is what makes it
        #    identifiable from the spread pattern across a whole episode.
        successor[STEP_INDEX] = values[STEP_INDEX] + 1.0
        for robot in range(self.num_robots):
            base = ROBOT_OFFSET + ROBOT_FIELD_WIDTH * robot
            successor[base] = float(positions[robot, 0])
            successor[base + 1] = float(positions[robot, 1])
            successor[base + 2] = float(tanks[robot])
            successor[base + 3] = float(healths[robot])
        successor[self.fire_offset :] = fire.ravel().astype(np.float64)
        return successor

    def sample_next_state(self, state: FirefightingState, action: Any, n_samples: int = 1) -> Any:
        """Draw one or more successors of ``state`` under ``action``.

        Args:
            state: The state to step from.
            action: The joint action.
            n_samples: How many independent successors to draw. Defaults to 1.

        Returns:
            One ``float64`` state vector when ``n_samples`` is 1, otherwise an
            ``(n_samples, state_size)`` array.
        """
        if int(n_samples) == 1:
            return self._transition(state, action)
        return np.asarray([self._transition(state, action) for _ in range(int(n_samples))])

    # The density mirrors the six stages of ``_transition`` in one place on
    # purpose: splitting it would put the sampler's law and the density's law
    # in two files that can drift, which is the failure this method exists to
    # make impossible.
    # pylint: disable-next=too-many-locals,too-many-return-statements,too-many-branches,too-many-statements
    def _successor_log_probability(
        self, state: FirefightingState, action: Any, candidate: np.ndarray
    ) -> float:
        """Exact log-probability of one candidate successor.

        The six stages are conditionally independent given what precedes them,
        and -- crucially -- the intermediate map is recoverable from the pair
        ``(fire, next fire)``: wet is reachable only by suppression, burnt only
        by burnout, and a cell that ignites this step cannot also grow this
        step. So there is no marginalisation to do and the density is a plain
        product over robots and cells.

        Args:
            state: The state the step was taken from.
            action: The joint action.
            candidate: A candidate successor.

        Returns:
            The log-probability, or ``-inf`` when the candidate is
            unreachable.
        """
        values = np.asarray(state, dtype=np.float64)
        if candidate.shape != (self.state_size,) or not np.all(np.isfinite(candidate)):
            return -np.inf
        if candidate[STEP_INDEX] != values[STEP_INDEX] + 1.0:
            return -np.inf
        if self.wind(candidate) != self.wind(values):
            return -np.inf

        actions = self.decode_action(action)
        robots = self.robots(values)
        next_robots = self.robots(candidate)
        fire = self.fire_map(values)
        next_fire = self.fire_map(candidate)

        # 1. Motion.
        log_probability = 0.0
        targets = self._move_targets(robots, actions, fire)
        positions = next_robots[:, :2]
        for robot, target in enumerate(targets):
            stayed = tuple(positions[robot]) == tuple(robots[robot, :2])
            if target is None:
                if not stayed:
                    return -np.inf
                continue
            if tuple(positions[robot]) == target:
                log_probability += np.log1p(-self.slip_probability)
            elif stayed:
                log_probability += (
                    np.log(self.slip_probability) if self.slip_probability > 0.0 else -np.inf
                )
            else:
                return -np.inf
        if not np.isfinite(log_probability):
            return -np.inf

        # 2. Suppression and the deterministic tank rule.
        sprayers = self._spraying_robots(robots, actions)
        counts = self._coverage_counts(positions, sprayers)
        soak = self._suppression_probability(fire, counts)
        expected_tanks = robots[:, 2].copy()
        for robot in sprayers:
            expected_tanks[robot] -= 1
        for robot in range(self.num_robots):
            if (int(positions[robot, 0]), int(positions[robot, 1])) == self.depot_cell:
                expected_tanks[robot] = self.max_tank
        if not np.array_equal(next_robots[:, 2], expected_tanks):
            return -np.inf

        # 3-4. Per-cell map law. ``intermediate`` is the post-suppression map,
        #      recovered from the two maps as argued above.
        intermediate = fire.copy()
        ignitable = (fire == int(FireCategory.UNBURNT)) & ~self._obstacle_mask
        wet = int(FireCategory.WET)
        with np.errstate(divide="ignore"):
            log_soak = np.log(soak)
            log_dry = np.log1p(-soak)
        for index in range(self.num_cells):
            row, col = divmod(index, self.num_cols)
            before, after = int(fire[row, col]), int(next_fire[row, col])
            if before in (int(FireCategory.BURNT), wet):
                if after != before:
                    return -np.inf
                continue
            if after == wet:
                log_probability += float(log_soak[row, col])
                intermediate[row, col] = wet
                continue
            log_probability += float(log_dry[row, col])
            if not np.isfinite(log_probability):
                return -np.inf
            if before == int(FireCategory.UNBURNT) and after not in (
                int(FireCategory.UNBURNT),
                int(FireCategory.SMOLDERING),
            ):
                return -np.inf
            if before == int(FireCategory.SMOLDERING) and after not in (
                int(FireCategory.SMOLDERING),
                int(FireCategory.BURNING),
            ):
                return -np.inf
            if before == int(FireCategory.BURNING) and after not in (
                int(FireCategory.BURNING),
                int(FireCategory.BURNT),
            ):
                return -np.inf

        ignition = self._ignition_probability(intermediate, self.wind(values))
        for index in range(self.num_cells):
            row, col = divmod(index, self.num_cols)
            before, after = int(fire[row, col]), int(next_fire[row, col])
            if after == wet or before in (int(FireCategory.BURNT), wet):
                continue
            if before == int(FireCategory.UNBURNT):
                if not ignitable[row, col]:
                    # An obstacle cell never ignites, so only "stayed unburnt"
                    # is reachable and it carries no probability of its own.
                    if after != int(FireCategory.UNBURNT):
                        return -np.inf
                    continue
                probability = (
                    ignition[row, col]
                    if after == int(FireCategory.SMOLDERING)
                    else 1.0 - ignition[row, col]
                )
            elif before == int(FireCategory.SMOLDERING):
                probability = (
                    self.growth_probability
                    if after == int(FireCategory.BURNING)
                    else 1.0 - self.growth_probability
                )
            else:
                probability = (
                    self.burnout_probability
                    if after == int(FireCategory.BURNT)
                    else 1.0 - self.burnout_probability
                )
            if probability <= 0.0:
                return -np.inf
            log_probability += float(np.log(probability))

        # 5. Heat damage is deterministic given the final map and the poses.
        for robot in range(self.num_robots):
            damage = self._heat_damage[
                next_fire[int(positions[robot, 0]), int(positions[robot, 1])]
            ]
            if int(next_robots[robot, 3]) != max(0, int(robots[robot, 3]) - int(damage)):
                return -np.inf
        return float(log_probability)

    def transition_log_probability(
        self, state: FirefightingState, action: Any, next_states: Any
    ) -> np.ndarray:
        """Log-probability of each candidate successor.

        Args:
            state: The state the step was taken from.
            action: The joint action.
            next_states: Candidate successors.

        Returns:
            One log-probability per candidate; ``-inf`` for unreachable ones.
        """
        candidates = np.atleast_2d(np.asarray(next_states, dtype=np.float64))
        return np.asarray(
            [self._successor_log_probability(state, action, candidate) for candidate in candidates]
        )

    # -- observations ---------------------------------------------------

    def visible_mask(self, state: FirefightingState) -> np.ndarray:
        """Return the cells some live robot can see.

        Two robots standing together see barely more than one, so spreading
        out is what buys information. A disabled robot sees nothing.

        Args:
            state: A state vector.

        Returns:
            ``(num_rows, num_cols)`` boolean mask.
        """
        robots = self.robots(state)
        visible = np.zeros((self.num_rows, self.num_cols), dtype=bool)
        for robot in range(self.num_robots):
            if robots[robot, 3] <= 0:
                continue
            row, col = int(robots[robot, 0]), int(robots[robot, 1])
            visible[
                max(0, row - self.sensing_radius) : row + self.sensing_radius + 1,
                max(0, col - self.sensing_radius) : col + self.sensing_radius + 1,
            ] = True
        return visible

    def _robot_fields(self, state: FirefightingState) -> np.ndarray:
        """Return the exactly-reported part of an observation.

        Args:
            state: A state vector.

        Returns:
            ``(4 * num_robots,)`` ``float64`` array of row, column, tank and
            health per robot.
        """
        values = np.asarray(state, dtype=np.float64)
        return values[ROBOT_OFFSET : ROBOT_OFFSET + ROBOT_FIELD_WIDTH * self.num_robots].copy()

    def _draw_observation(self, next_state: FirefightingState) -> np.ndarray:
        """Draw one noisy reading of ``next_state``.

        Args:
            next_state: The state being observed.

        Returns:
            A ``float64`` observation vector.
        """
        visible = self.visible_mask(next_state)
        fire = self.fire_map(next_state)
        reported = np.full(self.num_cells, UNKNOWN_CATEGORY, dtype=np.float64)
        seen = np.flatnonzero(visible.ravel())
        if seen.size:
            truth = fire.ravel()[seen]
            wrong = np.random.random(seen.size) < self.observation_error_probability
            # A uniform offset of 1..4 lands on each of the four wrong
            # categories with equal probability, which is exactly the
            # off-diagonal mass the confusion matrix spreads.
            offsets = np.random.randint(1, NUM_CATEGORIES, size=seen.size)
            reported[seen] = np.where(wrong, (truth + offsets) % NUM_CATEGORIES, truth)
        return np.concatenate([self._robot_fields(next_state), reported])

    def sample_observation(
        self, next_state: FirefightingState, action: Any, n_samples: int = 1
    ) -> Any:
        """Draw one or more readings of ``next_state``.

        The observation depends on the successor alone, not on the action.

        Args:
            next_state: The state being observed.
            action: Ignored.
            n_samples: How many independent readings to draw. Defaults to 1.

        Returns:
            One ``float64`` observation vector when ``n_samples`` is 1,
            otherwise an ``(n_samples, observation_size)`` array.
        """
        del action
        if int(n_samples) == 1:
            return self._draw_observation(next_state)
        return np.asarray([self._draw_observation(next_state) for _ in range(int(n_samples))])

    def observation_log_probability(
        self, next_state: FirefightingState, action: Any, observations: Any
    ) -> np.ndarray:
        """Log-likelihood of each reading under ``next_state``.

        The poses, tanks and healths are reported exactly, so any mismatch in
        them gives ``-inf``. Every visible cell contributes its confusion
        probability. Unobserved cells contribute nothing, because the unknown
        marker is deterministic given the poses -- but reporting a *category*
        for a cell nobody is looking at is impossible and gives ``-inf``.

        Args:
            next_state: The state being observed.
            action: Ignored.
            observations: Candidate readings.

        Returns:
            One log-likelihood per candidate.
        """
        del action
        expected = self._robot_fields(next_state)
        visible = self.visible_mask(next_state).ravel()
        truth = self.fire_map(next_state).ravel()
        candidates = np.atleast_2d(np.asarray(observations, dtype=np.float64))

        error = self.observation_error_probability
        with np.errstate(divide="ignore"):
            log_correct = float(np.log1p(-error))
            log_wrong = float(np.log(error / (NUM_CATEGORIES - 1)))

        scores = np.zeros(len(candidates), dtype=np.float64)
        for position, candidate in enumerate(candidates):
            if candidate.shape != (self.observation_size,):
                scores[position] = -np.inf
                continue
            if not np.array_equal(candidate[: expected.size], expected):
                scores[position] = -np.inf
                continue
            reported = candidate[expected.size :]
            if np.any(reported[~visible] != UNKNOWN_CATEGORY):
                scores[position] = -np.inf
                continue
            seen = reported[visible]
            if np.any(seen < 0) or np.any(seen >= NUM_CATEGORIES) or np.any(seen % 1.0 != 0.0):
                scores[position] = -np.inf
                continue
            matches = int(np.count_nonzero(seen == truth[visible]))
            mismatches = int(seen.size) - matches
            # Each term is added only when its count is non-zero. Both endpoints
            # of ``observation_error_probability`` are legal and documented: at
            # 0 the wrong-category log is ``-inf``, at 1 the correct-category
            # log is, and ``0 * -inf`` is ``NaN`` rather than the 0 it should
            # be. A NaN here does not raise -- it silently poisons a particle
            # weight, and from there the whole belief.
            score = 0.0
            if matches:
                score += matches * log_correct
            if mismatches:
                score += mismatches * log_wrong
            scores[position] = float(score)
        return scores

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        """Check whether two readings are the same array of numbers.

        Args:
            observation1: A reading.
            observation2: Another reading.

        Returns:
            ``True`` when they are elementwise equal.
        """
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
            The reading's raw ``float64`` bytes.
        """
        return np.ascontiguousarray(observation, dtype=np.float64).tobytes()

    def hash_action(self, action: Any) -> Hashable:
        """Return a hashable key for a joint action, which is already an int.

        Args:
            action: The joint action.

        Returns:
            The action as a plain ``int``.
        """
        return int(action)

    # -- reward ---------------------------------------------------------

    @property
    def reward_requires_next_state(self) -> bool:
        """Every reward term reads the realised successor, so this is ``True``."""
        return True

    def reward(self, state: FirefightingState, action: Any, next_state: Any = None) -> float:
        """Score one transition.

        The success bonus is paid on the transition *into* a fire-free state,
        and the step cost is charged on that transition too -- which is why the
        largest reachable reward is ``success_reward - step_cost``.

        Args:
            state: The state the step was taken from.
            action: The joint action.
            next_state: The realised successor. Defaults to ``None``, in which
                case one is drawn: every term here reads the successor, so
                there is no correct number to return without one and returning
                a wrong one silently would be worse than the extra draw.

        Returns:
            The immediate reward.
        """
        if next_state is None:
            next_state = self.sample_next_state(state=state, action=action)
        values = np.asarray(state, dtype=np.float64)
        successor = np.asarray(next_state, dtype=np.float64)
        fire = self.fire_map(values)
        next_fire = self.fire_map(successor)
        robots = self.robots(values)
        next_robots = self.robots(successor)

        smoldering = int(np.count_nonzero(next_fire == int(FireCategory.SMOLDERING)))
        burning = int(np.count_nonzero(next_fire == int(FireCategory.BURNING)))
        newly_burnt = int(
            np.count_nonzero(
                (next_fire == int(FireCategory.BURNT)) & (fire != int(FireCategory.BURNT))
            )
        )
        health_lost = int(np.sum(robots[:, 3] - next_robots[:, 3]))
        sprays = len(self._spraying_robots(robots, self.decode_action(action)))

        reward = -self.step_cost
        reward -= self.smoldering_cell_cost * smoldering
        reward -= self.burning_cell_cost * burning
        reward -= self.burnt_cell_cost * newly_burnt
        reward -= self.damage_cost * health_lost
        reward -= self.water_cost * sprays
        if smoldering == 0 and burning == 0:
            reward += self.success_reward
        return float(reward)

    # -- termination ----------------------------------------------------

    def is_terminal(self, state: FirefightingState) -> bool:
        """Whether the fire is out, every robot is down, or time has run out.

        Args:
            state: A state vector.

        Returns:
            ``True`` for a terminal state.
        """
        if not np.any(self.alight_mask(self.fire_map(state))):
            return True
        if self.is_all_robots_disabled_terminal and not np.any(self.robots(state)[:, 3] > 0):
            return True
        return self.step_count(state) >= self.max_steps

    # -- distributions --------------------------------------------------

    def initial_state_dist(self) -> Distribution:
        """A uniform hidden wind and a uniformly placed fire, robots at their posts.

        Returns:
            The reset distribution.
        """
        return MultiAgentFirefightingInitialStateDistribution(
            state_size=self.state_size,
            fire_offset=self.fire_offset,
            wind_direction_index=self.wind_direction_index,
            wind_strength_index=self.wind_strength_index,
            robot_start_cells=self.robot_start_cells,
            max_tank=self.max_tank,
            max_health=self.max_health,
            num_initial_fires=self.num_initial_fires,
            ignitable_indices=self._ignitable_indices,
            num_cols=self.num_cols,
        )

    def initial_observation_dist(self) -> DiscreteDistribution:
        """The pre-episode reading: robots at their posts, nothing sensed yet.

        Returns:
            A point mass on the known robot fields with every cell marked
            unknown. This is a sentinel, not a scan: the first real reading
            arrives with the first transition, and scoring this one against a
            state would be scoring a reading that no observation model
            produced.
        """
        observation = np.full(self.observation_size, UNKNOWN_CATEGORY, dtype=np.float64)
        for robot, (row, col) in enumerate(self.robot_start_cells):
            base = ROBOT_FIELD_WIDTH * robot
            observation[base] = float(row)
            observation[base + 1] = float(col)
            observation[base + 2] = float(self.max_tank)
            observation[base + 3] = float(self.max_health)
        return DiscreteDistribution(values=[observation], probs=np.array([1.0]))

    # -- metrics --------------------------------------------------------

    def step_info(self, state: Any, action: Any, next_state: Any) -> Dict[str, float]:
        """Report the per-step channels this environment's metrics are built on.

        Draws no randomness: every channel is a deterministic read of
        ``state``, ``action`` and ``next_state``.

        The state-shaped channels are read from ``next_state`` when there is
        one, for the reason Battleship documents: the episode runner checks its
        step budget before it checks terminality, so an episode whose final
        allowed step puts the fire out records no terminal bookkeeping step,
        and reading from ``state`` alone would score that episode as a
        still-burning timeout.

        Args:
            state: The state the step was taken from, or the final state on the
                terminal bookkeeping step.
            action: The joint action, or ``None`` on the terminal step.
            next_state: The realised successor, or ``None`` on the terminal
                step.

        Returns:
            The channels named by :class:`MultiAgentFirefightingStepChannel`.
            The two transition channels report ``0.0`` on the terminal step,
            where no action was taken.
        """
        health_lost = 0.0
        sprays = 0.0
        if action is not None and next_state is not None:
            robots = self.robots(state)
            health_lost = float(np.sum(robots[:, 3] - self.robots(next_state)[:, 3]))
            sprays = float(len(self._spraying_robots(robots, self.decode_action(action))))

        scored = state if next_state is None else next_state
        fire = self.fire_map(scored)
        robots = self.robots(scored)
        alight = self.alight_mask(fire)
        alight_cells = float(np.count_nonzero(alight))
        extinguished = float(alight_cells == 0.0)
        disabled = float(np.count_nonzero(robots[:, 3] <= 0))
        # Precedence: goal, then failure, then timeout. Goal wins over failure
        # so a fire put out by robots that then burned out still counts as a
        # success, and timeout is written as the residual so the three
        # ``ended_by_*`` rates always sum to one.
        failure = float(
            extinguished == 0.0
            and self.is_all_robots_disabled_terminal
            and disabled == float(self.num_robots)
        )
        return {
            MultiAgentFirefightingStepChannel.FIRE_EXTINGUISHED.value: extinguished,
            MultiAgentFirefightingStepChannel.ALL_ROBOTS_DISABLED.value: failure,
            MultiAgentFirefightingStepChannel.TIMED_OUT_WITH_FIRE.value: 1.0
            - extinguished
            - failure,
            MultiAgentFirefightingStepChannel.RECORDED_STEP.value: 1.0,
            MultiAgentFirefightingStepChannel.ROBOT_IN_ALIGHT_CELL.value: float(
                sum(
                    1
                    for robot in range(self.num_robots)
                    if robots[robot, 3] > 0 and alight[int(robots[robot, 0]), int(robots[robot, 1])]
                )
            ),
            MultiAgentFirefightingStepChannel.HEALTH_LOST.value: health_lost,
            MultiAgentFirefightingStepChannel.SUPPRESS_ACTIONS.value: sprays,
            MultiAgentFirefightingStepChannel.ALIGHT_CELLS.value: alight_cells,
            MultiAgentFirefightingStepChannel.BURNT_CELL_FRACTION.value: float(
                np.count_nonzero(fire == int(FireCategory.BURNT))
            )
            / float(self.num_cells),
            MultiAgentFirefightingStepChannel.ROBOTS_DISABLED.value: disabled,
        }

    def get_metric_specs(self) -> List[StepInfoMetric]:
        """Declare the multi-agent firefighting metrics.

        Completion reduces with ``ANY``. ``BURNT`` and ``WET`` are absorbing,
        so a fire-free map cannot be undone and ``ANY`` and ``LAST`` agree
        here; ``ANY`` is the honest name for "the task was completed at some
        point" and is what the reduction's own docstring recommends for a
        reach-a-goal task.

        The danger is reported both as a count and as a severity, because the
        worst moment and the total say different things about a planner: a
        planner that lets the fire reach forty cells and then beats it out is
        not the same as one that never let it past five, and the totals alone
        would not distinguish them.

        Returns:
            One spec per metric named in :class:`MultiAgentFirefightingMetrics`.
        """
        channel = MultiAgentFirefightingStepChannel
        return [
            StepInfoMetric(
                name=MultiAgentFirefightingMetrics.TASK_COMPLETION_RATE.value,
                channel=channel.FIRE_EXTINGUISHED.value,
                per_episode=EpisodeReduction.ANY,
            ),
            StepInfoMetric(
                name=MultiAgentFirefightingMetrics.ENDED_BY_GOAL.value,
                channel=channel.FIRE_EXTINGUISHED.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=MultiAgentFirefightingMetrics.ENDED_BY_FAILURE.value,
                channel=channel.ALL_ROBOTS_DISABLED.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=MultiAgentFirefightingMetrics.ENDED_BY_TIMEOUT.value,
                channel=channel.TIMED_OUT_WITH_FIRE.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=MultiAgentFirefightingMetrics.AVERAGE_EPISODE_LENGTH.value,
                channel=channel.RECORDED_STEP.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=MultiAgentFirefightingMetrics.ROBOT_STEPS_IN_FIRE.value,
                channel=channel.ROBOT_IN_ALIGHT_CELL.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=MultiAgentFirefightingMetrics.ROBOT_HEALTH_LOST.value,
                channel=channel.HEALTH_LOST.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=MultiAgentFirefightingMetrics.SUPPRESSANT_UNITS_USED.value,
                channel=channel.SUPPRESS_ACTIONS.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=MultiAgentFirefightingMetrics.MAX_SIMULTANEOUS_ALIGHT_CELLS.value,
                channel=channel.ALIGHT_CELLS.value,
                per_episode=EpisodeReduction.MAX,
            ),
            StepInfoMetric(
                name=MultiAgentFirefightingMetrics.MAX_BURNT_CELL_FRACTION.value,
                channel=channel.BURNT_CELL_FRACTION.value,
                per_episode=EpisodeReduction.MAX,
            ),
            StepInfoMetric(
                name=MultiAgentFirefightingMetrics.ROBOTS_DISABLED_AT_END.value,
                channel=channel.ROBOTS_DISABLED.value,
                per_episode=EpisodeReduction.LAST,
            ),
        ]

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
        # almost none of them render anything, so the renderer's palette
        # tables and fonts stay out of a planning run's memory.
        # pylint: disable-next=import-outside-toplevel
        from POMDPPlanners.environments.multiagent_firefighting_pomdp.multiagent_firefighting_visualization.multiagent_firefighting_visualizer import (  # noqa: E501
            MultiAgentFirefightingVisualizer,
        )

        cache_path = output_dir / f"multiagent_firefighting_{episode_index}.gif"
        MultiAgentFirefightingVisualizer(self).create_visualization(history, cache_path)

    def build_episode_trace(
        self, history: List[StepData], episode_index: int, policy_name: Optional[str] = None
    ) -> "EpisodeTrace":
        """Write this episode as data, beside the GIF.

        Args:
            history: Episode history.
            episode_index: Zero-based episode index within its run.
            policy_name: Name of the policy that produced the episode.

        Returns:
            The episode's trace, with payload kind
            ``multiagent_firefighting.v1``.
        """
        # Imported here rather than at module scope, for the same reason the
        # renderer is: the exporter pulls in the trace schema, and this module
        # is imported by every worker of every run, almost none of which write
        # anything.
        # pylint: disable-next=import-outside-toplevel
        from POMDPPlanners.environments.multiagent_firefighting_pomdp.multiagent_firefighting_visualization.trace_exporter import (  # noqa: E501
            build_multiagent_firefighting_trace,
        )

        return build_multiagent_firefighting_trace(
            environment=self,
            history=history,
            episode_index=episode_index,
            policy_name=policy_name,
        )


def create_firefighting_state(
    env: MultiAgentFirefightingPOMDP,
    robots: Sequence[Sequence[int]],
    wind: Tuple[int, int],
    fire: np.ndarray,
    step: int = 0,
) -> np.ndarray:
    """Build one state vector in ``env``'s layout.

    Exists so tests and the golden-visualization fixtures can state a world
    directly instead of reproducing the packing by hand in several places.

    Args:
        env: The environment whose layout is used.
        robots: One ``(row, col, tank, health)`` sequence per robot.
        wind: ``(direction, strength)``.
        fire: ``(num_rows, num_cols)`` category codes.
        step: Step counter. Defaults to 0.

    Returns:
        A ``float64`` state vector.

    Raises:
        ValueError: If the robots or the fire map do not fit the environment.
    """
    if len(robots) != env.num_robots:
        raise ValueError(f"expected {env.num_robots} robots, got {len(robots)}")
    grid = np.asarray(fire, dtype=np.float64)
    if grid.shape != (env.num_rows, env.num_cols):
        raise ValueError(f"expected a {env.num_rows}x{env.num_cols} fire map, got {grid.shape}")

    state = np.zeros(env.state_size, dtype=np.float64)
    state[STEP_INDEX] = float(step)
    for robot, fields in enumerate(robots):
        base = ROBOT_OFFSET + ROBOT_FIELD_WIDTH * robot
        state[base : base + ROBOT_FIELD_WIDTH] = np.asarray(fields, dtype=np.float64)
    state[env.wind_direction_index] = float(wind[0])
    state[env.wind_strength_index] = float(wind[1])
    state[env.fire_offset :] = grid.ravel()
    return state


__all__ = [
    "FireCategory",
    "FirefightingAction",
    "FirefightingState",
    "MultiAgentFirefightingMetrics",
    "MultiAgentFirefightingPOMDP",
    "MultiAgentFirefightingStepChannel",
    "NUM_WIND_VALUES",
    "ROBOT_FIELD_WIDTH",
    "ROBOT_OFFSET",
    "STEP_INDEX",
    "UNKNOWN_CATEGORY",
    "WindDirection",
    "WindStrength",
    "create_firefighting_state",
]
