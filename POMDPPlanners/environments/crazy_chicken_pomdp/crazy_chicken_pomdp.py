# SPDX-License-Identifier: MIT

"""A Crazy Chicken shooter as a POMDP: clear the flock before it reaches the ship.

The ship sits on row 0 of a ``W`` by ``H`` grid and may step left, step right,
stay, or fire. Above it, ``N`` chickens patrol sideways and bounce off the
walls; each step a patrolling chicken may switch, unseen, into a dive and start
dropping one row per step. A projectile rises one row per step and kills the
first chicken it reaches, dying with it. The episode ends when the flock is
cleared, when a chicken reaches the ship's cell, or when the step budget runs
out.

**What is hidden.** The ship's own column, its cooldown and the sky full of its
own projectiles are all known exactly -- they follow from the actions it took.
What it does not know is where the chickens are and which of them are diving,
and it learns that from two sensors that each give half an answer:

* the **camera** covers a cone opening upward from the ship and reports a
  chicken's *column* offset, with noise, saying nothing about its row;
* the **radar** covers a disc around the ship and reports a chicken's *row*
  distance, with noise, plus whether it is dropping, which it gets wrong a tenth
  of the time.

Each sensor misses a chicken it can see about a tenth of the time. The
observation always carries one slot per chicken, reported or not, so its length
never changes and silence is itself a reading: a chicken well inside both
sensors' reach that reports nothing is unlikely, which pushes the belief towards
worlds where it is somewhere else. That is the whole reason the likelihood
multiplies a factor in for *every* chicken rather than only the reported ones.

**Two things the design proposal left open, decided here.** A chicken that
reaches row 0 outside the ship's column pulls up: it goes back to patrolling at
the top row, keeping its column and direction. Without such a rule a dive would
either have to carry the chicken off the grid -- which would let the flock clear
itself and make the completion bonus free -- or park it permanently on a row no
projectile can reach, which is unwinnable. And the dive coin is flipped *before*
the chickens move, so a chicken that switches this step also drops this step;
flipping it afterwards would delay every dive by one step for no gain.

Classes:
    CrazyChickenPOMDP: The environment.
    CrazyChickenAction: Its four action indices.
    CrazyChickenMetrics: The metric names it reports.
    CrazyChickenStepChannel: The per-step channels those metrics are built from.
    ObservationMode: Fully versus partially observable.
"""

# pylint: disable=too-many-lines  # one environment, its sensors and its metrics

import math
from enum import Enum, IntEnum
from pathlib import Path
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.metrics import MetricValue
from POMDPPlanners.core.simulation.step_info_metrics import (
    EpisodeReduction,
    StepInfoMetric,
    extract_episode_step_infos,
    require_non_empty_histories,
)
from POMDPPlanners.environments.crazy_chicken_pomdp.crazy_chicken_schema import (
    CHICKEN_ALIVE,
    CHICKEN_COLUMN,
    CHICKEN_DIRECTION,
    CHICKEN_MODE,
    CHICKEN_ROW,
    COOLDOWN_INDEX,
    MODE_DIVE,
    MODE_PATROL,
    NO_PROJECTILE,
    OBSERVATION_CHICKEN_WIDTH,
    OBSERVATION_SHIP_WIDTH,
    OBSERVED_CAMERA_OFFSET,
    OBSERVED_CAMERA_REPORTED,
    OBSERVED_RADAR_DROP,
    OBSERVED_RADAR_REPORTED,
    OBSERVED_RADAR_ROWS,
    OBSERVED_SHIP_COLUMN_INDEX,
    SHIP_COLUMN_INDEX,
    SHIP_HIT_INDEX,
    STEP_INDEX,
    CrazyChickenInitialStateDistribution,
    chicken_slots,
    make_state,
    observation_size,
    projectile_rows,
    state_size,
)
from POMDPPlanners.environments.crazy_chicken_pomdp.crazy_chicken_sensors import (
    camera_sees,
    radar_sees,
    rounded_normal_pmf,
    sample_rounded_normal,
)
from POMDPPlanners.utils.statistics_utils import confidence_interval

if TYPE_CHECKING:
    from POMDPPlanners.core.simulation import History


#: Log-probability standing in for an impossible observation. ``-inf`` is
#: correct but propagates into ``0 * -inf`` NaNs inside weight normalisation, so
#: the likelihood floors at a number no finite reading can reach instead.
IMPOSSIBLE_LOG_PROBABILITY = -1e18

#: Episodes below this count make a t-interval undefined; mirror the unbounded
#: interval the shared step-info aggregator reports in that case.
_MIN_EPISODES_FOR_CONFIDENCE_INTERVAL = 2


class CrazyChickenAction(IntEnum):
    """The ship's four actions.

    Attributes:
        FIRE: Launch a projectile, if the cooldown is over and this column is
            clear. Otherwise it behaves exactly like ``STAY``, and no shot cost
            is charged.
        LEFT: Step one column left, clamped at the wall.
        RIGHT: Step one column right, clamped at the wall.
        STAY: Hold position.
    """

    FIRE = 0
    LEFT = 1
    RIGHT = 2
    STAY = 3


class ObservationMode(Enum):
    """Whether the ship sees the world or only its two sensors.

    Attributes:
        PARTIAL: The camera and radar readings described in the module
            docstring. This is the POMDP.
        FULL: The observation is the state itself, for use as a fully observable
            control baseline on the same dynamics and the same reward.
    """

    PARTIAL = "partial"
    FULL = "full"


class CrazyChickenStepChannel(Enum):
    """Per-step channels reported by :meth:`CrazyChickenPOMDP.step_info`."""

    FLOCK_CLEARED = "flock_cleared"
    SHIP_DESTROYED = "ship_destroyed"
    STILL_RUNNING = "still_running"
    RECORDED_STEP = "recorded_step"
    CHICKENS_KILLED = "chickens_killed"
    SHOT_FIRED = "shot_fired"
    CHICKEN_ENCROACHMENT_CELLS = "chicken_encroachment_cells"


class CrazyChickenMetrics(Enum):
    """Metric names for the Crazy Chicken environment."""

    TASK_COMPLETION_RATE = "task_completion_rate"
    ENDED_BY_GOAL = "ended_by_goal"
    ENDED_BY_FAILURE = "ended_by_failure"
    ENDED_BY_TIMEOUT = "ended_by_timeout"
    AVERAGE_EPISODE_LENGTH = "average_episode_length"
    AVERAGE_CHICKENS_KILLED = "average_chickens_killed"
    AVERAGE_SHOTS_FIRED = "average_shots_fired"
    AVERAGE_HITS_TAKEN = "average_hits_taken"
    MAX_CHICKEN_ENCROACHMENT_CELLS = "max_chicken_encroachment_cells"
    SHOT_ACCURACY = "shot_accuracy"


#: Type alias for a Crazy Chicken state.
CrazyChickenState = np.ndarray


def resolve_observation_mode(value: Union[ObservationMode, str]) -> ObservationMode:
    """Return ``value`` as an :class:`ObservationMode`.

    Raises on an unknown name so a typo in a config file fails at construction
    rather than silently selecting the partially observable default.

    Args:
        value: The member, or its string value.

    Returns:
        The resolved member.

    Raises:
        ValueError: If ``value`` names no member.
    """
    if isinstance(value, ObservationMode):
        return value
    try:
        return ObservationMode(str(value))
    except ValueError as error:
        names = ", ".join(sorted(member.value for member in ObservationMode))
        raise ValueError(f"unknown observation_mode {value!r}; expected one of {names}") from error


# pylint: disable-next=too-many-public-methods,too-many-instance-attributes
class CrazyChickenPOMDP(DiscreteActionsEnvironment):
    """Clear a flock of chickens from a grid before one of them reaches the ship.

    The task is complete when every chicken slot is dead. It fails when a
    chicken reaches the ship's cell, and times out at ``max_steps``. Those three
    are the only ways an episode ends, and they are what the three
    ``ended_by_*`` metrics report.
    """

    # pylint: disable-next=too-many-arguments,too-many-locals,too-many-statements,too-many-branches
    def __init__(
        self,
        num_columns: int = 8,
        num_rows: int = 7,
        num_chickens: int = 4,
        fire_cooldown: int = 1,
        dive_probability: float = 0.15,
        initial_dive_probability: float = 0.0,
        camera_detection_probability: float = 0.9,
        radar_detection_probability: float = 0.9,
        ship_column_noise_std: float = 0.5,
        camera_offset_noise_std: float = 1.0,
        radar_range_noise_std: float = 1.0,
        drop_flag_error_probability: float = 0.1,
        camera_slope: float = 1.0,
        radar_radius: float = 6.0,
        observation_mode: Union[ObservationMode, str] = ObservationMode.PARTIAL,
        kill_reward: float = 10.0,
        shot_cost: float = 1.0,
        step_cost: float = 0.1,
        ship_hit_penalty: float = 50.0,
        clear_reward: float = 50.0,
        max_steps: int = 60,
        discount_factor: float = 0.95,
        name: str = "CrazyChicken",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the Crazy Chicken POMDP.

        Args:
            num_columns: Grid width. Defaults to 8. Wide enough that the ship
                cannot cover the whole sky from one place, narrow enough that a
                projectile is worth aiming.
            num_rows: Grid height. Defaults to 7, so a chicken that starts at the
                top takes six steps to reach the ship -- long enough for the
                ship to notice the dive and answer it.
            num_chickens: Number of chicken slots. Defaults to 4. Every slot is
                carried in both the state and the observation for the whole
                episode, alive or not, so this is what sets both vector lengths.
            fire_cooldown: Steps the ship must wait after a shot. Defaults to 1.
            dive_probability: Chance per step that a patrolling chicken switches
                into a dive. Defaults to 0.15, which puts the typical patrol at
                around six or seven steps -- long enough to be worth tracking,
                short enough that waiting is not a strategy.
            initial_dive_probability: Chance a chicken is already diving when
                the episode starts. Defaults to 0.0, so every episode opens with
                the whole flock patrolling.
            camera_detection_probability: Chance the camera reports a chicken
                inside its cone. Defaults to 0.9.
            radar_detection_probability: Chance the radar reports a chicken
                inside its disc. Defaults to 0.9.
            ship_column_noise_std: Noise on the ship's own-column reading.
                Defaults to 0.5. The ship's column is in fact determined by its
                own actions, so this reading is redundant evidence rather than
                the only source of it; it exists so that the observation is a
                complete reading of the world rather than a chicken report with
                a hole in it.
            camera_offset_noise_std: Noise on a reported column offset. Defaults
                to 1.0 cell, which is the value the design proposal's worked
                example uses.
            radar_range_noise_std: Noise on a reported row distance. Defaults to
                1.0 cell.
            drop_flag_error_probability: Chance the radar's dropping flag is
                inverted. Defaults to 0.1. This is the only channel that reports
                the hidden mode at all, which is what makes the mode worth
                inferring rather than reading off.
            camera_slope: Half-slope of the camera cone. Defaults to 1.0, a
                45-degree cone.
            radar_radius: Radar radius in cells. Defaults to 6.0.
            observation_mode: ``ObservationMode.PARTIAL`` (the default) for the
                two-sensor reading, or ``ObservationMode.FULL`` for a fully
                observable baseline whose observation is the state. Accepts the
                member or its string value.
            kill_reward: Paid per chicken killed. Defaults to 10.0.
            shot_cost: Charged per shot *actually* fired. Defaults to 1.0. A
                ``FIRE`` that the cooldown or an occupied column blocks costs
                nothing, because nothing left the ship.
            step_cost: Charged every step. Defaults to 0.1.
            ship_hit_penalty: Charged when a chicken reaches the ship. Defaults
                to 50.0.
            clear_reward: Paid when the last chicken dies. Defaults to 50.0.
            max_steps: Transitions allowed per episode. Defaults to 60.
            discount_factor: Discount factor. Defaults to 0.95.
            name: Environment name. Defaults to ``"CrazyChicken"``.
            output_dir: Output directory for logging. Defaults to ``None``.
            debug: Enable debug logging. Defaults to ``False``.
            use_queue_logger: Whether to use queue-based logging.

        Raises:
            ValueError: If the geometry, the sensor model or the episode limits
                are outside their valid ranges.
        """
        if int(num_columns) < 2:
            raise ValueError(
                "num_columns must be at least 2: a one-column grid gives a patrolling "
                f"chicken nowhere to bounce between, got {num_columns}"
            )
        if int(num_rows) < 2:
            raise ValueError(
                "num_rows must be at least 2: row 0 is the ship's, so a chicken needs "
                f"at least one row of its own, got {num_rows}"
            )
        if int(num_chickens) < 1:
            raise ValueError(f"num_chickens must be at least 1, got {num_chickens}")
        # Placements are drawn without replacement so that one projectile can
        # never be worth two kills, which would put the reward above its
        # declared maximum.
        available_cells = (int(num_rows) - 1) * int(num_columns)
        if int(num_chickens) > available_cells:
            raise ValueError(
                f"{num_chickens} chickens do not fit on {available_cells} start cells "
                f"({num_rows - 1} rows above the ship by {num_columns} columns)"
            )
        if int(fire_cooldown) < 0:
            raise ValueError(f"fire_cooldown must be non-negative, got {fire_cooldown}")
        if int(max_steps) < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        for label, probability in (
            ("dive_probability", dive_probability),
            ("initial_dive_probability", initial_dive_probability),
            ("camera_detection_probability", camera_detection_probability),
            ("radar_detection_probability", radar_detection_probability),
            ("drop_flag_error_probability", drop_flag_error_probability),
        ):
            if not 0.0 <= float(probability) <= 1.0:
                raise ValueError(f"{label} must be in [0, 1], got {probability}")
        for label, deviation in (
            ("ship_column_noise_std", ship_column_noise_std),
            ("camera_offset_noise_std", camera_offset_noise_std),
            ("radar_range_noise_std", radar_range_noise_std),
        ):
            if float(deviation) < 0.0:
                raise ValueError(f"{label} must be non-negative, got {deviation}")
        if float(camera_slope) < 0.0:
            raise ValueError(f"camera_slope must be non-negative, got {camera_slope}")
        if float(radar_radius) < 0.0:
            raise ValueError(f"radar_radius must be non-negative, got {radar_radius}")
        for label, amount in (
            ("kill_reward", kill_reward),
            ("shot_cost", shot_cost),
            ("step_cost", step_cost),
            ("ship_hit_penalty", ship_hit_penalty),
            ("clear_reward", clear_reward),
        ):
            if float(amount) < 0.0:
                raise ValueError(
                    f"{label} must be non-negative; it is a magnitude and the reward "
                    f"formula already carries its sign, got {amount}"
                )

        observation_mode = resolve_observation_mode(observation_mode)

        # The bound is enumerated rather than estimated, because a wrong reward
        # range is the single most repeated bug in this repository.
        #
        # Best step: no shot fired (firing only ever subtracts), every
        # projectile in flight connects, and that empties the flock. At most one
        # projectile exists per column, and chickens never share a cell, so a
        # single step cannot kill more than ``min(num_chickens, num_columns)``.
        # The clear bonus is paid in the same step as the kills that earned it.
        #
        # Worst step: the step cost, a shot that was genuinely fired, and a
        # chicken reaching the ship, with the kill and clear terms contributing
        # nothing. This bound is deliberately *conservative* rather than tight.
        # Under the current kill rule those last two cannot actually coincide: a
        # chicken can only reach the ship by diving into its column, a shot is
        # launched into that same column, and the rising shot meets the falling
        # chicken -- so firing protects the ship's column for exactly the step
        # it is fired. Tightening the bound to
        # ``-step_cost - max(shot_cost, ship_hit_penalty)`` would make it depend
        # on that argument, and a later change to how a projectile and a dive
        # resolve would then silently put a real reward outside the declared
        # range. A declared range that is one shot cost too wide costs nothing.
        #
        # The clear bonus and the ship hit genuinely cannot coincide, which is
        # why the maximum above leaves the penalty out: a chicken that reaches
        # the ship is alive, so the flock is not clear.
        most_kills_in_one_step = min(int(num_chickens), int(num_columns))
        max_reward = float(kill_reward) * most_kills_in_one_step + float(clear_reward)
        max_reward -= float(step_cost)
        min_reward = -float(step_cost) - float(shot_cost) - float(ship_hit_penalty)

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

        self.num_columns = int(num_columns)
        self.num_rows = int(num_rows)
        self.num_chickens = int(num_chickens)
        self.fire_cooldown = int(fire_cooldown)
        self.dive_probability = float(dive_probability)
        self.initial_dive_probability = float(initial_dive_probability)
        self.camera_detection_probability = float(camera_detection_probability)
        self.radar_detection_probability = float(radar_detection_probability)
        self.ship_column_noise_std = float(ship_column_noise_std)
        self.camera_offset_noise_std = float(camera_offset_noise_std)
        self.radar_range_noise_std = float(radar_range_noise_std)
        self.drop_flag_error_probability = float(drop_flag_error_probability)
        self.camera_slope = float(camera_slope)
        self.radar_radius = float(radar_radius)
        self.observation_mode = observation_mode
        self.kill_reward = float(kill_reward)
        self.shot_cost = float(shot_cost)
        self.step_cost = float(step_cost)
        self.ship_hit_penalty = float(ship_hit_penalty)
        self.clear_reward = float(clear_reward)
        self.max_steps = int(max_steps)

        #: The ship starts in the middle, which is the only column from which
        #: both walls are the same distance away.
        self.ship_start_column = self.num_columns // 2
        self.state_size = state_size(self.num_chickens, self.num_columns)
        self.observation_size = (
            self.state_size
            if self.observation_mode is ObservationMode.FULL
            else observation_size(self.num_chickens)
        )
        #: Largest Chebyshev distance any chicken can be from the ship, which is
        #: what the encroachment channel is measured back from.
        self.max_chicken_distance = float(max(self.num_columns - 1, self.num_rows - 1))

    # -- state accessors ------------------------------------------------

    def chickens(self, state: CrazyChickenState) -> np.ndarray:
        """Return the ``(num_chickens, 5)`` chicken block of ``state`` as a view.

        Args:
            state: A state vector.

        Returns:
            Rows of ``(column, row, direction, mode, alive)``.
        """
        return chicken_slots(np.asarray(state, dtype=np.float64), self.num_chickens)

    def projectiles(self, state: CrazyChickenState) -> np.ndarray:
        """Return the per-column projectile rows of ``state`` as a view.

        Args:
            state: A state vector.

        Returns:
            One row per column, :data:`NO_PROJECTILE` where empty.
        """
        return projectile_rows(np.asarray(state, dtype=np.float64), self.num_chickens)

    def ship_column(self, state: CrazyChickenState) -> int:
        """Return the ship's column in ``state``.

        Args:
            state: A state vector.

        Returns:
            The column index.
        """
        return int(round(float(np.asarray(state, dtype=np.float64)[SHIP_COLUMN_INDEX])))

    def live_chicken_count(self, state: CrazyChickenState) -> int:
        """Return how many chicken slots in ``state`` are still alive.

        Args:
            state: A state vector.

        Returns:
            The count, between 0 and ``num_chickens``.
        """
        return int(np.count_nonzero(self.chickens(state)[:, CHICKEN_ALIVE] > 0.0))

    # -- dynamics -------------------------------------------------------

    def get_actions(self) -> List[int]:
        """Return the four actions: fire, left, right, stay."""
        return [int(action) for action in CrazyChickenAction]

    def hash_action(self, action: Any) -> Hashable:
        """Return a hashable key for an action (already an int)."""
        return int(action)

    def fires(self, state: CrazyChickenState, action: Any) -> bool:
        """Whether ``action`` actually launches a projectile from ``state``.

        A ``FIRE`` that the cooldown blocks, or that finds this column already
        holding a projectile, does exactly what ``STAY`` does and is charged
        nothing. This is deterministic in ``(state, action)``, which is why the
        shot cost can be scored without a successor.

        Args:
            state: The state fired from.
            action: The action taken.

        Returns:
            ``True`` if a projectile leaves the ship.
        """
        if int(action) != int(CrazyChickenAction.FIRE):
            return False
        values = np.asarray(state, dtype=np.float64)
        if values[COOLDOWN_INDEX] > 0.0:
            return False
        column = self.ship_column(values)
        return bool(self.projectiles(values)[column] == NO_PROJECTILE)

    def _moved_ship_column(self, state: CrazyChickenState, action: Any) -> int:
        """Where the ship ends up, clamped at the walls."""
        delta = 0
        if int(action) == int(CrazyChickenAction.LEFT):
            delta = -1
        elif int(action) == int(CrazyChickenAction.RIGHT):
            delta = 1
        return int(min(max(self.ship_column(state) + delta, 0), self.num_columns - 1))

    def _draw_dive_switches(self) -> np.ndarray:
        """Flip one dive coin per chicken slot.

        One draw per slot, alive or not, so the number of random numbers a step
        consumes never depends on how the episode is going. A varying draw count
        would make two seeded runs diverge the moment one of them lost a chicken
        earlier than the other.

        Returns:
            A boolean array of length ``num_chickens``.
        """
        return np.random.random(self.num_chickens) < self.dive_probability

    # pylint: disable-next=too-many-locals
    def _step_once(
        self,
        state: CrazyChickenState,
        action: Any,
        switches: Optional[np.ndarray] = None,
    ) -> CrazyChickenState:
        """Advance one step, with this step's dive coins drawn or supplied.

        The order is: flip the dive coins, move the ship, move the chickens,
        move the projectiles and launch any new one, resolve kills, then resolve
        whatever reached row 0. Dives are decided before the chickens move so
        that a chicken which switches this step also drops this step.

        Args:
            state: The state to advance.
            action: The action taken.
            switches: Dive coins to use instead of drawing them, one per slot.
                :meth:`transition_log_probability` passes these to replay a
                candidate successor without touching the RNG. Defaults to
                ``None``, which draws.

        Returns:
            A fresh successor state vector.
        """
        values = np.asarray(state, dtype=np.float64)
        successor = np.array(values, dtype=np.float64, copy=True)
        successor[STEP_INDEX] = values[STEP_INDEX] + 1.0

        fired = self.fires(values, action)
        ship_column = self._moved_ship_column(values, action)
        successor[SHIP_COLUMN_INDEX] = float(ship_column)
        successor[COOLDOWN_INDEX] = (
            float(self.fire_cooldown) if fired else max(values[COOLDOWN_INDEX] - 1.0, 0.0)
        )

        flock = chicken_slots(successor, self.num_chickens)
        alive = flock[:, CHICKEN_ALIVE] > 0.0
        coins = self._draw_dive_switches() if switches is None else np.asarray(switches, dtype=bool)
        flock[alive & (flock[:, CHICKEN_MODE] == MODE_PATROL) & coins, CHICKEN_MODE] = MODE_DIVE

        old_rows = np.array(flock[:, CHICKEN_ROW], copy=True)
        diving = alive & (flock[:, CHICKEN_MODE] == MODE_DIVE)
        patrolling = alive & ~diving
        flock[diving, CHICKEN_ROW] -= 1.0
        if np.any(patrolling):
            columns = flock[patrolling, CHICKEN_COLUMN]
            directions = flock[patrolling, CHICKEN_DIRECTION]
            stepped = columns + directions
            # Bouncing reverses the direction *and then* takes the step, so a
            # chicken at a wall turns and moves in one step rather than spending
            # a step standing still against it.
            directions = np.where(
                (stepped < 0) | (stepped > self.num_columns - 1), -directions, directions
            )
            flock[patrolling, CHICKEN_DIRECTION] = directions
            flock[patrolling, CHICKEN_COLUMN] = columns + directions

        sky = projectile_rows(successor, self.num_chickens)
        old_projectiles = np.array(sky, copy=True)
        in_flight = old_projectiles != NO_PROJECTILE
        sky[in_flight] = old_projectiles[in_flight] + 1.0
        sky[in_flight & (old_projectiles + 1.0 > self.num_rows - 1)] = NO_PROJECTILE
        if fired:
            old_projectiles[ship_column] = 0.0
            sky[ship_column] = 1.0

        self._resolve_kills(flock, sky, old_projectiles, old_rows)
        self._resolve_arrivals(flock, ship_column, successor)
        return successor

    def _resolve_kills(
        self,
        flock: np.ndarray,
        sky: np.ndarray,
        old_projectiles: np.ndarray,
        old_rows: np.ndarray,
    ) -> None:
        """Kill every chicken a projectile reached, and the projectile with it.

        A projectile and a chicken meet when they end the step in the same cell
        *or* when they swapped past each other -- a chicken diving down through
        a projectile rising up would otherwise pass through it untouched, which
        is the one case a naive same-cell test gets visibly wrong.

        The column tested is the chicken's column after it moved. A patrolling
        chicken that steps sideways into a rising projectile's column is hit;
        one that steps out of it is not.

        A projectile kills at most one chicken -- it is destroyed by the first
        one it reaches, which is the *lowest* candidate rather than the one in
        the lowest-numbered slot. The two differ only when two chickens share a
        column within the one row a projectile crosses, but picking by slot
        index there would make the outcome depend on the order the flock happens
        to be stored in, which is not a fact about the world.

        Args:
            flock: The successor's chicken block, modified in place.
            sky: The successor's projectile rows, modified in place.
            old_projectiles: Projectile rows before they rose, with the newly
                fired shot recorded at row 0.
            old_rows: Chicken rows before they moved.
        """
        for column in range(self.num_columns):
            if sky[column] == NO_PROJECTILE:
                continue
            new_row = sky[column]
            old_row = old_projectiles[column]
            struck = -1
            for index in range(self.num_chickens):
                if flock[index, CHICKEN_ALIVE] <= 0.0:
                    continue
                if flock[index, CHICKEN_COLUMN] != column:
                    continue
                if new_row < flock[index, CHICKEN_ROW] or old_row > old_rows[index]:
                    continue
                if struck < 0 or flock[index, CHICKEN_ROW] < flock[struck, CHICKEN_ROW]:
                    struck = index
            if struck >= 0:
                flock[struck, CHICKEN_ALIVE] = 0.0
                sky[column] = NO_PROJECTILE

    def _resolve_arrivals(self, flock: np.ndarray, ship_column: int, successor: np.ndarray) -> None:
        """Settle every live chicken that reached row 0.

        One in the ship's column destroys it and ends the episode. One anywhere
        else pulls up: back to patrolling at the top row, same column, same
        direction. See the module docstring for why a pull-up exists at all.

        Args:
            flock: The successor's chicken block, modified in place.
            ship_column: The ship's column after its move.
            successor: The successor state, whose ship-hit flag is set here.
        """
        for index in range(self.num_chickens):
            if flock[index, CHICKEN_ALIVE] <= 0.0 or flock[index, CHICKEN_ROW] > 0.0:
                continue
            if int(flock[index, CHICKEN_COLUMN]) == int(ship_column):
                flock[index, CHICKEN_ROW] = 0.0
                successor[SHIP_HIT_INDEX] = 1.0
                continue
            flock[index, CHICKEN_ROW] = float(self.num_rows - 1)
            flock[index, CHICKEN_MODE] = MODE_PATROL

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> Any:
        """Sample one or more successors of ``(state, action)``.

        Args:
            state: The state to advance.
            action: The action taken.
            n_samples: How many independent successors to draw. Defaults to 1.

        Returns:
            One ``float64`` state vector, or an ``(n_samples, state_size)``
            array.
        """
        if int(n_samples) == 1:
            return self._step_once(state, action)
        return np.asarray([self._step_once(state, action) for _ in range(int(n_samples))])

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        """Log-probability of each candidate successor of ``(state, action)``.

        Everything except the dive coins is deterministic, so a candidate is
        scored by rebuilding it from the coins it implies and checking that the
        rebuild is exactly the candidate. A candidate that no set of coins
        produces scores :data:`IMPOSSIBLE_LOG_PROBABILITY`.

        Args:
            state: The state advanced from.
            action: The action taken.
            next_states: Candidate successors.

        Returns:
            One log-probability per candidate, shape ``(N,)``.
        """
        candidates = np.atleast_2d(np.asarray(next_states, dtype=np.float64))
        scores = np.full(len(candidates), IMPOSSIBLE_LOG_PROBABILITY, dtype=np.float64)
        before = chicken_slots(np.asarray(state, dtype=np.float64), self.num_chickens)
        for index, candidate in enumerate(candidates):
            if candidate.shape != (self.state_size,):
                continue
            switched = self._implied_dive_switches(before, candidate)
            if switched is None:
                continue
            rebuilt = self._step_once(state, action, switches=switched)
            if not np.array_equal(rebuilt, candidate):
                continue
            scores[index] = self._dive_coin_log_probability(before, switched)
        return scores

    def _implied_dive_switches(
        self, before: np.ndarray, candidate: np.ndarray
    ) -> Optional[np.ndarray]:
        """Which chickens must have switched into a dive to reach ``candidate``.

        A chicken that is dead in the candidate, or that pulled up, tells us
        nothing about its coin, so its mode is read back from whether the
        candidate's rebuild matches -- which the caller checks. Here only the
        unambiguous reading is returned: a live chicken that was patrolling and
        is now diving must have switched.

        Args:
            before: The prior state's chicken block.
            candidate: A candidate successor state.

        Returns:
            A boolean array of implied switches, or ``None`` when the candidate
            is malformed.
        """
        after = chicken_slots(candidate, self.num_chickens)
        if after.shape != before.shape:
            return None
        was_patrolling = before[:, CHICKEN_MODE] == MODE_PATROL
        # A dead-or-pulled-up chicken's stored mode is patrol whatever its coin
        # was, so the only coin that can be read back is one that left a live
        # diving chicken behind. Every other slot is assumed not to have
        # switched, and the caller's exact rebuild check rejects the reading if
        # that assumption was wrong.
        now_diving = after[:, CHICKEN_MODE] == MODE_DIVE
        return was_patrolling & now_diving

    def _dive_coin_log_probability(self, before: np.ndarray, switches: np.ndarray) -> float:
        """Log-probability of the dive coins ``switches`` from a prior block.

        Only a live, patrolling chicken has a coin that matters; every other
        slot's coin is drawn but discarded, so it contributes nothing.
        """
        eligible = (before[:, CHICKEN_ALIVE] > 0.0) & (before[:, CHICKEN_MODE] == MODE_PATROL)
        if self.dive_probability <= 0.0:
            return 0.0 if not np.any(switches & eligible) else IMPOSSIBLE_LOG_PROBABILITY
        if self.dive_probability >= 1.0:
            return 0.0 if np.all(switches[eligible]) else IMPOSSIBLE_LOG_PROBABILITY
        switched = int(np.count_nonzero(switches & eligible))
        held = int(np.count_nonzero(eligible)) - switched
        return switched * math.log(self.dive_probability) + held * math.log(
            1.0 - self.dive_probability
        )

    # -- observations ---------------------------------------------------

    def _offsets(self, next_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(column offsets, row distances, alive mask)`` for a state."""
        flock = chicken_slots(next_state, self.num_chickens)
        column_offsets = flock[:, CHICKEN_COLUMN] - next_state[SHIP_COLUMN_INDEX]
        row_distances = flock[:, CHICKEN_ROW]
        return column_offsets, row_distances, flock[:, CHICKEN_ALIVE] > 0.0

    def sensor_reach(self, next_state: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Which chickens each sensor could report from ``next_state``.

        A dead chicken is inside neither reach, which is what makes a cleared
        slot silent rather than merely unlucky.

        Args:
            next_state: The realised successor state.

        Returns:
            ``(camera reach, radar reach)``, each a boolean array of length
            ``num_chickens``.
        """
        values = np.asarray(next_state, dtype=np.float64)
        column_offsets, row_distances, alive = self._offsets(values)
        camera = alive & camera_sees(column_offsets, row_distances, self.camera_slope)
        radar = alive & radar_sees(column_offsets, row_distances, self.radar_radius)
        return camera, radar

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        """Draw one or more readings of ``next_state``.

        The reading depends on the successor only, never on the action that
        produced it.

        Args:
            next_state: The realised successor state.
            action: Ignored; present for the API.
            n_samples: How many independent readings to draw. Defaults to 1.

        Returns:
            One ``float64`` observation vector, or an
            ``(n_samples, observation_size)`` array.
        """
        del action
        if int(n_samples) == 1:
            return self._sample_observation_once(next_state)
        return np.asarray([self._sample_observation_once(next_state) for _ in range(n_samples)])

    def _sample_observation_once(self, next_state: Any) -> np.ndarray:
        """Draw one reading of ``next_state``."""
        values = np.asarray(next_state, dtype=np.float64)
        if self.observation_mode is ObservationMode.FULL:
            return np.array(values, dtype=np.float64, copy=True)

        column_offsets, row_distances, _ = self._offsets(values)
        camera, radar = self.sensor_reach(values)
        modes = chicken_slots(values, self.num_chickens)[:, CHICKEN_MODE]

        observation = np.zeros(self.observation_size, dtype=np.float64)
        observation[OBSERVED_SHIP_COLUMN_INDEX] = float(
            sample_rounded_normal(values[SHIP_COLUMN_INDEX], self.ship_column_noise_std)
        )
        for index in range(self.num_chickens):
            base = OBSERVATION_SHIP_WIDTH + OBSERVATION_CHICKEN_WIDTH * index
            if camera[index] and np.random.random() < self.camera_detection_probability:
                observation[base + OBSERVED_CAMERA_REPORTED] = 1.0
                observation[base + OBSERVED_CAMERA_OFFSET] = float(
                    sample_rounded_normal(column_offsets[index], self.camera_offset_noise_std)
                )
            if radar[index] and np.random.random() < self.radar_detection_probability:
                observation[base + OBSERVED_RADAR_REPORTED] = 1.0
                observation[base + OBSERVED_RADAR_ROWS] = float(
                    sample_rounded_normal(row_distances[index], self.radar_range_noise_std)
                )
                true_drop = -1.0 if modes[index] == MODE_DIVE else 0.0
                flipped = np.random.random() < self.drop_flag_error_probability
                observation[base + OBSERVED_RADAR_DROP] = (
                    (-1.0 - true_drop) if flipped else true_drop
                )
        return observation

    def _observation_log_likelihood(self, next_state: np.ndarray, observation: np.ndarray) -> float:
        """Log ``Z(o | s')`` for one successor and one reading.

        The single implementation every likelihood path goes through, so the
        batched and scalar entry points cannot drift apart.
        """
        if self.observation_mode is ObservationMode.FULL:
            return 0.0 if np.array_equal(next_state, observation) else IMPOSSIBLE_LOG_PROBABILITY
        if observation.shape != (self.observation_size,):
            return IMPOSSIBLE_LOG_PROBABILITY

        column_offsets, row_distances, _ = self._offsets(next_state)
        camera, radar = self.sensor_reach(next_state)
        modes = chicken_slots(next_state, self.num_chickens)[:, CHICKEN_MODE]

        total = self._log(
            float(
                rounded_normal_pmf(
                    observation[OBSERVED_SHIP_COLUMN_INDEX],
                    next_state[SHIP_COLUMN_INDEX],
                    self.ship_column_noise_std,
                )
            )
        )
        for index in range(self.num_chickens):
            base = OBSERVATION_SHIP_WIDTH + OBSERVATION_CHICKEN_WIDTH * index
            total += self._camera_log_factor(
                reported=observation[base + OBSERVED_CAMERA_REPORTED] > 0.0,
                reading=observation[base + OBSERVED_CAMERA_OFFSET],
                in_reach=bool(camera[index]),
                truth=float(column_offsets[index]),
            )
            total += self._radar_log_factor(
                reported=observation[base + OBSERVED_RADAR_REPORTED] > 0.0,
                rows=observation[base + OBSERVED_RADAR_ROWS],
                drop=observation[base + OBSERVED_RADAR_DROP],
                in_reach=bool(radar[index]),
                truth=float(row_distances[index]),
                true_drop=-1.0 if modes[index] == MODE_DIVE else 0.0,
            )
            if total <= IMPOSSIBLE_LOG_PROBABILITY:
                return IMPOSSIBLE_LOG_PROBABILITY
        return float(total)

    def _camera_log_factor(
        self, reported: bool, reading: float, in_reach: bool, truth: float
    ) -> float:
        """One chicken's camera factor, in log space."""
        if not in_reach:
            return 0.0 if not reported else IMPOSSIBLE_LOG_PROBABILITY
        if not reported:
            return self._log(1.0 - self.camera_detection_probability)
        return self._log(self.camera_detection_probability) + self._log(
            float(rounded_normal_pmf(reading, truth, self.camera_offset_noise_std))
        )

    # pylint: disable-next=too-many-arguments
    def _radar_log_factor(
        self,
        reported: bool,
        rows: float,
        drop: float,
        in_reach: bool,
        truth: float,
        true_drop: float,
    ) -> float:
        """One chicken's radar factor, in log space."""
        if not in_reach:
            return 0.0 if not reported else IMPOSSIBLE_LOG_PROBABILITY
        if not reported:
            return self._log(1.0 - self.radar_detection_probability)
        flag = (
            1.0 - self.drop_flag_error_probability
            if drop == true_drop
            else self.drop_flag_error_probability
        )
        return (
            self._log(self.radar_detection_probability)
            + self._log(float(rounded_normal_pmf(rows, truth, self.radar_range_noise_std)))
            + self._log(flag)
        )

    @staticmethod
    def _log(value: float) -> float:
        """Natural log, with zero mapped to the impossible floor."""
        return math.log(value) if value > 0.0 else IMPOSSIBLE_LOG_PROBABILITY

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        """Log-likelihood of each candidate reading under one successor.

        Args:
            next_state: The realised successor state.
            action: Ignored; the reading does not depend on it.
            observations: Candidate readings.

        Returns:
            One log-probability per candidate, shape ``(N,)``.
        """
        del action
        successor = np.asarray(next_state, dtype=np.float64)
        candidates = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        return np.asarray(
            [self._observation_log_likelihood(successor, candidate) for candidate in candidates],
            dtype=np.float64,
        )

    def observation_log_probability_single(
        self, next_state: Any, action: Any, observation: Any
    ) -> float:
        """Scalar log-likelihood for one ``(successor, reading)`` pair."""
        del action
        return self._observation_log_likelihood(
            np.asarray(next_state, dtype=np.float64),
            np.asarray(observation, dtype=np.float64),
        )

    def observation_log_probability_per_state(
        self, next_states: Any, action: Any, observation: Any
    ) -> np.ndarray:
        """Log-likelihood of one reading under each candidate successor."""
        del action
        candidate = np.asarray(observation, dtype=np.float64)
        return np.asarray(
            [
                self._observation_log_likelihood(np.asarray(state, dtype=np.float64), candidate)
                for state in next_states
            ],
            dtype=np.float64,
        )

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        """Check whether two readings are the same vector of numbers."""
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
            Its raw ``float64`` bytes, which is the standard surrogate here for
            an ndarray observation and matches ``array_equal``. Masked fields
            are stored as zeros precisely so that two readings which report the
            same things hash alike.
        """
        return np.ascontiguousarray(observation, dtype=np.float64).tobytes()

    # -- reward ---------------------------------------------------------

    @property
    def reward_requires_next_state(self) -> bool:
        """Kills, the clear bonus and the ship hit are all decided by the transition."""
        return True

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        """Score one step: kills and the clear bonus, less the shot and step costs.

        Without a successor only the part that is already determined can be
        scored -- the step cost and, if the shot really leaves the ship, its
        cost. That fallback is deliberately not an expectation over kills: a
        planner asking for the reward of a belief node gets a number it can
        compare across actions (firing is charged, missing is not), and the
        realised kill is scored on the step it actually happens.

        Args:
            state: The state the step was taken from.
            action: The action taken.
            next_state: The realised successor, or ``None``.

        Returns:
            The immediate reward.
        """
        charged = -self.step_cost - (self.shot_cost if self.fires(state, action) else 0.0)
        if next_state is None:
            return float(charged)
        before = self.live_chicken_count(state)
        after = self.live_chicken_count(next_state)
        total = charged + self.kill_reward * float(before - after)
        was_hit = bool(np.asarray(state, dtype=np.float64)[SHIP_HIT_INDEX] > 0.0)
        is_hit = bool(np.asarray(next_state, dtype=np.float64)[SHIP_HIT_INDEX] > 0.0)
        # Charged on the step the ship is lost, not on every step after it: the
        # flag stays set, and a driver that kept stepping a terminal state would
        # otherwise bill the penalty again and again.
        if is_hit and not was_hit:
            total -= self.ship_hit_penalty
        if after == 0 and before > 0:
            total += self.clear_reward
        return float(total)

    # -- terminal / initial ---------------------------------------------

    def is_terminal(self, state: Any) -> bool:
        """Whether the flock is cleared, the ship is gone, or time is up.

        Args:
            state: A state vector.

        Returns:
            ``True`` in any of those three cases.
        """
        values = np.asarray(state, dtype=np.float64)
        if values[SHIP_HIT_INDEX] > 0.0:
            return True
        if int(round(float(values[STEP_INDEX]))) >= self.max_steps:
            return True
        return self.live_chicken_count(values) == 0

    def initial_state_dist(self) -> Distribution:
        """A fresh flock per episode, the ship in the middle with an empty sky."""
        return CrazyChickenInitialStateDistribution(
            num_chickens=self.num_chickens,
            num_columns=self.num_columns,
            num_rows=self.num_rows,
            ship_column=self.ship_start_column,
            lowest_start_row=1,
            initial_dive_probability=self.initial_dive_probability,
        )

    def initial_observation_dist(self) -> DiscreteDistribution:
        """The pre-sensor reading: the ship's known column and no chicken reports.

        Returns:
            A point mass on a reading that names the ship's start column and
            leaves every chicken slot silent. It is a sentinel taken before any
            sensor has run, not a claim that the sky is empty, and the belief
            never weights particles with it.
        """
        if self.observation_mode is ObservationMode.FULL:
            observation = make_state(
                num_chickens=self.num_chickens,
                num_columns=self.num_columns,
                ship_column=self.ship_start_column,
                chickens=np.zeros((self.num_chickens, 5), dtype=np.float64),
            )
        else:
            observation = np.zeros(self.observation_size, dtype=np.float64)
            observation[OBSERVED_SHIP_COLUMN_INDEX] = float(self.ship_start_column)
        return DiscreteDistribution(values=[observation], probs=np.array([1.0]))

    # -- metrics --------------------------------------------------------

    def step_info(self, state: Any, action: Any, next_state: Any) -> Dict[str, float]:
        """Report the per-step channels this environment's metrics are built on.

        Draws no randomness: every channel is a deterministic function of the
        arguments, and the one channel that would need the transition -- whether
        a shot left the ship -- is computed from ``(state, action)``, which is
        where firing is decided.

        Outcome channels are read from ``next_state`` when there is one, for the
        reason Battleship and occupancy-grid mapping both document: the episode
        runner checks its step budget before it checks terminality, so an
        episode whose final allowed step clears the flock records no terminal
        bookkeeping step, and reading the outcome from ``state`` alone would
        score that win as a timeout.

        Args:
            state: The state the step was taken from, or the final state on the
                terminal bookkeeping step.
            action: The action taken, or ``None`` on the terminal step.
            next_state: The realised successor, or ``None`` on the terminal step.

        Returns:
            The channels named by :class:`CrazyChickenStepChannel`.
        """
        subject = np.asarray(state if next_state is None else next_state, dtype=np.float64)
        cleared = float(self.live_chicken_count(subject) == 0)
        destroyed = float(subject[SHIP_HIT_INDEX] > 0.0)

        killed = 0.0
        if action is not None and next_state is not None:
            killed = float(self.live_chicken_count(state) - self.live_chicken_count(next_state))
        fired = float(action is not None and self.fires(state, action))

        return {
            CrazyChickenStepChannel.FLOCK_CLEARED.value: cleared,
            CrazyChickenStepChannel.SHIP_DESTROYED.value: destroyed,
            CrazyChickenStepChannel.STILL_RUNNING.value: float(cleared == 0.0 and destroyed == 0.0),
            CrazyChickenStepChannel.RECORDED_STEP.value: 1.0,
            CrazyChickenStepChannel.CHICKENS_KILLED.value: killed,
            CrazyChickenStepChannel.SHOT_FIRED.value: fired,
            CrazyChickenStepChannel.CHICKEN_ENCROACHMENT_CELLS.value: self._encroachment(subject),
        }

    def _encroachment(self, state: np.ndarray) -> float:
        """How far the nearest live chicken has pushed in, in cells.

        Reported this way round -- large means close -- because the episode
        reduction this feeds is ``MAX`` and there is no ``MIN``. It is
        ``max_chicken_distance`` minus the smallest Chebyshev distance from the
        ship to a live chicken, so the distance itself is recoverable by
        subtracting the metric from ``max_chicken_distance``. Chebyshev rather
        than Euclidean so the value is a whole number of cells and can never go
        negative.

        A state with no live chickens reports ``0.0``: the flock is gone, so
        nothing is encroaching. That is a real reading rather than a missing
        one, which is why the channel is emitted on every step.
        """
        flock = chicken_slots(state, self.num_chickens)
        alive = flock[:, CHICKEN_ALIVE] > 0.0
        if not np.any(alive):
            return 0.0
        column_offsets = np.abs(flock[alive, CHICKEN_COLUMN] - state[SHIP_COLUMN_INDEX])
        distances = np.maximum(column_offsets, flock[alive, CHICKEN_ROW])
        return float(self.max_chicken_distance - float(np.min(distances)))

    def get_metric_specs(self) -> List[StepInfoMetric]:
        """Declare the Crazy Chicken metrics that a per-step channel can express.

        Completion reduces with ``ANY``: clearing the flock happens once and
        ends the episode, so it cannot be undone by a later step. The three
        ``ended_by_*`` channels reduce with ``LAST`` and sum to one per episode,
        because ``still_running`` is exactly the complement of the other two.

        The danger here is a chicken getting close, and it is reported both ways
        the metrics skill asks for: ``average_hits_taken`` counts the times the
        flock actually got through (at most one, since it ends the episode), and
        ``max_chicken_encroachment_cells`` is the severity -- how close the
        nearest one ever got, which separates a planner that was never
        threatened from one that survived by a cell.

        Returns:
            One spec per metric that a channel reduction can produce.
            ``shot_accuracy`` is not among them; see :meth:`compute_metrics`.
        """
        return [
            StepInfoMetric(
                name=CrazyChickenMetrics.TASK_COMPLETION_RATE.value,
                channel=CrazyChickenStepChannel.FLOCK_CLEARED.value,
                per_episode=EpisodeReduction.ANY,
            ),
            StepInfoMetric(
                name=CrazyChickenMetrics.ENDED_BY_GOAL.value,
                channel=CrazyChickenStepChannel.FLOCK_CLEARED.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=CrazyChickenMetrics.ENDED_BY_FAILURE.value,
                channel=CrazyChickenStepChannel.SHIP_DESTROYED.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=CrazyChickenMetrics.ENDED_BY_TIMEOUT.value,
                channel=CrazyChickenStepChannel.STILL_RUNNING.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=CrazyChickenMetrics.AVERAGE_EPISODE_LENGTH.value,
                channel=CrazyChickenStepChannel.RECORDED_STEP.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=CrazyChickenMetrics.AVERAGE_CHICKENS_KILLED.value,
                channel=CrazyChickenStepChannel.CHICKENS_KILLED.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=CrazyChickenMetrics.AVERAGE_SHOTS_FIRED.value,
                channel=CrazyChickenStepChannel.SHOT_FIRED.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=CrazyChickenMetrics.AVERAGE_HITS_TAKEN.value,
                channel=CrazyChickenStepChannel.SHIP_DESTROYED.value,
                per_episode=EpisodeReduction.MAX,
            ),
            StepInfoMetric(
                name=CrazyChickenMetrics.MAX_CHICKEN_ENCROACHMENT_CELLS.value,
                channel=CrazyChickenStepChannel.CHICKEN_ENCROACHMENT_CELLS.value,
                per_episode=EpisodeReduction.MAX,
            ),
        ]

    def get_metric_names(self) -> List[str]:
        """Every metric this environment produces, including ``shot_accuracy``."""
        return [spec.name for spec in self.get_metric_specs()] + [
            CrazyChickenMetrics.SHOT_ACCURACY.value
        ]

    def compute_metrics(self, histories: "List[History]") -> List[MetricValue]:
        """Aggregate the declared channels, then add the one ratio among them.

        ``shot_accuracy`` is kills per shot, and a ratio of two per-episode sums
        is the one shape :class:`StepInfoMetric` cannot express: a reduction
        collapses a single channel, and a mean of per-step ratios is not the
        episode's ratio. It is therefore computed here, from the same two
        channels the counts are built from, so the three numbers cannot
        disagree.

        Episodes that never fired are left out of the average rather than scored
        as zero, because zero kills from zero shots says nothing about aim. When
        *no* episode fired the ratio is undefined for the batch as a whole; it
        is reported as 0.0 with an unbounded interval, which is what the shared
        aggregator reports for any metric it cannot put an interval on.

        Args:
            histories: Episode histories to analyse.

        Returns:
            The channel-derived metrics followed by ``shot_accuracy``.
        """
        require_non_empty_histories(histories, type(self).__name__)
        metrics = list(super().compute_metrics(histories))
        ratios: List[float] = []
        for episode in extract_episode_step_infos(histories):
            shots = sum(
                float(info.get(CrazyChickenStepChannel.SHOT_FIRED.value, 0.0)) for info in episode
            )
            if shots <= 0.0:
                continue
            kills = sum(
                float(info.get(CrazyChickenStepChannel.CHICKENS_KILLED.value, 0.0))
                for info in episode
            )
            ratios.append(kills / shots)
        if not ratios:
            metrics.append(
                MetricValue(CrazyChickenMetrics.SHOT_ACCURACY.value, 0.0, -np.inf, np.inf)
            )
            return metrics
        lower, upper = (
            confidence_interval(ratios)
            if len(ratios) >= _MIN_EPISODES_FOR_CONFIDENCE_INTERVAL
            else (-np.inf, np.inf)
        )
        metrics.append(
            MetricValue(
                CrazyChickenMetrics.SHOT_ACCURACY.value,
                float(np.mean(ratios)),
                float(lower),
                float(upper),
            )
        )
        return metrics

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
        # almost none of them render anything, so the renderer's sprites and
        # palette tables stay out of a planning run's memory.
        # pylint: disable-next=import-outside-toplevel
        from POMDPPlanners.environments.crazy_chicken_pomdp.crazy_chicken_visualizer import (
            CrazyChickenVisualizer,
        )

        cache_path = output_dir / f"crazy_chicken_{episode_index}.gif"
        CrazyChickenVisualizer(self).create_visualization(history, cache_path)


def noiseless_preset(**overrides: Any) -> Dict[str, Any]:
    """Constructor keywords for the deterministic-sensor preset.

    Every sensor always reports, adds no noise and never inverts the drop flag,
    so the observation becomes a deterministic function of the successor state.
    The transition is untouched: dives are still drawn, which is what keeps this
    a POMDP whose only certainty is the reading.

    Args:
        **overrides: Values merged on top (overrides win).

    Returns:
        Keyword arguments for :class:`CrazyChickenPOMDP`.
    """
    preset: Dict[str, Any] = {
        "camera_detection_probability": 1.0,
        "radar_detection_probability": 1.0,
        "ship_column_noise_std": 0.0,
        "camera_offset_noise_std": 0.0,
        "radar_range_noise_std": 0.0,
        "drop_flag_error_probability": 0.0,
    }
    preset.update(overrides)
    return preset


def create_crazy_chicken_state(
    environment: CrazyChickenPOMDP,
    chickens: Sequence[Sequence[float]],
    ship_column: Optional[int] = None,
    cooldown: int = 0,
    ship_hit: bool = False,
    step: int = 0,
    projectiles: Optional[Sequence[float]] = None,
) -> CrazyChickenState:
    """Build a state vector for ``environment`` from its parts.

    Args:
        environment: The environment whose layout the state must match.
        chickens: One ``(column, row, direction, mode, alive)`` row per slot.
        ship_column: The ship's column. Defaults to the environment's start
            column.
        cooldown: Steps remaining before the ship may fire. Defaults to 0.
        ship_hit: Whether a chicken has reached the ship. Defaults to ``False``.
        step: Step counter. Defaults to 0.
        projectiles: Per-column projectile rows. Defaults to an empty sky.

    Returns:
        A ``float64`` state vector of length ``environment.state_size``.
    """
    return make_state(
        num_chickens=environment.num_chickens,
        num_columns=environment.num_columns,
        ship_column=(environment.ship_start_column if ship_column is None else int(ship_column)),
        chickens=chickens,
        cooldown=cooldown,
        ship_hit=ship_hit,
        step=step,
        projectiles=projectiles,
    )
