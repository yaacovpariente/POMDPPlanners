# SPDX-License-Identifier: MIT

"""State and observation layout for the Chicheck Invaders POMDP, and its state prior.

Both a state and an observation are fixed-length ``float64`` vectors. Fixed
length is what lets a particle filter hold them in one array and what lets two
observations be compared field by field; it is also why a dead chicken keeps its
slot rather than being removed, and why an unreported sensor writes zeros into
its slot instead of shortening the vector.

State layout, for ``N`` chicken slots::

    [ step | ship column | fire cooldown | ship hit
      | chicken 0: x, y, direction, mode, alive
      | ...
      | chicken N-1: x, y, direction, mode, alive ]

``mode`` is 0 for patrol and 1 for dive; ``alive`` is 0 or 1. ``ship hit`` is
carried in the state rather than recomputed because :meth:`is_terminal` is given
a state alone, with no transition to re-derive the collision from.

Nothing here records a shot. The ship's gun is hitscan: a shot resolves inside
the step that fired it, so there is never anything in flight for a state to
carry, and the only thing the gun leaves behind is the cooldown.

Observation layout, for ``N`` chicken slots::

    [ own-column reading
      | chicken 0: camera reported, camera offset, radar reported, radar rows, radar drop
      | ... ]

Every masked field is written as ``0.0``, so two observations that report the
same things compare equal field by field and hash alike. The ``reported`` flags
are what disambiguate a masked zero from a genuine reading of zero.

Classes:
    ChicheckInvadersInitialStateDistribution: The per-episode chicken placement prior.

Functions:
    chicken_block: Slice bounds of one chicken's fields inside a state.
    chicken_slots: The ``(N, 5)`` view of every chicken slot in a state.
    make_state: Build one state vector from its parts.
    state_size: Length of a state vector.
    observation_size: Length of an observation vector.
"""

from typing import Any, List, Sequence, Tuple

import numpy as np

from POMDPPlanners.core.distributions import Distribution


#: Index of the step counter inside a state vector.
STEP_INDEX = 0
#: Index of the ship's column inside a state vector.
SHIP_COLUMN_INDEX = 1
#: Index of the fire cooldown inside a state vector.
COOLDOWN_INDEX = 2
#: Index of the "a chicken reached the ship" flag inside a state vector.
SHIP_HIT_INDEX = 3
#: Number of scalar ship fields preceding the chicken slots.
SHIP_WIDTH = 4

#: Number of fields one chicken slot occupies.
CHICKEN_WIDTH = 5
#: Offsets of the five fields inside one chicken slot.
CHICKEN_COLUMN = 0
CHICKEN_ROW = 1
CHICKEN_DIRECTION = 2
CHICKEN_MODE = 3
CHICKEN_ALIVE = 4

#: ``mode`` values. Patrol chickens hold their row; dive chickens drop one row
#: per step. The environment's vertical speed ``v`` is ``-1`` for a dive and
#: ``0`` for a patrol, which is what the radar's drop flag reports.
MODE_PATROL = 0.0
MODE_DIVE = 1.0

#: Index of the own-column reading inside an observation vector.
OBSERVED_SHIP_COLUMN_INDEX = 0
#: Number of scalar ship fields preceding the per-chicken observation slots.
OBSERVATION_SHIP_WIDTH = 1
#: Number of fields one chicken's observation slot occupies.
OBSERVATION_CHICKEN_WIDTH = 5
#: Offsets of the five fields inside one chicken's observation slot.
OBSERVED_CAMERA_REPORTED = 0
OBSERVED_CAMERA_OFFSET = 1
OBSERVED_RADAR_REPORTED = 2
OBSERVED_RADAR_ROWS = 3
OBSERVED_RADAR_DROP = 4


def state_size(num_chickens: int) -> int:
    """Length of a state vector for this flock size.

    Args:
        num_chickens: Number of chicken slots.

    Returns:
        The vector length.
    """
    return SHIP_WIDTH + CHICKEN_WIDTH * int(num_chickens)


def observation_size(num_chickens: int) -> int:
    """Length of a partially observable observation vector.

    Args:
        num_chickens: Number of chicken slots.

    Returns:
        The vector length.
    """
    return OBSERVATION_SHIP_WIDTH + OBSERVATION_CHICKEN_WIDTH * int(num_chickens)


def chicken_block(index: int) -> Tuple[int, int]:
    """Slice bounds of chicken ``index``'s fields inside a state vector.

    Args:
        index: Zero-based chicken slot.

    Returns:
        ``(start, stop)`` indices.
    """
    start = SHIP_WIDTH + CHICKEN_WIDTH * int(index)
    return start, start + CHICKEN_WIDTH


def chicken_slots(state: np.ndarray, num_chickens: int) -> np.ndarray:
    """Return the ``(num_chickens, 5)`` chicken block of ``state`` as a view.

    A view rather than a copy: the transition writes through it, and the cost of
    copying the block once per chicken per rollout step is the whole difference
    between a usable and an unusable scalar model here.

    Args:
        state: A state vector.
        num_chickens: Number of chicken slots.

    Returns:
        A ``(num_chickens, 5)`` ``float64`` view.
    """
    stop = SHIP_WIDTH + CHICKEN_WIDTH * int(num_chickens)
    return state[SHIP_WIDTH:stop].reshape(int(num_chickens), CHICKEN_WIDTH)


def make_state(
    num_chickens: int,
    ship_column: int,
    chickens: Sequence[Sequence[float]],
    cooldown: int = 0,
    ship_hit: bool = False,
    step: int = 0,
) -> np.ndarray:
    """Build one state vector from its parts.

    Exists so tests and pinned scenarios can write a specific world down without
    reproducing the layout arithmetic, which is the kind of duplication that
    goes stale silently when the layout changes.

    Args:
        num_chickens: Number of chicken slots.
        ship_column: The ship's column.
        chickens: One ``(x, y, direction, mode, alive)`` row per chicken slot.
        cooldown: Steps remaining before the ship may fire. Defaults to 0.
        ship_hit: Whether a chicken has reached the ship. Defaults to ``False``.
        step: Step counter. Defaults to 0.

    Returns:
        A ``float64`` state vector.

    Raises:
        ValueError: If ``chickens`` has the wrong shape.
    """
    rows = np.asarray(chickens, dtype=np.float64).reshape(-1, CHICKEN_WIDTH)
    if len(rows) != int(num_chickens):
        raise ValueError(f"expected {num_chickens} chicken rows, got {len(rows)}")
    state = np.zeros(state_size(num_chickens), dtype=np.float64)
    state[STEP_INDEX] = float(step)
    state[SHIP_COLUMN_INDEX] = float(ship_column)
    state[COOLDOWN_INDEX] = float(cooldown)
    state[SHIP_HIT_INDEX] = float(bool(ship_hit))
    chicken_slots(state, num_chickens)[:] = rows
    return state


class ChicheckInvadersInitialStateDistribution(Distribution):
    """Where the chickens are when an episode starts, and nothing else.

    The ship's column, its cooldown and the empty sky are identical in every
    draw, because the ship knows where it is and knows it has not fired. All the
    randomness is in the flock: which cells the chickens hold, which way each is
    walking, and which of them are already diving. That is exactly the hidden
    state, which is why it is drawn per episode rather than fixed in the
    environment's configuration -- with one flock baked into the config every
    belief particle would carry the same world and the task would collapse into
    an MDP.

    Chickens are placed on distinct cells. Two on one cell is not forbidden
    later -- sideways patrols can walk into each other -- but starting that way
    would waste a slot, since a hitscan shot only ever removes the lowest
    chicken in its column.

    Attributes:
        num_chickens: Number of chicken slots.
        num_columns: Grid width.
        num_rows: Grid height.
        ship_column: The ship's starting column.
        lowest_start_row: Lowest row a chicken may start on.
        initial_dive_probability: Chance a chicken starts already diving.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        num_chickens: int,
        num_columns: int,
        num_rows: int,
        ship_column: int,
        lowest_start_row: int,
        initial_dive_probability: float,
    ):
        """Initialize the flock prior.

        Args:
            num_chickens: Number of chicken slots.
            num_columns: Grid width.
            num_rows: Grid height.
            ship_column: The ship's starting column.
            lowest_start_row: Lowest row a chicken may start on. One rather than
                zero, because row zero is the ship's own row: a chicken placed
                there would either end the episode before the first action or
                have to be teleported away, and neither is a world anyone is
                planning in.
            initial_dive_probability: Chance a chicken starts already diving
                rather than patrolling.
        """
        self.num_chickens = int(num_chickens)
        self.num_columns = int(num_columns)
        self.num_rows = int(num_rows)
        self.ship_column = int(ship_column)
        self.lowest_start_row = int(lowest_start_row)
        self.initial_dive_probability = float(initial_dive_probability)

    def sample(self, n_samples: int = 1) -> List[Any]:
        """Draw ``n_samples`` independent initial states.

        Draws from the global ``np.random`` stream, which is the convention
        every environment here follows and what makes ``np.random.seed``
        reproduce a run.

        Args:
            n_samples: How many states to draw. Defaults to 1.

        Returns:
            A list of ``float64`` state vectors, each its own buffer so a caller
            mutating one particle cannot corrupt another.
        """
        rows = np.arange(self.lowest_start_row, self.num_rows)
        cells = np.array(
            [(column, row) for row in rows for column in range(self.num_columns)],
            dtype=np.float64,
        )
        states: List[Any] = []
        for _ in range(int(n_samples)):
            chosen = np.random.choice(len(cells), size=self.num_chickens, replace=False)
            chickens = np.zeros((self.num_chickens, CHICKEN_WIDTH), dtype=np.float64)
            chickens[:, CHICKEN_COLUMN] = cells[chosen, 0]
            chickens[:, CHICKEN_ROW] = cells[chosen, 1]
            chickens[:, CHICKEN_DIRECTION] = np.where(
                np.random.random(self.num_chickens) < 0.5, -1.0, 1.0
            )
            chickens[:, CHICKEN_MODE] = np.where(
                np.random.random(self.num_chickens) < self.initial_dive_probability,
                MODE_DIVE,
                MODE_PATROL,
            )
            chickens[:, CHICKEN_ALIVE] = 1.0
            states.append(
                make_state(
                    num_chickens=self.num_chickens,
                    ship_column=self.ship_column,
                    chickens=chickens,
                )
            )
        return states

    def probability(self, values: List[Any]) -> np.ndarray:
        """Probability of each candidate initial state under this prior.

        Args:
            values: Candidate initial states.

        Returns:
            One probability per candidate: zero for anything this prior cannot
            produce, and otherwise the flat placement mass times the per-chicken
            direction and mode mass.
        """
        cell_count = (self.num_rows - self.lowest_start_row) * self.num_columns
        # Placements are drawn without replacement over an unordered set of
        # cells, but the slots are distinguishable, so one *assignment* of
        # chickens to cells has mass 1 / (cell_count)_{num_chickens}.
        arrangements = 1.0
        for offset in range(self.num_chickens):
            arrangements *= float(cell_count - offset)
        probabilities = np.zeros(len(values), dtype=np.float64)
        for index, candidate in enumerate(values):
            state = np.asarray(candidate, dtype=np.float64)
            if len(state) != state_size(self.num_chickens):
                continue
            if not self._is_supported(state):
                continue
            slots = chicken_slots(state, self.num_chickens)
            mass = 1.0 / arrangements
            for slot in slots:
                mass *= 0.5
                mass *= (
                    self.initial_dive_probability
                    if slot[CHICKEN_MODE] == MODE_DIVE
                    else 1.0 - self.initial_dive_probability
                )
            probabilities[index] = mass
        return probabilities

    # pylint: disable-next=too-many-return-statements
    def _is_supported(self, state: np.ndarray) -> bool:
        """Whether ``state`` is one this prior can produce at all.

        One early return per condition, rather than one long conjunction: each
        line names a separate thing the prior fixes, and a reader checking
        whether a condition is missing can scan them.
        """
        if state[STEP_INDEX] != 0.0 or state[COOLDOWN_INDEX] != 0.0:
            return False
        if state[SHIP_HIT_INDEX] != 0.0 or state[SHIP_COLUMN_INDEX] != float(self.ship_column):
            return False
        slots = chicken_slots(state, self.num_chickens)
        if not np.all(slots[:, CHICKEN_ALIVE] == 1.0):
            return False
        # The two categorical fields are checked as well as the positions. A
        # candidate carrying direction 0 or mode 7 is not a state this prior can
        # draw, and handing it positive mass would let a malformed particle be
        # weighted as though it were an ordinary one.
        if not np.all(np.isin(slots[:, CHICKEN_DIRECTION], (-1.0, 1.0))):
            return False
        if not np.all(np.isin(slots[:, CHICKEN_MODE], (MODE_PATROL, MODE_DIVE))):
            return False
        columns = slots[:, CHICKEN_COLUMN]
        rows = slots[:, CHICKEN_ROW]
        if not np.all((columns >= 0) & (columns < self.num_columns)):
            return False
        if not np.all((rows >= self.lowest_start_row) & (rows < self.num_rows)):
            return False
        cells = {(float(c), float(r)) for c, r in zip(columns, rows)}
        return len(cells) == self.num_chickens
