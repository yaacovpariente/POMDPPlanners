# SPDX-License-Identifier: MIT

"""Legal fleet layouts for the Battleship POMDP.

A Battleship board is not a free binary occupancy grid: the occupied cells must
decompose into a fleet of straight ships of given lengths that do not overlap
(and, optionally, do not touch). Every belief this environment holds must stay
inside that set, otherwise a planner reasons about boards the world can never
produce.

This module builds the set once, exhaustively, and hands it out as a dense
``(num_layouts, num_cells)`` ``uint8`` table. Downstream that table is what
makes an *exact* belief cheap: conditioning on a deterministic probe outcome is
a single boolean filter over a column, so the belief never depletes and never
drifts off the legal set (see :mod:`battleship_belief`).

Enumeration is over **placement configurations**, not over distinct occupancy
masks. A configuration is one choice of position and orientation per ship, so
the uniform prior implemented here is the usual "drop each ship at random,
reject overlaps" prior. Two different configurations can share an occupancy
mask (a length-3 and a length-2 ship in a row cover the same five cells as a
length-2 followed by a length-3), and both rows are kept: the mask's prior
probability is then correctly proportional to how many ways the fleet can
produce it.

Classes:
    FleetLayoutTable: The enumerated layouts plus the geometry that defines them.
    BattleshipInitialStateDistribution: Uniform distribution over initial states.
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from POMDPPlanners.core.distributions import Distribution


def _ship_placements(board_size: int, length: int) -> List[Tuple[int, ...]]:
    """Enumerate every on-board placement of one straight ship.

    Args:
        board_size: Side length of the square board.
        length: Ship length in cells.

    Returns:
        One tuple of flat cell indices per placement, horizontal placements
        first then vertical, each in row-major scan order. A length-1 ship
        would be enumerated twice under both orientations, so its vertical
        pass is skipped.

    Raises:
        ValueError: If the ship cannot fit on the board.
    """
    if length < 1:
        raise ValueError(f"ship length must be at least 1, got {length}")
    if length > board_size:
        raise ValueError(f"ship of length {length} does not fit on a {board_size}x{board_size} board")

    placements: List[Tuple[int, ...]] = []
    for row in range(board_size):
        for col in range(board_size - length + 1):
            placements.append(tuple(row * board_size + col + offset for offset in range(length)))
    if length == 1:
        # A single cell is the same placement under either orientation;
        # enumerating both would double every length-1 ship's prior weight.
        return placements
    for row in range(board_size - length + 1):
        for col in range(board_size):
            placements.append(
                tuple((row + offset) * board_size + col for offset in range(length))
            )
    return placements


def _neighbourhood_mask(board_size: int, cells: Sequence[int]) -> int:
    """Return the bitmask of ``cells`` plus their 8-connected neighbours."""
    mask = 0
    for cell in cells:
        row, col = divmod(cell, board_size)
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                n_row, n_col = row + d_row, col + d_col
                if 0 <= n_row < board_size and 0 <= n_col < board_size:
                    mask |= 1 << (n_row * board_size + n_col)
    return mask


class FleetLayoutTable:
    """Every legal placement configuration of one fleet on one square board.

    Attributes:
        board_size: Side length of the square board.
        ship_lengths: Ship lengths, as given.
        num_cells: ``board_size ** 2``.
        masks: ``(num_layouts, num_cells)`` ``uint8`` occupancy table, one row
            per placement configuration. Row order is deterministic.
    """

    def __init__(
        self,
        board_size: int,
        ship_lengths: Sequence[int],
        allow_adjacent_ships: bool = True,
        max_layouts: int = 2_000_000,
    ):
        """Enumerate the layouts.

        Args:
            board_size: Side length of the square board.
            ship_lengths: Ship lengths. Order does not matter; ships of equal
                length are deduplicated (see below).
            allow_adjacent_ships: When ``False``, ships may not touch, not even
                diagonally. Defaults to ``True``.
            max_layouts: Refuse to enumerate beyond this many configurations.

        Raises:
            ValueError: If the fleet cannot be placed at all, or if the
                configuration count exceeds ``max_layouts``.
        """
        self.board_size = int(board_size)
        self.ship_lengths: Tuple[int, ...] = tuple(int(length) for length in ship_lengths)
        self.allow_adjacent_ships = bool(allow_adjacent_ships)
        self.num_cells = self.board_size * self.board_size

        rows = self._enumerate(max_layouts)
        if not rows:
            raise ValueError(
                f"no legal layout exists for ships {self.ship_lengths} on a "
                f"{self.board_size}x{self.board_size} board with "
                f"allow_adjacent_ships={self.allow_adjacent_ships}"
            )
        self.masks: np.ndarray = np.asarray(rows, dtype=np.uint8)

    def _enumerate(self, max_layouts: int) -> List[np.ndarray]:
        # Ships are placed longest-first: the long ship has the fewest
        # placements, so the search prunes earliest.
        ordered = sorted(self.ship_lengths, reverse=True)
        per_length_placements = {
            length: _ship_placements(self.board_size, length) for length in set(ordered)
        }
        blocked_masks = {
            length: [
                _neighbourhood_mask(self.board_size, cells) if not self.allow_adjacent_ships
                else sum(1 << cell for cell in cells)
                for cells in per_length_placements[length]
            ]
            for length in set(ordered)
        }

        rows: List[np.ndarray] = []
        n_ships = len(ordered)

        def recurse(index: int, occupied: int, blocked: int, min_choice: int) -> None:
            if index == n_ships:
                row = np.zeros(self.num_cells, dtype=np.uint8)
                cells = occupied
                while cells:
                    lowest = cells & -cells
                    row[lowest.bit_length() - 1] = 1
                    cells ^= lowest
                rows.append(row)
                # Checked after the append, not before it: checking on entry
                # lets the last layout through, because the call that appends it
                # returns without another check. That off-by-one defeats the one
                # thing this guard is for -- refusing a fleet whose table would
                # not fit in memory.
                if len(rows) > max_layouts:
                    raise ValueError(
                        f"fleet {self.ship_lengths} on a {self.board_size}x{self.board_size} "
                        f"board has more than {max_layouts} placement configurations; "
                        "the exact BattleshipBelief cannot enumerate it. Use a smaller "
                        "board or a smaller fleet."
                    )
                return
            length = ordered[index]
            placements = per_length_placements[length]
            blocked_for_length = blocked_masks[length]
            # Ships of equal length are interchangeable, so a configuration is
            # counted once by requiring their placement indices to increase.
            # Without this the prior would over-weight nothing in particular but
            # the table would carry k! copies of every layout.
            same_length_as_previous = index > 0 and ordered[index - 1] == length
            start = min_choice if same_length_as_previous else 0
            for choice in range(start, len(placements)):
                cell_mask = 0
                for cell in placements[choice]:
                    cell_mask |= 1 << cell
                if cell_mask & blocked:
                    continue
                recurse(
                    index + 1,
                    occupied | cell_mask,
                    blocked | blocked_for_length[choice],
                    choice + 1,
                )

        recurse(0, 0, 0, 0)
        return rows

    @property
    def num_layouts(self) -> int:
        """Number of enumerated placement configurations."""
        return int(self.masks.shape[0])

    def consistent_indices(
        self, revealed_cells: np.ndarray, revealed_values: np.ndarray
    ) -> np.ndarray:
        """Indices of the layouts matching every revealed cell.

        Args:
            revealed_cells: Flat indices of the cells whose occupancy is known.
            revealed_values: The known occupancy (0 or 1) of each of those cells.

        Returns:
            A sorted ``int64`` array of row indices into :attr:`masks`.
        """
        if len(revealed_cells) == 0:
            return np.arange(self.num_layouts, dtype=np.int64)
        columns = self.masks[:, np.asarray(revealed_cells, dtype=np.int64)]
        keep = np.all(columns == np.asarray(revealed_values, dtype=np.uint8), axis=1)
        return np.flatnonzero(keep).astype(np.int64)


class BattleshipInitialStateDistribution(Distribution):
    """Uniform distribution over the fleet's legal starting boards.

    Every initial state has an all-zero probe half, so the distribution is
    exactly the uniform prior over :attr:`FleetLayoutTable.masks` rows lifted
    into the environment's state layout. Sampling reads rows out of the table
    rather than materialising every state, because the table has tens of
    thousands of rows and ``initial_state_dist()`` is called per episode and
    per belief construction.
    """

    def __init__(self, layouts: FleetLayoutTable):
        """Initialize the distribution.

        Args:
            layouts: The enumerated legal layouts.
        """
        self.layouts = layouts

    def sample(self, n_samples: int = 1) -> List[Any]:
        """Draw ``n_samples`` initial states.

        Args:
            n_samples: How many states to draw. Defaults to 1.

        Returns:
            A list of ``float64`` state arrays of length ``2 * num_cells``.
        """
        indices = np.random.randint(0, self.layouts.num_layouts, size=int(n_samples))
        states = np.zeros((int(n_samples), 2 * self.layouts.num_cells), dtype=np.float64)
        states[:, : self.layouts.num_cells] = self.layouts.masks[indices]
        return [states[i] for i in range(int(n_samples))]

    def probability(self, values: List[Any]) -> np.ndarray:
        """Probability of each candidate initial state.

        Args:
            values: Candidate states.

        Returns:
            ``float64`` array of probabilities. A state whose probe half is not
            all-zero, or whose occupancy half is not a legal layout, has
            probability zero.
        """
        num_cells = self.layouts.num_cells
        probs = np.zeros(len(values), dtype=np.float64)
        for i, value in enumerate(values):
            state = np.asarray(value, dtype=np.float64)
            if state.shape[0] != 2 * num_cells or np.any(state[num_cells:] > 0.5):
                continue
            occupancy = (state[:num_cells] > 0.5).astype(np.uint8)
            matches = int(np.count_nonzero(np.all(self.layouts.masks == occupancy, axis=1)))
            probs[i] = matches / self.layouts.num_layouts
        return probs


_LAYOUT_CACHE: Dict[Tuple[int, Tuple[int, ...], bool], "FleetLayoutTable"] = {}


def get_layout_table(
    board_size: int,
    ship_lengths: Sequence[int],
    allow_adjacent_ships: bool = True,
    max_layouts: int = 2_000_000,
) -> FleetLayoutTable:
    """Return the layout table for one board geometry, building it at most once.

    The table is a pure function of the geometry and is read-only, so a single
    instance is shared process-wide. Enumeration takes long enough that
    rebuilding it per environment instance — and environments are rebuilt per
    worker process and per tree node's model reference — would dominate.

    Args:
        board_size: Side length of the square board.
        ship_lengths: Ship lengths.
        allow_adjacent_ships: Whether ships may touch.
        max_layouts: Enumeration cap.

    Returns:
        The shared :class:`FleetLayoutTable`.
    """
    key = (int(board_size), tuple(int(x) for x in ship_lengths), bool(allow_adjacent_ships))
    table = _LAYOUT_CACHE.get(key)
    if table is None:
        return _LAYOUT_CACHE.setdefault(
            key,
            FleetLayoutTable(
                board_size=board_size,
                ship_lengths=ship_lengths,
                allow_adjacent_ships=allow_adjacent_ships,
                max_layouts=max_layouts,
            ),
        )
    # The cache is keyed on geometry alone, because the table it holds is a
    # function of geometry alone. The cap is not: it is the caller's limit, and
    # re-checking it here is what stops the same environment from constructing
    # in a warm process and raising in a cold one. That difference would show up
    # only inside a worker after unpickling, or after a from_dict in a fresh
    # process, which is the hardest place to read a stack trace.
    if table.num_layouts > int(max_layouts):
        raise ValueError(
            f"fleet {table.ship_lengths} on a {table.board_size}x{table.board_size} "
            f"board has more than {max_layouts} placement configurations; "
            "the exact BattleshipBelief cannot enumerate it. Use a smaller "
            "board or a smaller fleet."
        )
    return table
