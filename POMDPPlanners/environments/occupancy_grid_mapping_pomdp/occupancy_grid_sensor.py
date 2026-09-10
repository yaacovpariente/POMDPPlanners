# SPDX-License-Identifier: MIT

"""Ray casting and the inverse sensor model for occupancy-grid mapping.

This module holds the two halves of the sensor that
:class:`~POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_pomdp.OccupancyGridMappingPOMDP`
is built on, kept apart from the environment so each can be tested on its own:

* **Forward model** -- :func:`cast_scan` casts ``num_beams`` rays from a grid
  cell over a field of view and reports one range per beam.
* **Inverse sensor model** -- :func:`scan_log_odds_delta` turns that scan into
  the per-cell log-odds increments an occupancy grid accumulates, following the
  standard formulation (Moravec and Elfes 1985; Thrun, Burgard and Fox,
  *Probabilistic Robotics*, ch. 9): cells the beam passes through get free-space
  evidence, the cell it stops in gets occupied evidence, and cells behind that
  one are left untouched because the beam never saw them.

**Both are deterministic.** Range noise is added by the environment's
observation sampler, not here, and deliberately never reaches the map update:
the map a robot builds is the map its *nominal* scan implies. Keeping the two
apart is what lets the transition kernel stay a function of ``(state, action)``
while the observation likelihood stays a plain Gaussian.

Geometry conventions, fixed here and relied on by every caller:

* ``row`` increases downwards, ``col`` increases to the right.
* A heading is one of four indices -- ``0`` north, ``1`` east, ``2`` south,
  ``3`` west -- and a bearing is measured in degrees clockwise from north, so
  the unit direction of bearing ``theta`` is ``(-cos(theta), +sin(theta))`` in
  ``(row, col)``.
* A beam that stops in a cell reports the Euclidean distance between the two
  cell *centres*, ``hypot(dr, dc)``. Range is therefore exact and symmetric
  rather than quantised by the stepping used to enumerate the ray.

Functions:
    build_ray_templates: Precompute the cells each beam crosses, per heading.
    cast_scan: Cast one scan against a true occupancy grid.
    scan_log_odds_delta: Inverse sensor model for one scan.
    log_odds_from_probability: Convert a probability to log-odds.
    grid_entropy_bits: Binary entropy of an occupancy grid, in bits.
"""

import math
from typing import List, Tuple

import numpy as np


#: Number of headings. North, east, south, west, in that order.
NUM_HEADINGS = 4

#: ``(row, col)`` step for each heading index.
HEADING_STEPS: Tuple[Tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))

#: Short labels for the heading indices, used by the visualizer and tests.
HEADING_LABELS: Tuple[str, ...] = ("N", "E", "S", "W")

#: Fraction of a cell each step advances while enumerating a ray. Small enough
#: that no cell the ray crosses is skipped for the ranges this environment uses,
#: and only paid once, at construction.
_RAY_STEP = 0.05


def log_odds_from_probability(probability: float) -> float:
    """Return ``log(p / (1 - p))``.

    Args:
        probability: A probability strictly inside ``(0, 1)``.

    Returns:
        The log-odds of ``probability``.

    Raises:
        ValueError: If ``probability`` is not strictly inside ``(0, 1)``. A
            probability of exactly 0 or 1 is infinite evidence, which no finite
            number of further readings could ever revise.
    """
    if not 0.0 < probability < 1.0:
        raise ValueError(
            f"probability must be strictly inside (0, 1), got {probability}"
        )
    return float(math.log(probability / (1.0 - probability)))


def grid_entropy_bits(log_odds: np.ndarray) -> float:
    """Total binary entropy of an occupancy grid held in log-odds, in bits.

    The grid's cells are treated as independent Bernoulli variables, which is
    the assumption occupancy grid mapping is defined under, so the grid's
    entropy is the sum of its cells'. Bits rather than nats because one unknown
    cell is then worth exactly ``1.0`` and a whole unknown grid exactly its cell
    count, which makes both the reward scale and the termination threshold
    readable without conversion.

    Computed from the log-odds directly via ``p = sigmoid(l)`` with the
    numerically stable branch of the sigmoid, because that is the form the state
    carries and round-tripping through probabilities is where repeated Bayesian
    updates underflow.

    Args:
        log_odds: Occupancy log-odds, any shape.

    Returns:
        Summed binary entropy in bits, in ``[0, log_odds.size]``.
    """
    values = np.asarray(log_odds, dtype=np.float64)
    # Stable sigmoid: exp() is only ever evaluated on a non-positive argument.
    positive = values >= 0.0
    exp_negative_abs = np.exp(-np.abs(values))
    probability = np.where(
        positive, 1.0 / (1.0 + exp_negative_abs), exp_negative_abs / (1.0 + exp_negative_abs)
    )
    # A cell clamped hard enough that p rounds to 0 or 1 contributes no entropy;
    # clipping keeps log2 off its singularity instead of relying on nan_to_num.
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    per_cell = -(clipped * np.log2(clipped) + (1.0 - clipped) * np.log2(1.0 - clipped))
    return float(np.sum(per_cell))


def build_ray_templates(
    num_beams: int, field_of_view_degrees: float, max_range_cells: float
) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """Precompute, per heading, the cells every beam crosses and at what range.

    A beam's geometry depends only on its bearing and the sensor range, never on
    where the robot is or what the map holds, so it is enumerated once at
    construction and reused for every scan. Without this the per-scan cost is a
    Python loop over sub-cell steps, which a tree-search planner pays tens of
    thousands of times per decision.

    Args:
        num_beams: Number of beams, evenly spaced across the field of view.
        field_of_view_degrees: Angular width of the fan, centred on the heading.
            ``360`` gives a full surround scan.
        max_range_cells: Maximum sensor range, in cell widths.

    Returns:
        One entry per heading, in heading-index order. Each entry is
        ``(offsets, ranges)`` where ``offsets`` is an ``(num_beams, length)``
        ``int64`` array of ``(row, col)`` offsets flattened into two planes --
        specifically ``offsets[..., 0]`` rows and ``offsets[..., 1]`` columns,
        so its true shape is ``(num_beams, length, 2)`` -- ordered from nearest
        to furthest, and ``ranges`` is the matching
        ``(num_beams, length)`` ``float64`` array of centre-to-centre distances.
        Beams shorter than ``length`` are padded by repeating their last cell,
        with the padded range set to ``inf`` so a padded slot can never be
        mistaken for a real return.

    Raises:
        ValueError: If any argument is outside its valid range.
    """
    if num_beams < 1:
        raise ValueError(f"num_beams must be at least 1, got {num_beams}")
    if not 0.0 < field_of_view_degrees <= 360.0:
        raise ValueError(
            f"field_of_view_degrees must be in (0, 360], got {field_of_view_degrees}"
        )
    if max_range_cells <= 0.0:
        raise ValueError(f"max_range_cells must be positive, got {max_range_cells}")

    # Beams sit at bin centres rather than endpoints. For a 360-degree fan that
    # is what stops the first and last beam from landing on the same bearing;
    # for a narrower fan it keeps the fan symmetric about the heading.
    span = float(field_of_view_degrees)
    offsets_degrees = [
        -span / 2.0 + span * (index + 0.5) / float(num_beams) for index in range(num_beams)
    ]

    templates: List[Tuple[np.ndarray, np.ndarray]] = []
    for heading in range(NUM_HEADINGS):
        heading_bearing = 90.0 * heading
        beams: List[List[Tuple[int, int, float]]] = []
        for offset_degrees in offsets_degrees:
            bearing = math.radians(heading_bearing + offset_degrees)
            direction_row = -math.cos(bearing)
            direction_col = math.sin(bearing)
            seen = set()
            cells: List[Tuple[int, int, float]] = []
            steps = int(math.ceil(max_range_cells / _RAY_STEP))
            for step in range(1, steps + 1):
                distance = step * _RAY_STEP
                cell = (
                    int(round(distance * direction_row)),
                    int(round(distance * direction_col)),
                )
                if cell == (0, 0) or cell in seen:
                    continue
                centre_range = math.hypot(cell[0], cell[1])
                if centre_range > max_range_cells:
                    continue
                seen.add(cell)
                cells.append((cell[0], cell[1], centre_range))
            beams.append(cells)

        length = max((len(cells) for cells in beams), default=0)
        # A zero-length template would make every beam a max-range miss, which
        # is a silently useless sensor. One slot keeps the array shapes valid.
        length = max(length, 1)
        offsets = np.zeros((num_beams, length, 2), dtype=np.int64)
        ranges = np.full((num_beams, length), np.inf, dtype=np.float64)
        for beam_index, cells in enumerate(beams):
            for slot, (delta_row, delta_col, centre_range) in enumerate(cells):
                offsets[beam_index, slot] = (delta_row, delta_col)
                ranges[beam_index, slot] = centre_range
            # Pad by repeating the last real cell. The padded range stays inf,
            # and cast_scan never reads a padded slot because the ray always
            # terminates at or before it.
            if cells:
                offsets[beam_index, len(cells) :] = offsets[beam_index, len(cells) - 1]
        templates.append((offsets, ranges))

    return tuple(templates)


def cast_scan(
    occupancy: np.ndarray,
    row: int,
    col: int,
    template: Tuple[np.ndarray, np.ndarray],
    max_range_cells: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast one scan from ``(row, col)`` against a true occupancy grid.

    Every beam walks its precomputed cells outwards and stops at the first one
    that is occupied, or at the first one outside the grid. A beam that reaches
    the end of its template without stopping is a max-range miss.

    Args:
        occupancy: ``(num_rows, num_cols)`` array, non-zero where occupied.
        row: Robot row.
        col: Robot column.
        template: The ``(offsets, ranges)`` pair for the robot's heading, as
            returned by :func:`build_ray_templates`.
        max_range_cells: Maximum sensor range, reported by a beam that misses.

    Returns:
        Tuple of:
            - ``ranges``: ``(num_beams,)`` ``float64`` nominal range per beam;
              ``max_range_cells`` for a beam with no return.
            - ``hit``: ``(num_beams,)`` ``bool``, ``True`` where the beam
              stopped on an occupied cell.
            - ``stop_slot``: ``(num_beams,)`` ``int64`` index into the template
              of the cell the beam stopped in, or the template length when it
              ran to the end. Cells at strictly smaller slots were traversed and
              are free-space evidence.
    """
    occupancy_grid = np.asarray(occupancy)
    num_rows, num_cols = occupancy_grid.shape
    offsets, template_ranges = template
    num_beams, length, _ = offsets.shape

    rows = offsets[:, :, 0] + int(row)
    cols = offsets[:, :, 1] + int(col)
    inside = (rows >= 0) & (rows < num_rows) & (cols >= 0) & (cols < num_cols)
    # Index with a safe row so the gather never goes out of bounds; the result
    # is masked by ``inside`` immediately afterwards.
    safe_rows = np.where(inside, rows, 0)
    safe_cols = np.where(inside, cols, 0)
    occupied = (occupancy_grid[safe_rows, safe_cols] != 0) & inside

    # A beam stops on the first occupied cell or the first cell off the grid,
    # whichever comes first. Leaving the grid is a miss, not a return: nothing
    # was detected there, the beam simply left the mapped world.
    stops = occupied | (~inside)
    any_stop = stops.any(axis=1)
    stop_slot = np.where(any_stop, np.argmax(stops, axis=1), length)

    beam_index = np.arange(num_beams)
    hit = np.zeros(num_beams, dtype=bool)
    hit[any_stop] = occupied[beam_index[any_stop], stop_slot[any_stop]]

    ranges = np.full(num_beams, float(max_range_cells), dtype=np.float64)
    ranges[hit] = template_ranges[beam_index[hit], stop_slot[hit]]
    return ranges, hit, stop_slot.astype(np.int64)


def scan_log_odds_delta(
    hit: np.ndarray,
    stop_slot: np.ndarray,
    template: Tuple[np.ndarray, np.ndarray],
    row: int,
    col: int,
    num_rows: int,
    num_cols: int,
    free_log_odds: float,
    occupied_log_odds: float,
) -> np.ndarray:
    """Inverse sensor model: per-cell log-odds increments implied by one scan.

    Per beam, following the standard occupancy-grid inverse sensor model:

    * cells strictly between the robot and where the beam stopped are free-space
      evidence and receive ``free_log_odds`` (negative);
    * the cell a beam stopped *on* is occupied evidence and receives
      ``occupied_log_odds`` (positive);
    * cells beyond that one are occluded -- the beam carries no information
      about them -- and are left alone;
    * a beam with no return marks its whole traversed length free and updates no
      cell as occupied. This case is what makes exploration work at all: without
      it, looking into open space would be indistinguishable from not looking,
      and the information-gain reward would never pay for moving outwards.

    Beams are treated as independent, so a cell crossed by several beams in the
    same scan accumulates one increment per beam. That is the usual convention
    and it is why the environment clamps the accumulated log-odds.

    Args:
        hit: ``(num_beams,)`` bool from :func:`cast_scan`.
        stop_slot: ``(num_beams,)`` int from :func:`cast_scan`.
        template: The heading's ``(offsets, ranges)`` pair.
        row: Robot row.
        col: Robot column.
        num_rows: Grid rows.
        num_cols: Grid columns.
        free_log_odds: Increment applied to a cell observed as free. Negative.
        occupied_log_odds: Increment applied to a cell observed as occupied.
            Positive.

    Returns:
        ``(num_rows * num_cols,)`` ``float64`` array of log-odds increments in
        row-major order.
    """
    offsets, _ = template
    num_beams, length, _ = offsets.shape

    rows = offsets[:, :, 0] + int(row)
    cols = offsets[:, :, 1] + int(col)
    flat = rows * int(num_cols) + cols

    # Everything before the stop slot was traversed, so it is in the grid by
    # construction: the first cell off the grid is itself a stop.
    slots = np.arange(length)[None, :]
    free_mask = slots < stop_slot[:, None]
    free_flat = flat[free_mask]

    num_cells = int(num_rows) * int(num_cols)
    delta = np.bincount(free_flat, minlength=num_cells).astype(np.float64) * float(free_log_odds)

    if np.any(hit):
        beam_index = np.arange(num_beams)
        hit_flat = flat[beam_index[hit], stop_slot[hit]]
        delta += np.bincount(hit_flat, minlength=num_cells).astype(np.float64) * float(
            occupied_log_odds
        )
    return delta
