# SPDX-License-Identifier: MIT

"""Ray casting and the inverse sensor model for occupancy-grid mapping.

This module holds the two halves of the sensor that
``OccupancyGridMappingPOMDP`` is built on, kept apart from the environment so
each can be tested on its own:

* **Forward model** -- :func:`cast_scan` casts ``num_beams`` rays from a grid
  cell over a field of view and reports one range per beam.
* **Inverse sensor model** -- :func:`scan_log_odds_delta` turns that scan into
  the per-cell log-odds increments an occupancy grid accumulates, following the
  standard formulation (Moravec and Elfes 1985; Thrun, Burgard and Fox,
  *Probabilistic Robotics*, ch. 9): cells the beam passes through get free-space
  evidence, the cell it stops in gets occupied evidence, and cells behind that
  one are left untouched because the beam never saw them.

The forward ray cast is noise-free. The transition adds per-beam range noise
before ``observed_scan_log_odds_delta`` interprets the measured ranges. Two
noise laws live here, selected by the environment's ``range_noise_model``: the
original unbounded Gaussian, and a normal truncated to ``[0, +inf)`` for callers
who need a range law that cannot produce a negative reading. The legacy
``scan_log_odds_delta`` helper operates on explicit hit/slot arrays and is used
only by geometry tests; the environment never uses hidden hits for mapping.

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
    observed_scan_evidence_counts: Classify one measured scan into per-cell
        free and occupied sighting counts.
    batch_observed_scan_evidence_counts: The same for one scan per particle.
    observed_scan_log_odds_delta: Log-odds increments of one measured scan.
    log_odds_from_probability: Convert a probability to log-odds.
    grid_entropy_bits: Binary entropy of an occupancy grid, in bits.
    gaussian_log_density: Per-beam log density of the unbounded range law.
    truncated_normal_log_norm: Per-beam log normaliser of the truncated law.
    truncated_normal_log_density: Per-beam log density of the truncated law.
    truncated_normal_from_upper_tail: Quantile function of the truncated law.
    sample_truncated_normal_ranges: Draw truncated-normal ranges.
    resolve_range_noise_model: Coerce a string or member to a range noise model.
    range_log_density: Per-beam log density under the selected range law.
    scan_log_density: Whole-scan log density under the selected range law.
    sample_ranges: Draw noisy ranges under the selected range law.
    quadrature_ranges: Transform fixed quadrature points under the selected law.

Classes:
    RangeNoiseModel: Which per-beam range noise law the sensor uses.
"""

import math
from enum import Enum
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

# The normal CDF, its inverse and its log are compiled ufuncs that pylint
# cannot see inside scipy.special, as with the native extensions elsewhere.
from scipy.special import log_ndtr, ndtr, ndtri  # pylint: disable=no-name-in-module


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
        raise ValueError(f"probability must be strictly inside (0, 1), got {probability}")
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
        raise ValueError(f"field_of_view_degrees must be in (0, 360], got {field_of_view_degrees}")
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


def observed_scan_evidence_counts(
    observed_ranges,
    template,
    row,
    col,
    num_rows,
    num_cols,
    max_range_cells,
):
    """Classify measured ranges into per-cell free and occupied sighting counts.

    This is the geometric half of the inverse sensor model, split out from the
    arithmetic half so that a rule written on probabilities and the built-in
    log-odds rule classify a scan in exactly one place. It consults neither the
    hidden occupancy nor a hidden hit flag.

    A reading below maximum selects the nearest valid ray-cell centre; ties
    select the nearer cell. A reading at/above maximum frees the full ray.
    Out-of-grid cells and padded template slots never receive evidence.
    Gaussian tails remain in the observation; only this inverse interpretation
    selects grid cells. Negative readings select the first valid cell.

    The robot's own cell is not counted here. It takes one free sighting of its
    own, which every caller adds, because it is evidence from the robot being
    there rather than from any beam.

    Args:
        observed_ranges: ``(num_beams,)`` measured ranges.
        template: The heading's ``(offsets, distances)`` pair.
        row: Robot row.
        col: Robot column.
        num_rows: Grid rows.
        num_cols: Grid columns.
        max_range_cells: Sensor range in cell widths.

    Returns:
        A ``(free_counts, occupied_counts)`` pair of ``(num_rows * num_cols,)``
        integer arrays in row-major order, each counting one entry per beam.
    """
    offsets, distances = template
    rows = offsets[:, :, 0] + row
    cols = offsets[:, :, 1] + col
    valid = (
        (rows >= 0) & (rows < num_rows) & (cols >= 0) & (cols < num_cols) & np.isfinite(distances)
    )
    # Once a ray leaves the grid it cannot return (straight rays, convex grid).
    valid = np.logical_and.accumulate(valid, axis=1)
    has_cell = valid.any(axis=1)
    nearest = np.argmin(
        np.where(valid, np.abs(distances - observed_ranges[:, None]), np.inf), axis=1
    )
    hit = (observed_ranges < max_range_cells) & has_cell
    slots = np.arange(distances.shape[1])[None, :]
    free = valid & ((slots < nearest[:, None]) | ~hit[:, None])
    occupied = valid & (slots == nearest[:, None]) & hit[:, None]
    flat = rows * num_cols + cols
    return (
        np.bincount(flat[free], minlength=num_rows * num_cols),
        np.bincount(flat[occupied], minlength=num_rows * num_cols),
    )


def batch_observed_scan_evidence_counts(
    observed_ranges,
    ray_templates,
    rows,
    cols,
    headings,
    num_rows,
    num_cols,
    max_range_cells,
):
    """Batched :func:`observed_scan_evidence_counts`: one scan per particle.

    Particles are grouped by heading so that one array pass covers every
    particle that shares a ray template. Rows of the two count arrays are
    disjoint between groups, so the per-group accumulation below gives each
    particle exactly the counts its own scan implies.

    Args:
        observed_ranges: ``(N, num_beams)`` measured ranges, one row per particle.
        ray_templates: The per-heading ``(offsets, distances)`` pairs.
        rows: ``(N,)`` robot rows.
        cols: ``(N,)`` robot columns.
        headings: ``(N,)`` heading indices.
        num_rows: Grid rows.
        num_cols: Grid columns.
        max_range_cells: Sensor range in cell widths.

    Returns:
        A ``(free_counts, occupied_counts)`` pair of ``(N, num_rows * num_cols)``
        integer arrays.
    """
    count = observed_ranges.shape[0]
    num_cells = int(num_rows) * int(num_cols)
    free_counts = np.zeros((count, num_cells), dtype=np.int64)
    occupied_counts = np.zeros((count, num_cells), dtype=np.int64)
    slot_offsets = np.arange(count) * num_cells
    for heading in np.unique(headings):
        index = np.flatnonzero(headings == heading)
        offsets, distances = ray_templates[int(heading)]
        length = distances.shape[1]
        ray_rows = offsets[None, :, :, 0] + rows[index, None, None]
        ray_cols = offsets[None, :, :, 1] + cols[index, None, None]
        valid = (
            (ray_rows >= 0)
            & (ray_rows < num_rows)
            & (ray_cols >= 0)
            & (ray_cols < num_cols)
            & np.isfinite(distances)[None]
        )
        valid = np.logical_and.accumulate(valid, axis=2)
        has_cell = valid.any(axis=2)
        gaps = np.abs(distances[None] - observed_ranges[index][:, :, None])
        nearest = np.argmin(np.where(valid, gaps, np.inf), axis=2)
        hit = (observed_ranges[index] < max_range_cells) & has_cell
        slots = np.arange(length)[None, None, :]
        free = valid & ((slots < nearest[:, :, None]) | ~hit[:, :, None])
        occupied = valid & (slots == nearest[:, :, None]) & hit[:, :, None]
        flat = ray_rows * num_cols + ray_cols + slot_offsets[index][:, None, None]
        free_counts += np.bincount(flat[free], minlength=count * num_cells).reshape(
            count, num_cells
        )
        occupied_counts += np.bincount(flat[occupied], minlength=count * num_cells).reshape(
            count, num_cells
        )
    return free_counts, occupied_counts


def observed_scan_log_odds_delta(
    observed_ranges,
    template,
    row,
    col,
    num_rows,
    num_cols,
    max_range_cells,
    free_log_odds,
    occupied_log_odds,
):
    """Map measured ranges without consulting occupancy or hidden hit flags.

    Classification is :func:`observed_scan_evidence_counts`; this function only
    turns its two counts into the log-odds increment each cell accumulates.
    """
    free_counts, occupied_counts = observed_scan_evidence_counts(
        observed_ranges, template, row, col, num_rows, num_cols, max_range_cells
    )
    return free_counts * free_log_odds + occupied_counts * occupied_log_odds


def gaussian_log_density(values: ArrayLike, nominal: ArrayLike, std: float) -> np.ndarray:
    """Per-beam log density of ``values`` under ``N(nominal, std^2)``.

    The untruncated range law, kept here beside its truncated twin so the two
    branches of the sensor read as one pair rather than as an addition bolted
    onto the environment.

    Args:
        values: Measured ranges, any shape broadcastable against ``nominal``.
        nominal: Noise-free ranges of the same broadcast shape.
        std: Range noise standard deviation. Positive.

    Returns:
        Log densities, of the broadcast shape.
    """
    residual = (np.asarray(values, dtype=np.float64) - np.asarray(nominal, dtype=np.float64)) / std
    return -0.5 * residual**2 - math.log(std * math.sqrt(2.0 * math.pi))


def truncated_normal_log_norm(nominal: ArrayLike, std: float) -> np.ndarray:
    """``log Phi(nominal / std)``: the truncation normaliser, per beam.

    This is the mass the untruncated normal would have put below zero, divided
    out. It depends on ``nominal``, so it is a per-beam and per-state quantity,
    not a constant that can be hoisted out of a particle loop: two map particles
    that predict different ranges for the same beam are normalised differently,
    and dropping that difference would quietly bias every importance weight.

    Computed with ``log_ndtr`` rather than ``log(ndtr(x))``: once ``nominal``
    is more than about eight standard deviations above zero, ``ndtr`` rounds to
    exactly 1 and the log to exactly 0, whereas ``log_ndtr`` still returns the
    true ``-Phi(-x)`` tail. The difference is far below anything a particle
    weight can feel, but there is no reason to throw it away.

    Args:
        nominal: Noise-free ranges, non-negative, any shape.
        std: Range noise standard deviation. Positive.

    Returns:
        The log normaliser, of ``nominal``'s shape. Always non-positive, and
        effectively ``0.0`` once ``nominal`` is several ``std`` above zero.
    """
    return log_ndtr(np.asarray(nominal, dtype=np.float64) / std)


def truncated_normal_log_density(values: ArrayLike, nominal: ArrayLike, std: float) -> np.ndarray:
    """Per-beam log density of a normal truncated to ``[0, +inf)``.

    ``f(z | rho, sigma) = phi((z - rho)/sigma) / (sigma * Phi(rho/sigma))`` for
    ``z >= 0``, and zero below. A negative reading is impossible under this law,
    so it scores ``-inf`` rather than the small-but-finite score the untruncated
    normal would give it -- that is the whole point of the option.

    Args:
        values: Measured ranges, any shape broadcastable against ``nominal``.
        nominal: Noise-free ranges, non-negative, of the same broadcast shape.
        std: Range noise standard deviation. Positive.

    Returns:
        Log densities of the broadcast shape, ``-inf`` where ``values < 0``.
    """
    values = np.asarray(values, dtype=np.float64)
    density = gaussian_log_density(values, nominal, std) - truncated_normal_log_norm(nominal, std)
    return np.where(values >= 0.0, density, -np.inf)


def truncated_normal_from_upper_tail(
    upper_tail: ArrayLike, nominal: ArrayLike, std: float
) -> np.ndarray:
    """Quantile function of the ``[0, +inf)``-truncated normal, by upper tail.

    Returns the ``z >= 0`` whose upper-tail probability under the truncated law
    is ``upper_tail``. Written in the upper tail rather than the lower one
    because that is the numerically safe direction here: the factor
    ``Phi(nominal/std)`` is the one close to 1, and

        z = nominal - std * Phi^-1(upper_tail * Phi(nominal / std))

    then degenerates gracefully. At the environment's defaults a beam at full
    range sits ten standard deviations above zero, ``Phi(10) == 1.0`` in double
    precision, and the expression reduces exactly to the untruncated quantile --
    so turning truncation on costs nothing where truncation does not bite.

    Exact inversion, so it is equally usable for drawing samples (feed uniforms)
    and for transforming a fixed quadrature rule (feed ``Phi(-point)``). Neither
    use rejects, loops, or drops a tail.

    Args:
        upper_tail: Upper-tail probabilities in ``(0, 1]``, any shape
            broadcastable against ``nominal``. A value of exactly ``0`` would
            map to ``+inf``; callers drawing uniforms must exclude it.
        nominal: Noise-free ranges, non-negative, of the same broadcast shape.
        std: Range noise standard deviation. Positive.

    Returns:
        Ranges of the broadcast shape, all ``>= 0``.
    """
    nominal = np.asarray(nominal, dtype=np.float64)
    upper_tail = np.asarray(upper_tail, dtype=np.float64)
    values = nominal - std * ndtri(upper_tail * ndtr(nominal / std))
    # Only a floating-point guard on the boundary itself: ``upper_tail == 1``
    # is the lower end of the support, an exact zero analytically, and
    # ``ndtri(ndtr(x))`` returns it a few ulps off on either side. It is not a
    # clamp -- no interior probability mass is moved onto zero, because no
    # interior tail value reaches it.
    return np.where(upper_tail >= 1.0, 0.0, np.maximum(values, 0.0))


def sample_truncated_normal_ranges(nominal: ArrayLike, std: float) -> np.ndarray:
    """Draw one truncated-normal range per entry of ``nominal``.

    Draws from the global NumPy stream, like every other generative path in this
    environment, consuming exactly one uniform per beam.

    Args:
        nominal: Noise-free ranges, non-negative, any shape.
        std: Range noise standard deviation. Positive.

    Returns:
        Sampled ranges of ``nominal``'s shape, all ``>= 0``.
    """
    nominal = np.asarray(nominal, dtype=np.float64)
    # ``random_sample`` yields [0, 1); the complement moves that to (0, 1] so an
    # exact zero -- which the quantile maps to +inf -- can never be drawn.
    upper_tail = 1.0 - np.random.random_sample(nominal.shape)
    return truncated_normal_from_upper_tail(upper_tail, nominal, std)


class RangeNoiseModel(Enum):
    """Which per-beam range noise law the sensor draws from and scores with.

    Attributes:
        GAUSSIAN: Unbounded ``N(rho, sigma^2)``. The original law, and the
            default, so existing results stay reproducible.
        TRUNCATED_NORMAL: The same normal truncated to ``[0, +inf)`` and
            renormalised, for callers who need a range law that cannot report a
            negative distance. Readings above the nominal maximum range are
            untouched; only the impossible side is cut off.
    """

    GAUSSIAN = "gaussian"
    TRUNCATED_NORMAL = "truncated_normal"


def resolve_range_noise_model(value: Union[RangeNoiseModel, str]) -> RangeNoiseModel:
    """Coerce a member or its string value to a :class:`RangeNoiseModel`.

    Strings are accepted so an environment can be configured from a YAML file
    without importing the enum.

    Args:
        value: A :class:`RangeNoiseModel`, or one of its string values.

    Returns:
        The matching member.

    Raises:
        ValueError: If ``value`` names no member. The message lists the valid
            names, because a silently accepted typo here would fall through to
            whichever branch is written first and look like a modelling result.
    """
    if isinstance(value, RangeNoiseModel):
        return value
    try:
        return RangeNoiseModel(value)
    except ValueError as error:
        valid = ", ".join(repr(member.value) for member in RangeNoiseModel)
        raise ValueError(
            f"range_noise_model must be one of {valid} or a RangeNoiseModel, got {value!r}"
        ) from error


def range_log_density(
    values: ArrayLike, nominal: ArrayLike, std: float, model: RangeNoiseModel
) -> np.ndarray:
    """Per-beam log density of measured ranges under the selected law.

    Args:
        values: Measured ranges, any shape broadcastable against ``nominal``.
        nominal: Noise-free ranges of the same broadcast shape.
        std: Range noise standard deviation. Positive.
        model: Which range law to score under.

    Returns:
        Log densities of the broadcast shape. Sum over the beam axis for a
        scan's log density -- note the truncated law's normaliser varies per
        beam, so that sum is not the constant it is under the Gaussian law.

    Raises:
        ValueError: If ``model`` is not a known range noise model.
    """
    if model is RangeNoiseModel.GAUSSIAN:
        return gaussian_log_density(values, nominal, std)
    if model is RangeNoiseModel.TRUNCATED_NORMAL:
        return truncated_normal_log_density(values, nominal, std)
    raise ValueError(f"unknown range noise model: {model}")


def scan_log_density(
    values: ArrayLike, nominal: ArrayLike, std: float, model: RangeNoiseModel
) -> np.ndarray:
    """Log density of whole scans: the per-beam densities summed over the last axis.

    Under the Gaussian law the normaliser is one constant for the whole scan,
    and it is applied once, after the sum, in the arithmetic the environment
    used before the truncated law existed. Summing ``range_log_density`` per
    beam instead would give the same value up to rounding -- and that rounding
    is enough to move a particle filter's weights by a few ulps, which is a
    silent change to every seeded Gaussian result, including the golden
    visualization. Under the truncated law the normaliser is per beam, so
    there the per-beam sum is the only correct form.

    Args:
        values: Measured ranges, shape ``(..., num_beams)``, broadcastable
            against ``nominal``.
        nominal: Noise-free ranges of the same broadcast shape.
        std: Range noise standard deviation. Positive.
        model: Which range law to score under.

    Returns:
        Log densities of the broadcast shape with the last axis removed.

    Raises:
        ValueError: If ``model`` is not a known range noise model.
    """
    values = np.asarray(values, dtype=np.float64)
    nominal = np.asarray(nominal, dtype=np.float64)
    if model is RangeNoiseModel.GAUSSIAN:
        residual = (values - nominal) / std
        num_beams = np.broadcast_shapes(values.shape, nominal.shape)[-1]
        return -0.5 * np.sum(residual**2, axis=-1) - num_beams * np.log(std * np.sqrt(2 * np.pi))
    if model is RangeNoiseModel.TRUNCATED_NORMAL:
        return np.sum(truncated_normal_log_density(values, nominal, std), axis=-1)
    raise ValueError(f"unknown range noise model: {model}")


def sample_ranges(nominal: ArrayLike, std: float, model: RangeNoiseModel) -> np.ndarray:
    """Draw one noisy range per entry of ``nominal`` under the selected law.

    Draws from the global NumPy stream. The Gaussian branch keeps its original
    ``np.random.normal`` call, and therefore its exact seeded output, so a run
    that does not select truncation reproduces bit for bit.

    Args:
        nominal: Noise-free ranges, non-negative, any shape.
        std: Range noise standard deviation. Positive.
        model: Which range law to draw from.

    Returns:
        Sampled ranges of ``nominal``'s shape.

    Raises:
        ValueError: If ``model`` is not a known range noise model.
    """
    nominal = np.asarray(nominal, dtype=np.float64)
    if model is RangeNoiseModel.GAUSSIAN:
        return nominal + np.random.normal(0.0, std, nominal.shape)
    if model is RangeNoiseModel.TRUNCATED_NORMAL:
        return sample_truncated_normal_ranges(nominal, std)
    raise ValueError(f"unknown range noise model: {model}")


def quadrature_ranges(
    nominal: ArrayLike, standard_points: ArrayLike, std: float, model: RangeNoiseModel
) -> np.ndarray:
    """Map fixed unit-normal quadrature points to ranges under the selected law.

    The expected-reward integral uses a fixed antithetic set of unit normals so
    it consumes no global randomness. Under truncation those points must be
    pushed through the truncated quantile function rather than scaled and
    shifted, or the integral would average over scans the model cannot produce
    -- including negative ones, which is exactly what the option exists to
    prevent. Transforming by upper-tail probability keeps the points antithetic,
    because ``Phi(-x)`` and ``Phi(x)`` sum to one.

    Args:
        nominal: Noise-free ranges, non-negative, broadcastable against
            ``standard_points``.
        standard_points: Unit-variance integration points of the same broadcast
            shape.
        std: Range noise standard deviation. Positive.
        model: Which range law the points are integrating over.

    Returns:
        Ranges of the broadcast shape.

    Raises:
        ValueError: If ``model`` is not a known range noise model.
    """
    nominal = np.asarray(nominal, dtype=np.float64)
    standard_points = np.asarray(standard_points, dtype=np.float64)
    if model is RangeNoiseModel.GAUSSIAN:
        return nominal + std * standard_points
    if model is RangeNoiseModel.TRUNCATED_NORMAL:
        return truncated_normal_from_upper_tail(ndtr(-standard_points), nominal, std)
    raise ValueError(f"unknown range noise model: {model}")
