# SPDX-License-Identifier: MIT

"""Pluggable map update rules for occupancy-grid mapping.

The environment's transition draws a noisy scan and then folds it into a
per-cell occupancy estimate. *How* it folds it in is a modelling choice, not a
property of the world: the rule shipped here is the one every earlier result
used, but it is one of many, and comparing two of them is a research question
this package should not force a fork to answer.

So the rule is an object. One rule instance serves the scalar transition, the
batched particle kernels and both rewards, which is what stops the three from
drifting apart -- they used to hold three copies of the same arithmetic.

Two base classes:

* :class:`OccupancyUpdateRule` -- the update written on log-odds ``L``. This is
  the general interface; everything else is a convenience on top of it.
* :class:`ProbabilityOccupancyUpdateRule` -- the update written on occupancy
  probabilities ``p``, with the ``L -> p -> L`` conversion and the clamp done
  for you. The textbook statement of an inverse sensor model is in ``p``, so an
  author porting one from a paper should not have to touch a logit.

:class:`NearestCellLogOddsUpdateRule` is the concrete default and holds the
environment's original arithmetic, unchanged.

A rule is part of what the environment *is*: two rules must never share a
cache entry. :meth:`OccupancyUpdateRule.parameters` is the rule's contribution
to that identity, and it doubles as the constructor keyword arguments used to
rebuild the rule from a serialized config, so a subclass must accept every key
it reports.

Classes:
    OccupancyUpdateRule: Abstract log-odds update rule.
    ProbabilityOccupancyUpdateRule: Abstract update rule written on probabilities.
    NearestCellLogOddsUpdateRule: The environment's original rule, the default.

Functions:
    default_update_rule: Build the default rule from the environment's settings.
    non_default_update_rule_id: The identity a non-default rule contributes.
"""

import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from POMDPPlanners.core.serialization import (
    deserialize_value,
    register_deserializer,
    register_serializer,
    serialize_value,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_sensor import (
    batch_observed_scan_evidence_counts,
    observed_scan_evidence_counts,
    observed_scan_log_odds_delta,
)
from POMDPPlanners.utils.config_to_id import config_to_id

#: Marker written by the serializer below, so a serialized rule is recognisable
#: in a config file without importing anything.
_SERIALIZED_TYPE = "OccupancyUpdateRule"


# Defined before the classes: ``__init_subclass__`` registers this the moment a
# subclass body is executed, which happens while this module is still loading.
def _serialize_update_rule(rule: "OccupancyUpdateRule") -> Dict[str, Any]:
    """Serialize a rule as its class path plus its parameters."""
    return {
        "__type__": _SERIALIZED_TYPE,
        "class": f"{type(rule).__module__}.{type(rule).__qualname__}",
        "params": {key: serialize_value(value) for key, value in rule.parameters().items()},
    }


def _deserialize_update_rule(data: Any) -> Any:
    """Rebuild a rule from :func:`_serialize_update_rule`'s output.

    Anything that is not such a payload is handed back untouched, so a ``None``
    ``update_rule`` in an old config still deserializes to ``None``.
    """
    if not isinstance(data, dict) or data.get("__type__") != _SERIALIZED_TYPE:
        return data
    module_path, _, class_name = data["class"].rpartition(".")
    rule_class = getattr(importlib.import_module(module_path), class_name)
    params = {key: deserialize_value(value) for key, value in data.get("params", {}).items()}
    return rule_class(**params)


class OccupancyUpdateRule(ABC):
    """How one observed scan turns a previous occupancy map into the next one.

    A rule reads only the previous log-odds, the *observed* pose and the
    *measured* ranges. It is never given the hidden true map or a hit flag;
    a rule that needed either would be describing an oracle, not a mapper.

    Subclasses must implement :meth:`update_log_odds` and :meth:`parameters`.
    The two batched entry points have working defaults, so a rule is usable as
    soon as its scalar form exists; override them when the loop is too slow.

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import (
        ...     NearestCellLogOddsUpdateRule,
        ...     OccupancyGridMappingPOMDP,
        ... )
        >>> rule = NearestCellLogOddsUpdateRule(
        ...     free_log_odds=-1.0, occupied_log_odds=1.0, log_odds_clamp=6.0
        ... )
        >>> env = OccupancyGridMappingPOMDP(update_rule=rule)
        >>> env.config_id != OccupancyGridMappingPOMDP().config_id
        True
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Teach the serializer about every concrete rule, including a user's.

        The registry matches on the exact type, so registering the base class
        would miss every subclass. Doing it here means a rule defined in a
        user's own module serializes into an environment config without that
        user registering anything.
        """
        super().__init_subclass__(**kwargs)
        register_serializer(cls, _serialize_update_rule)

    # -- the update -----------------------------------------------------

    @abstractmethod
    def update_log_odds(
        self,
        log_odds: np.ndarray,
        row: int,
        col: int,
        heading: int,
        observed_ranges: np.ndarray,
        template: Tuple[np.ndarray, np.ndarray],
        num_rows: int,
        num_cols: int,
        max_range_cells: float,
    ) -> np.ndarray:
        """Fold one observed scan into one map.

        Args:
            log_odds: ``(num_cells,)`` previous log-odds, row-major.
            row: Observed robot row.
            col: Observed robot column.
            heading: Observed heading index.
            observed_ranges: ``(num_beams,)`` measured ranges.
            template: The heading's ``(offsets, distances)`` ray template.
            num_rows: Grid rows.
            num_cols: Grid columns.
            max_range_cells: Sensor range in cell widths. A reading at or above
                it is a miss.

        Returns:
            ``(num_cells,)`` next log-odds. Must not modify ``log_odds``.
        """

    def batch_update_log_odds(
        self,
        log_odds: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        headings: np.ndarray,
        observed_ranges: np.ndarray,
        ray_templates: Sequence[Tuple[np.ndarray, np.ndarray]],
        num_rows: int,
        num_cols: int,
        max_range_cells: float,
    ) -> np.ndarray:
        """Fold one scan per particle into one map per particle.

        The default loops over :meth:`update_log_odds`, which is correct for
        any rule and fast enough for a few hundred particles. A rule used
        inside a planner's belief update should override it.

        Args:
            log_odds: ``(N, num_cells)`` previous log-odds.
            rows: ``(N,)`` observed robot rows.
            cols: ``(N,)`` observed robot columns.
            headings: ``(N,)`` observed heading indices.
            observed_ranges: ``(N, num_beams)`` measured ranges.
            ray_templates: The per-heading ray templates.
            num_rows: Grid rows.
            num_cols: Grid columns.
            max_range_cells: Sensor range in cell widths.

        Returns:
            ``(N, num_cells)`` next log-odds.
        """
        return np.stack(
            [
                self.update_log_odds(
                    log_odds[index],
                    int(rows[index]),
                    int(cols[index]),
                    int(headings[index]),
                    observed_ranges[index],
                    ray_templates[int(headings[index])],
                    num_rows,
                    num_cols,
                    max_range_cells,
                )
                for index in range(log_odds.shape[0])
            ]
        )

    def shared_update_log_odds(
        self,
        log_odds: np.ndarray,
        row: int,
        col: int,
        heading: int,
        observed_ranges: np.ndarray,
        ray_templates: Sequence[Tuple[np.ndarray, np.ndarray]],
        num_rows: int,
        num_cols: int,
        max_range_cells: float,
    ) -> np.ndarray:
        """Fold *one* scan into every particle's map.

        This is the filter's hot path: conditioning on a real observation gives
        every whole-map particle the same observed pose and the same measured
        ranges, and only the previous maps differ. The default broadcasts the
        one scan into ``N`` rows and defers to :meth:`batch_update_log_odds`;
        a rule whose scan classification does not depend on the previous map
        can override this to classify once.

        Args:
            log_odds: ``(N, num_cells)`` previous log-odds.
            row: The observed robot row, shared by every particle.
            col: The observed robot column.
            heading: The observed heading index.
            observed_ranges: ``(num_beams,)`` measured ranges.
            ray_templates: The per-heading ray templates.
            num_rows: Grid rows.
            num_cols: Grid columns.
            max_range_cells: Sensor range in cell widths.

        Returns:
            ``(N, num_cells)`` next log-odds.
        """
        count = log_odds.shape[0]
        return self.batch_update_log_odds(
            log_odds,
            np.full(count, int(row), dtype=np.int64),
            np.full(count, int(col), dtype=np.int64),
            np.full(count, int(heading), dtype=np.int64),
            np.broadcast_to(observed_ranges, (count, observed_ranges.shape[-1])),
            ray_templates,
            num_rows,
            num_cols,
            max_range_cells,
        )

    # -- identity -------------------------------------------------------

    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """The rule's parameters, as constructor keyword arguments.

        Used for two things at once, deliberately: it is the rule's
        contribution to the environment's ``config_id``, and it is what a
        serialized config is rebuilt from. A subclass must therefore accept
        every key it returns as a keyword argument, and must report every
        parameter that changes the update -- a parameter left out here is a
        parameter under which two different rules would silently share a cache
        entry.

        Returns:
            A JSON-serializable dict of parameter names to values.
        """

    @property
    def config_id(self) -> str:
        """Deterministic identifier of this rule, class included."""
        return config_to_id(
            {
                "class": f"{type(self).__module__}.{type(self).__qualname__}",
                "parameters": dict(sorted(self.parameters().items())),
            }
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, OccupancyUpdateRule):
            return NotImplemented
        return type(self) is type(other) and self.config_id == other.config_id

    def __hash__(self) -> int:
        return hash(self.config_id)

    def __repr__(self) -> str:
        arguments = ", ".join(
            f"{key}={value!r}" for key, value in sorted(self.parameters().items())
        )
        return f"{type(self).__name__}({arguments})"


class ProbabilityOccupancyUpdateRule(OccupancyUpdateRule):
    """An update rule written on occupancy probabilities instead of log-odds.

    Implement :meth:`update_probabilities`; this class classifies the scan,
    converts ``L -> p`` before your method and ``p -> L`` after it, and applies
    the clamp. The classification is the environment's: a reading below the
    maximum range marks the cells before the nearest ray-cell centre free and
    that cell occupied, a reading at or above it marks the whole in-grid ray
    free, and the robot's own cell takes one free sighting.

    A rule that needs a different *classification*, not just different
    arithmetic on the counts, should subclass :class:`OccupancyUpdateRule`
    directly.

    Attributes:
        log_odds_clamp: Symmetric bound applied to the resulting log-odds,
            which is the same thing as clipping ``p`` to
            ``[sigmoid(-clamp), sigmoid(clamp)]``.
    """

    def __init__(self, log_odds_clamp: float = 6.0):
        """Initialize the rule.

        Args:
            log_odds_clamp: Symmetric bound on the resulting log-odds. Must be
                positive; without it a cell swept by many beams becomes
                unrevisable.

        Raises:
            ValueError: If ``log_odds_clamp`` is not positive.
        """
        if float(log_odds_clamp) <= 0.0:
            raise ValueError(f"log_odds_clamp must be positive, got {log_odds_clamp}")
        self.log_odds_clamp = float(log_odds_clamp)

    @abstractmethod
    def update_probabilities(
        self,
        probabilities: np.ndarray,
        free_counts: np.ndarray,
        occupied_counts: np.ndarray,
    ) -> np.ndarray:
        """Return the next occupancy probabilities.

        Called once for a scalar update and once for a whole batch, so write it
        with array operations and it serves both.

        Args:
            probabilities: Previous occupancy probabilities in ``(0, 1)``,
                shaped ``(num_cells,)`` or ``(N, num_cells)``.
            free_counts: How many free sightings this scan gave each cell,
                same shape, including the robot's own cell.
            occupied_counts: How many occupied sightings this scan gave each
                cell, same shape.

        Returns:
            The next probabilities, same shape. A cell with no evidence should
            be returned unchanged.
        """

    def update_log_odds(
        self,
        log_odds,
        row,
        col,
        heading,
        observed_ranges,
        template,
        num_rows,
        num_cols,
        max_range_cells,
    ):
        """Classify the scan, hand ``p`` to :meth:`update_probabilities`, clamp."""
        free_counts, occupied_counts = observed_scan_evidence_counts(
            observed_ranges, template, row, col, num_rows, num_cols, max_range_cells
        )
        free_counts = free_counts.copy()
        free_counts[int(row) * int(num_cols) + int(col)] += 1
        return self._apply(log_odds, free_counts, occupied_counts)

    def batch_update_log_odds(
        self,
        log_odds,
        rows,
        cols,
        headings,
        observed_ranges,
        ray_templates,
        num_rows,
        num_cols,
        max_range_cells,
    ):
        """The batched twin of :meth:`update_log_odds`, in one array pass."""
        free_counts, occupied_counts = batch_observed_scan_evidence_counts(
            observed_ranges,
            ray_templates,
            rows,
            cols,
            headings,
            num_rows,
            num_cols,
            max_range_cells,
        )
        free_counts[np.arange(free_counts.shape[0]), rows * int(num_cols) + cols] += 1
        return self._apply(log_odds, free_counts, occupied_counts)

    def _apply(
        self, log_odds: np.ndarray, free_counts: np.ndarray, occupied_counts: np.ndarray
    ) -> np.ndarray:
        """Convert to ``p``, run the author's rule, convert back and clamp.

        The previous log-odds are already clamped, so ``p`` never reaches 0 or
        1 going in. It can come back as either -- a rule is allowed to say
        "certainly occupied" -- so the logit is taken under ``errstate`` and the
        resulting infinity is what the clamp is there to absorb.
        """
        probabilities = 1.0 / (1.0 + np.exp(-np.asarray(log_odds, dtype=np.float64)))
        updated = np.asarray(
            self.update_probabilities(probabilities, free_counts, occupied_counts),
            dtype=np.float64,
        )
        if updated.shape != probabilities.shape:
            raise ValueError(
                "update_probabilities must return the shape it was given, got "
                f"{updated.shape} for {probabilities.shape}"
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            next_log_odds = np.log(updated) - np.log1p(-updated)
        # A NaN would come from p outside [0, 1], which is an author error the
        # clamp cannot fix and which would otherwise poison the map silently.
        if np.any(np.isnan(next_log_odds)):
            raise ValueError("update_probabilities returned a value outside [0, 1]")
        return np.clip(next_log_odds, -self.log_odds_clamp, self.log_odds_clamp)

    def parameters(self) -> Dict[str, Any]:
        """The clamp. A subclass with its own parameters extends this dict."""
        return {"log_odds_clamp": self.log_odds_clamp}


class NearestCellLogOddsUpdateRule(OccupancyUpdateRule):
    """The environment's original rule: constant evidence at the nearest cell.

    Per beam, a reading below the maximum range selects the nearest valid
    ray-cell centre as the hit cell, gives every valid cell before it
    ``free_log_odds`` and that cell ``occupied_log_odds``, and leaves the cells
    behind it alone. A reading at or above the maximum range frees the whole
    in-grid ray and marks nothing occupied. The robot's own cell takes one
    further ``free_log_odds``. All terms are summed and the total is clamped
    once -- never per term, which would make the order of the beams matter.

    This is the rule every result produced before rules became pluggable used,
    and it stays the default. Its arithmetic is fixed: a golden episode hash in
    the test suite separates maps that differ by 1e-13.

    Attributes:
        free_log_odds: Increment for a cell a beam passed through. Negative.
        occupied_log_odds: Increment for the cell a beam stopped in. Positive.
        log_odds_clamp: Symmetric bound on the accumulated log-odds.
    """

    def __init__(self, free_log_odds: float, occupied_log_odds: float, log_odds_clamp: float):
        """Initialize the rule.

        Args:
            free_log_odds: Increment for a cell a beam passed through. Must be
                negative, or free space would be evidence of occupancy.
            occupied_log_odds: Increment for the cell a beam stopped in. Must
                be positive.
            log_odds_clamp: Symmetric bound on the accumulated log-odds. Must
                be positive.

        Raises:
            ValueError: If any of the three is on the wrong side of zero.
        """
        if float(free_log_odds) >= 0.0:
            raise ValueError(f"free_log_odds must be negative, got {free_log_odds}")
        if float(occupied_log_odds) <= 0.0:
            raise ValueError(f"occupied_log_odds must be positive, got {occupied_log_odds}")
        if float(log_odds_clamp) <= 0.0:
            raise ValueError(f"log_odds_clamp must be positive, got {log_odds_clamp}")
        self.free_log_odds = float(free_log_odds)
        self.occupied_log_odds = float(occupied_log_odds)
        self.log_odds_clamp = float(log_odds_clamp)

    def update_log_odds(
        self,
        log_odds,
        row,
        col,
        heading,
        observed_ranges,
        template,
        num_rows,
        num_cols,
        max_range_cells,
    ):
        """Sum every beam's constant evidence, add the robot's cell, clamp once."""
        delta = observed_scan_log_odds_delta(
            observed_ranges,
            template,
            row,
            col,
            num_rows,
            num_cols,
            max_range_cells,
            self.free_log_odds,
            self.occupied_log_odds,
        )
        delta[row * num_cols + col] += self.free_log_odds
        return np.clip(log_odds + delta, -self.log_odds_clamp, self.log_odds_clamp)

    def batch_update_log_odds(
        self,
        log_odds,
        rows,
        cols,
        headings,
        observed_ranges,
        ray_templates,
        num_rows,
        num_cols,
        max_range_cells,
    ):
        """One array pass over every particle, grouped by heading."""
        return np.clip(
            log_odds
            + self._deltas(
                observed_ranges,
                ray_templates,
                rows,
                cols,
                headings,
                num_rows,
                num_cols,
                max_range_cells,
            ),
            -self.log_odds_clamp,
            self.log_odds_clamp,
        )

    def shared_update_log_odds(
        self,
        log_odds,
        row,
        col,
        heading,
        observed_ranges,
        ray_templates,
        num_rows,
        num_cols,
        max_range_cells,
    ):
        """Classify the one shared scan once, then broadcast its increment.

        The increment does not depend on the previous map, so a filter with a
        thousand map particles classifies one scan rather than a thousand
        identical ones.
        """
        delta = observed_scan_log_odds_delta(
            observed_ranges,
            ray_templates[int(heading)],
            row,
            col,
            num_rows,
            num_cols,
            max_range_cells,
            self.free_log_odds,
            self.occupied_log_odds,
        )
        delta[int(row) * int(num_cols) + int(col)] += self.free_log_odds
        return np.clip(log_odds + delta[None, :], -self.log_odds_clamp, self.log_odds_clamp)

    def _deltas(
        self,
        observed_ranges,
        ray_templates,
        rows,
        cols,
        headings,
        num_rows,
        num_cols,
        max_range_cells,
    ) -> np.ndarray:
        """Per-particle log-odds increments, before the clamp."""
        free_counts, occupied_counts = batch_observed_scan_evidence_counts(
            observed_ranges,
            ray_templates,
            rows,
            cols,
            headings,
            num_rows,
            num_cols,
            max_range_cells,
        )
        deltas = free_counts * self.free_log_odds + occupied_counts * self.occupied_log_odds
        deltas[np.arange(deltas.shape[0]), rows * int(num_cols) + cols] += self.free_log_odds
        return deltas

    def parameters(self) -> Dict[str, Any]:
        """The three constants that define the update."""
        return {
            "free_log_odds": self.free_log_odds,
            "occupied_log_odds": self.occupied_log_odds,
            "log_odds_clamp": self.log_odds_clamp,
        }


def default_update_rule(
    free_log_odds: float, occupied_log_odds: float, log_odds_clamp: float
) -> NearestCellLogOddsUpdateRule:
    """The rule an environment uses when it is given none.

    Args:
        free_log_odds: The environment's ``miss_probability`` in log-odds.
        occupied_log_odds: The environment's ``hit_probability`` in log-odds.
        log_odds_clamp: The environment's clamp.

    Returns:
        The default rule built from those three.
    """
    return NearestCellLogOddsUpdateRule(
        free_log_odds=free_log_odds,
        occupied_log_odds=occupied_log_odds,
        log_odds_clamp=log_odds_clamp,
    )


def non_default_update_rule_id(
    rule: OccupancyUpdateRule,
    free_log_odds: float,
    occupied_log_odds: float,
    log_odds_clamp: float,
) -> Optional[str]:
    """The identity ``rule`` adds to a config, or ``None`` when it adds nothing.

    The default rule is fully described by ``hit_probability``,
    ``miss_probability`` and ``log_odds_clamp``, which a configuration already
    carries. Returning ``None`` for it is what keeps every cached result and
    every pinned ``config_id`` from before this option existed valid. Any other
    rule -- including the default class with different constants -- returns an
    identifier, so it can never collide with them.

    Args:
        rule: The environment's or updater's rule.
        free_log_odds: The configuration's free-space log-odds.
        occupied_log_odds: The configuration's occupied log-odds.
        log_odds_clamp: The configuration's clamp.

    Returns:
        ``rule.config_id``, or ``None`` if ``rule`` is the default one.
    """
    default = default_update_rule(free_log_odds, occupied_log_odds, log_odds_clamp)
    if rule.config_id == default.config_id:
        return None
    return rule.config_id


# Registered under both spellings because the registry is keyed on the exact
# annotation the caller hands it, and ``Environment.from_dict`` reads the
# constructor's ``Optional[OccupancyUpdateRule]`` rather than the bare class.
register_deserializer(OccupancyUpdateRule, _deserialize_update_rule)
register_deserializer(
    Optional[OccupancyUpdateRule],  # type: ignore[arg-type]
    _deserialize_update_rule,
)
