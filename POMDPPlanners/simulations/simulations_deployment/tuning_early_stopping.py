# SPDX-License-Identifier: MIT

"""Early stopping for hyperparameter studies.

Optuna has no "run up to ``n_trials`` but stop when it stops helping" flag.
Its pruners kill a *trial* mid-way rather than the study, and they are not
available here at all: pruning needs intermediate ``trial.report`` values and
Optuna forbids it for multi-objective studies, which is what
``HyperParameterTuningSimulationTask`` creates. Optuna's
``TerminatorCallback`` is single-objective only for the same kind of reason.

What is left is ``study.stop()`` from an ``optimize(callbacks=[...])`` hook,
which is what :class:`EarlyStoppingCallback` does. ``n_trials`` then becomes an
upper bound: set it to 1000 and the study ends at trial ~100 if the Pareto
front has not moved in ``patience`` trials.

Measuring "the front has not moved" needs a single number per trial. We use the
hypervolume of the Pareto front, because it is the only common front-quality
scalar that improves if and only if the front genuinely gains ground -- a new
front member that only trades one objective away for another leaves it flat.
The objectives are normalized against bounds frozen once ``min_trials`` trials
have finished, so the number is monotone over the run and can be plotted as a
convergence curve. Without freezing, one late outlier would rescale every past
value and the curve would not be comparable to itself.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import optuna
from optuna.trial import FrozenTrial

# Exact hypervolume by slicing costs O(n^d); with more front points than this we
# keep an evenly spaced subsample. The measure stays a valid lower bound and the
# callback only needs to see improvement, not the exact value.
MAX_FRONT_POINTS_FOR_HYPERVOLUME = 128


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """When to stop a study before its trial budget is spent.

    Attributes:
        patience: Stop after this many completed trials with no front
            improvement. Objective values here are means over ``num_episodes``
            episodes and are therefore noisy, so a plateau shorter than a
            hundred trials is often sampling noise rather than convergence.
        min_trials: Never stop before this many trials have completed. Also the
            point at which the normalization bounds are frozen, so it must be
            large enough to have seen both good and bad regions of the space.
        min_relative_improvement: Front quality must grow by at least this
            fraction to count as improvement. Guards against a plateau being
            hidden by floating-point drift in the hypervolume.
    """

    patience: int = 100
    min_trials: int = 50
    min_relative_improvement: float = 1e-3

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")
        if self.min_trials <= 0:
            raise ValueError(f"min_trials must be positive, got {self.min_trials}")
        if self.min_relative_improvement < 0:
            raise ValueError(
                f"min_relative_improvement must be non-negative, "
                f"got {self.min_relative_improvement}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Config-id friendly view. Included in task ids because stopping early changes results."""
        return {
            "patience": self.patience,
            "min_trials": self.min_trials,
            "min_relative_improvement": self.min_relative_improvement,
        }


def non_dominated(points: Sequence[Sequence[float]]) -> List[Tuple[float, ...]]:
    """Return the maximization-sense Pareto front of ``points``.

    Args:
        points: Objective vectors, all in maximization sense.

    Returns:
        The subset of points not dominated by any other point.
    """
    front: List[Tuple[float, ...]] = []
    for candidate in points:
        cand = tuple(float(v) for v in candidate)
        if any(
            all(o >= c for o, c in zip(other, cand)) and any(o > c for o, c in zip(other, cand))
            for other in points
            if tuple(float(v) for v in other) != cand
        ):
            continue
        if cand not in front:
            front.append(cand)
    return front


def hypervolume(points: Sequence[Sequence[float]]) -> float:
    """Hypervolume of ``points`` in maximization sense with the origin as reference.

    Exact, by the hyperplane-slicing recursion: sort by the last coordinate,
    and for each slab between consecutive coordinate values multiply the slab
    thickness by the hypervolume of the projection of the points above it.

    Args:
        points: Objective vectors with non-negative coordinates.

    Returns:
        The volume of the union of boxes spanned by the points and the origin.
        Zero for an empty input.
    """
    if not points:
        return 0.0
    front = non_dominated(points)
    if len(front) > MAX_FRONT_POINTS_FOR_HYPERVOLUME:
        step = len(front) / MAX_FRONT_POINTS_FOR_HYPERVOLUME
        front = [front[int(i * step)] for i in range(MAX_FRONT_POINTS_FOR_HYPERVOLUME)]
    return _hypervolume_recursive([tuple(max(0.0, float(v)) for v in p) for p in front])


def _hypervolume_recursive(points: List[Tuple[float, ...]]) -> float:
    if not points:
        return 0.0
    dim = len(points[0])
    if dim == 1:
        return max(p[0] for p in points)

    ordered = sorted(points, key=lambda p: p[-1], reverse=True)
    total = 0.0
    for index, point in enumerate(ordered):
        upper = point[-1]
        lower = ordered[index + 1][-1] if index + 1 < len(ordered) else 0.0
        thickness = upper - lower
        if thickness <= 0.0:
            continue
        projection = [p[:-1] for p in ordered[: index + 1]]
        total += _hypervolume_recursive(projection) * thickness
    return total


class EarlyStoppingCallback:
    """Optuna callback that stops a study once its Pareto front stops improving.

    Pass it in ``study.optimize(callbacks=[...])``. After the run, ``history``
    holds the front-quality curve that is the evidence the study converged, and
    ``stopped_at_trial`` says whether the stop fired or the budget simply ran
    out.

    With ``n_jobs > 1`` the stop is not exact: ``study.stop()`` prevents new
    trials from starting but lets in-flight ones finish, so the study overshoots
    by up to ``n_jobs - 1`` trials.
    """

    def __init__(
        self,
        config: EarlyStoppingConfig,
        directions: Sequence[optuna.study.StudyDirection],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the callback.

        Args:
            config: Patience settings.
            directions: The study's optimization directions, used to flip
                minimized objectives so every objective is maximized.
            logger: Where to report the stop. Defaults to this module's logger.
        """
        self.config = config
        self.directions = list(directions)
        self.logger = logger or logging.getLogger(__name__)

        self.history: List[Tuple[int, float]] = []
        self.stopped_at_trial: Optional[int] = None
        self._best_quality: Optional[float] = None
        self._last_improved_at: int = 0
        self._bounds: Optional[List[Tuple[float, float]]] = None

    def __call__(self, study: optuna.study.Study, trial: FrozenTrial) -> None:
        """Record front quality for this trial and stop the study if it has plateaued."""
        completed = [
            t.values
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None
        ]
        n_done = len(completed)
        if n_done < self.config.min_trials:
            return

        quality = self.front_quality(completed)
        self.history.append((n_done, quality))

        improved = self._best_quality is None or quality > self._best_quality * (
            1.0 + self.config.min_relative_improvement
        )
        if improved:
            self._best_quality = quality
            self._last_improved_at = n_done
            return

        if n_done - self._last_improved_at >= self.config.patience:
            self.stopped_at_trial = n_done
            self.logger.info(
                "Early stopping: Pareto front did not improve over the last %d trials "
                "(stopping after %d completed trials).",
                self.config.patience,
                n_done,
            )
            study.stop()

    def front_quality(self, values_per_trial: Sequence[Sequence[float]]) -> float:
        """Normalized hypervolume of the front formed by ``values_per_trial``.

        Args:
            values_per_trial: Objective vectors of every completed trial, in the
                study's own directions.

        Returns:
            Hypervolume in units of the frozen normalization box, comparable
            across calls and monotone over the run because the bounds are
            frozen on the first call. Roughly 1 when the front fills the range
            seen early on, and larger when later trials beat that range.
        """
        maximized = [self._to_maximization(values) for values in values_per_trial]
        if self._bounds is None:
            self._bounds = self._compute_bounds(maximized)
        return hypervolume([self._normalize(values) for values in maximized])

    def _to_maximization(self, values: Sequence[float]) -> List[float]:
        return [
            -float(value) if direction == optuna.study.StudyDirection.MINIMIZE else float(value)
            for value, direction in zip(values, self.directions)
        ]

    @staticmethod
    def _compute_bounds(maximized: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
        bounds = []
        for axis in range(len(maximized[0])):
            column = [values[axis] for values in maximized]
            low, high = min(column), max(column)
            # A degenerate axis (every trial identical so far) would divide by
            # zero; give it unit width so it contributes a constant factor.
            bounds.append((low, high) if high > low else (low, low + 1.0))
        return bounds

    def _normalize(self, maximized: Sequence[float]) -> List[float]:
        """Scale to the frozen bounds, clipping only from below.

        The lower bound is the hypervolume reference point, so anything worse
        than it contributes nothing and is clipped to zero. There is no upper
        clip: a study that keeps improving past the range seen in its first
        ``min_trials`` trials must keep showing improvement, and capping at 1
        would saturate the measure and stop the study exactly when it was still
        making progress.
        """
        assert self._bounds is not None
        return [
            max(0.0, (value - low) / (high - low))
            for value, (low, high) in zip(maximized, self._bounds)
        ]


def build_early_stopping_callback(
    config: Optional[EarlyStoppingConfig],
    directions: Sequence[optuna.study.StudyDirection],
    logger: Optional[logging.Logger] = None,
) -> Optional[Callable[[optuna.study.Study, FrozenTrial], None]]:
    """Build an :class:`EarlyStoppingCallback`, or None when early stopping is off.

    Args:
        config: Patience settings, or None to disable early stopping.
        directions: The study's optimization directions.
        logger: Where the callback reports the stop.

    Returns:
        The callback, or None if ``config`` is None.
    """
    if config is None:
        return None
    return EarlyStoppingCallback(config=config, directions=directions, logger=logger)
