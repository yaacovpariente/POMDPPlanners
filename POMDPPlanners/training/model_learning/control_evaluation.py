# SPDX-License-Identifier: MIT

"""The number the algorithm is actually judged on: return of the planner using the model.

The diagnostics in :mod:`~POMDPPlanners.training.model_learning.diagnostics`
describe the model. This describes the *decision*. Ross & Bagnell (ICML 2012)
report nothing else: every figure in the paper is average total cost of the
controller synthesized from the round-k model, against transitions collected so
far, and no curve of prediction error appears anywhere. That is deliberate --
their whole argument is that a model can have small training error and poor
control performance, so measuring the model cannot settle the question the
algorithm asks.

Three things make the curve mean something, and each is easy to leave out.

**A baseline on the same seeds.** A single rising curve says only that more data
helps, which is not in dispute. The paper's result is the *gap* between
iterating and not: DAgger against Batch, both collecting the same amount of
data, differing only in where it came from. Batch here is the same loop with the
planner half switched off, so every other detail is held fixed by construction
rather than by care.

**Several seeds, shared across methods.** The paper uses 20 repetitions and
gives all approaches the same 20. Sharing them removes the draw of start states
from the comparison; without it a gap of one standard error is unreadable.

**The best round, not the last.** The guarantee is on the best model of the
sequence or a mixture of them. Reporting the final round's return silently
reports a different quantity than the one the theory bounds, and one that can be
worse -- the sequence is not monotone.

The x-axis is cumulative transitions rather than round index, because rounds of
two methods are not the same amount of data and comparing them by index compares
different budgets.

Classes:
    ControlPoint: One round's return, with the data it cost.
    LearningCurve: One method's points for one seed.
    AggregatedCurve: A method's curve averaged across seeds.

Functions:
    evaluate_control: Run the evaluation rollouts and build a point.
    aggregate_curves: Average one method's curves across seeds.
    best_point: The round a curve should be reported at.
    save_learning_curves: Write curves to JSON.
    load_learning_curves: Read curves back.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ControlPoint:
    """One round's closed-loop performance, and what it cost to get there.

    Attributes:
        round_index: 1-based round the model came from.
        cumulative_transitions: Transitions the fit had seen, across all rounds.
            This is the x-axis: it is comparable between methods in a way the
            round index is not.
        returns: Per-episode return of the planner searching this round's model
            in the true world. Kept in full rather than reduced, so a later
            aggregation can pool episodes instead of averaging averages.
    """

    round_index: int
    cumulative_transitions: int
    returns: Tuple[float, ...]

    @property
    def mean_return(self) -> float:
        """Mean return across the evaluation episodes, or ``nan`` if there were none."""
        return float(np.mean(self.returns)) if self.returns else float("nan")

    @property
    def standard_error(self) -> float:
        """Standard error of the mean across episodes, or ``nan`` below two episodes."""
        if len(self.returns) < 2:
            return float("nan")
        return float(np.std(self.returns, ddof=1) / np.sqrt(len(self.returns)))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view of the point."""
        return {
            "round_index": self.round_index,
            "cumulative_transitions": self.cumulative_transitions,
            "returns": list(self.returns),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ControlPoint":
        """Rebuild a point from :meth:`to_dict` output."""
        return cls(
            round_index=int(data["round_index"]),
            cumulative_transitions=int(data["cumulative_transitions"]),
            returns=tuple(float(value) for value in data["returns"]),
        )


@dataclass(frozen=True)
class LearningCurve:
    """One method's per-round performance for one seed.

    Attributes:
        method: Name of the method the curve belongs to, e.g. ``"dagger"``.
        seed: The repetition's seed. Two methods compared at the same seed saw
            the same draws, which is what makes their difference readable.
        points: One point per round, in round order.
    """

    method: str
    seed: int
    points: Tuple[ControlPoint, ...]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view of the curve."""
        return {
            "method": self.method,
            "seed": self.seed,
            "points": [point.to_dict() for point in self.points],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningCurve":
        """Rebuild a curve from :meth:`to_dict` output."""
        return cls(
            method=str(data["method"]),
            seed=int(data["seed"]),
            points=tuple(ControlPoint.from_dict(point) for point in data["points"]),
        )


@dataclass(frozen=True)
class AggregatedCurve:
    """One method's curve averaged across seeds, ready to plot.

    Attributes:
        method: The method these seeds ran.
        num_seeds: Repetitions averaged at each round.
        cumulative_transitions: Mean transitions at each round. Methods can
            differ slightly here because episodes end at different lengths, which
            is why the axis is plotted rather than assumed shared.
        mean_returns: Mean across seeds of each seed's mean return.
        standard_errors: Standard error across seeds. This is the interval that
            answers "did the methods really differ", so it is across
            repetitions, not across the episodes inside one.
    """

    method: str
    num_seeds: int
    cumulative_transitions: Tuple[float, ...]
    mean_returns: Tuple[float, ...]
    standard_errors: Tuple[float, ...]


def evaluate_control(
    model: Any,
    evaluation_fn: Callable[[Any, int, int], Sequence[float]],
    round_index: int,
    cumulative_transitions: int,
    num_episodes: int,
) -> ControlPoint:
    """Run the evaluation rollouts for one round's model and record the returns.

    The rollouts are the caller's business -- which planner, what budget, what
    device -- for the same reason the training rollouts are: the loop should not
    decide how a planner is run. All this adds is the bookkeeping that makes one
    round's number comparable to another's.

    Args:
        model: The round's fitted transition, which the planner searches.
        evaluation_fn: Called as ``fn(model, round_index, num_episodes)`` and
            expected to run the planner against ``model`` in the *true* world,
            returning one return per episode. Evaluating in the model would
            measure the model's opinion of itself.
        round_index: 1-based round the model came from.
        cumulative_transitions: Transitions the fit had seen by this round.
        num_episodes: Evaluation episodes to request.

    Returns:
        The round's point.

    Raises:
        ValueError: If ``num_episodes`` is not positive.
    """
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")
    returns = tuple(float(value) for value in evaluation_fn(model, round_index, num_episodes))
    return ControlPoint(
        round_index=round_index,
        cumulative_transitions=int(cumulative_transitions),
        returns=returns,
    )


def aggregate_curves(curves: Sequence[LearningCurve]) -> AggregatedCurve:
    """Average one method's curves across seeds, round by round.

    Each seed contributes its own mean return, and the spread reported is across
    those -- not across the episodes within a seed. A method can be very
    consistent inside one repetition and swing wildly between them, and it is
    the second number that decides whether a gap between methods is real.

    Args:
        curves: Curves for a single method, one per seed.

    Returns:
        The averaged curve, truncated to the shortest curve's number of rounds.

    Raises:
        ValueError: If ``curves`` is empty or mixes methods.
    """
    if not curves:
        raise ValueError("aggregate_curves needs at least one curve")
    methods = {curve.method for curve in curves}
    if len(methods) != 1:
        raise ValueError(f"aggregate_curves expects one method, got {sorted(methods)}")

    num_rounds = min(len(curve.points) for curve in curves)
    transitions: List[float] = []
    means: List[float] = []
    errors: List[float] = []
    for index in range(num_rounds):
        per_seed = [curve.points[index] for curve in curves]
        seed_means = [point.mean_return for point in per_seed]
        transitions.append(float(np.mean([point.cumulative_transitions for point in per_seed])))
        means.append(float(np.mean(seed_means)))
        errors.append(
            float(np.std(seed_means, ddof=1) / np.sqrt(len(seed_means)))
            if len(seed_means) > 1
            else float("nan")
        )
    return AggregatedCurve(
        method=curves[0].method,
        num_seeds=len(curves),
        cumulative_transitions=tuple(transitions),
        mean_returns=tuple(means),
        standard_errors=tuple(errors),
    )


def best_point(curve: LearningCurve) -> Optional[ControlPoint]:
    """The round with the highest mean return -- the one the guarantee is about.

    The bound covers the best model of the sequence, not the final one, and the
    sequence is not monotone: a round can be worse than the one before it and the
    loop still be working. Reporting the last round instead is the quiet way to
    report a different, smaller quantity.

    Args:
        curve: One seed's curve.

    Returns:
        The best point, or ``None`` if the curve has no points with a return.
    """
    scored = [point for point in curve.points if not np.isnan(point.mean_return)]
    if not scored:
        return None
    return max(scored, key=lambda point: point.mean_return)


def save_learning_curves(curves: Sequence[LearningCurve], path: Path) -> Path:
    """Write curves to JSON so a plot can be regenerated without re-running.

    Args:
        curves: The curves to store, across methods and seeds.
        path: Destination file. Parent directories are created.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([curve.to_dict() for curve in curves], indent=2), encoding="utf-8")
    return path


def load_learning_curves(path: Path) -> List[LearningCurve]:
    """Read curves written by :func:`save_learning_curves`.

    Args:
        path: File to read.

    Returns:
        The stored curves, in the order written.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [LearningCurve.from_dict(entry) for entry in data]
