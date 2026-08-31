# SPDX-License-Identifier: MIT

"""Running the iterative loop against its baseline, on the same seeds.

A learning curve on its own establishes that more data helps, which nobody
doubts. The claim in Ross & Bagnell (ICML 2012) is comparative: collecting under
the learner's own policy beats collecting from a fixed exploration distribution,
at equal data. Their figures carry two families of lines for exactly that
reason, and the paper's robustness claim -- DAgger wins for *every* exploration
distribution, Batch only for one -- is only visible because both were run.

The baseline is not a separate implementation. ``Batch`` is this package's own
loop with ``planner_rollout_fn`` set to ``None``: same collector, same fit, same
aggregation, differing only in where the rows came from. Reimplementing it would
put an unrelated set of choices on the other side of the comparison and quietly
attribute their effect to the algorithm.

Two mistakes this module exists to prevent, both of which produce a plausible
plot:

**The baseline getting on-policy data.** A ``batch`` trainer built with a
planner rollout function is not the baseline -- it is a second DAgger run under
a different name, and the gap collapses for a reason that has nothing to do with
the algorithm. The configuration is checked rather than trusted.

**Methods running on different seeds.** The paper gives all approaches the same
20 seeds. Independent draws leave the difference between methods entangled with
the difference between start states, and at these episode counts that noise is
comparable to the effect. The seed is passed in here, not chosen per method.

Functions:
    run_learning_curves: Run every method at every seed and collect the curves.
    curves_by_method: Group a flat list of curves by their method name.
"""

from typing import Any, Callable, Dict, List, Sequence

from POMDPPlanners.training.model_learning.control_evaluation import LearningCurve
from POMDPPlanners.training.model_learning.dagger_trainer import DAggerModelTrainer
from POMDPPlanners.utils.logger import get_logger

#: The iterative method and the baseline it has to beat, in plot order.
DEFAULT_METHODS = ("dagger", "batch")

#: Method name whose trainer must have no planner rollouts -- see the module
#: docstring on what a mislabelled baseline does to the comparison.
BATCH_METHOD = "batch"


def run_learning_curves(
    trainer_factory: Callable[[str, int], DAggerModelTrainer],
    seeds: Sequence[int],
    methods: Sequence[str] = DEFAULT_METHODS,
    logger: Any = None,
) -> List[LearningCurve]:
    """Run each method at each seed and return one curve per pair.

    Args:
        trainer_factory: Called as ``fn(method, seed)`` and expected to return a
            freshly built :class:`DAggerModelTrainer` -- fresh because a dataset
            or learner reused across runs carries the previous run's data into
            the next one. The factory owns the difference between the methods:
            for ``"batch"`` it must pass ``planner_rollout_fn=None``. Every
            trainer must be given an ``evaluation_fn``, or there is no curve to
            collect.
        seeds: Repetition seeds, used for *every* method so the comparison is
            paired. The paper uses 20.
        methods: Method names to run.
        logger: Optional logger.

    Returns:
        One curve per ``(method, seed)``, flat, in method-then-seed order.

    Raises:
        ValueError: If ``seeds`` or ``methods`` is empty, if a trainer's seed
            does not match the one requested, if the batch trainer was given
            planner rollouts, or if a trainer has no ``evaluation_fn``.
    """
    if not seeds:
        raise ValueError("run_learning_curves needs at least one seed")
    if not methods:
        raise ValueError("run_learning_curves needs at least one method")
    log = logger or get_logger(__name__)

    curves: List[LearningCurve] = []
    for method in methods:
        for seed in seeds:
            trainer = trainer_factory(method, seed)
            _check_trainer(trainer, method, seed)
            log.info("learning curve: method %s, seed %d", method, seed)
            trainer.run()
            curves.append(trainer.learning_curve(method))
    return curves


def curves_by_method(curves: Sequence[LearningCurve]) -> Dict[str, List[LearningCurve]]:
    """Group curves by method name, keeping each method's seed order.

    Args:
        curves: Curves across methods and seeds.

    Returns:
        Method name to its curves.
    """
    grouped: Dict[str, List[LearningCurve]] = {}
    for curve in curves:
        grouped.setdefault(curve.method, []).append(curve)
    return grouped


def _check_trainer(trainer: DAggerModelTrainer, method: str, seed: int) -> None:
    """Fail loudly on the two misconfigurations that still produce a plot."""
    if trainer.seed != seed:
        raise ValueError(
            f"trainer for method {method!r} was built with seed {trainer.seed}, "
            f"but seed {seed} was requested; the comparison must be paired"
        )
    if trainer.evaluation_fn is None:
        raise ValueError(
            f"trainer for method {method!r} at seed {seed} has no evaluation_fn, "
            "so it records no control performance and contributes no curve"
        )
    if method == BATCH_METHOD and trainer.planner_rollout_fn is not None:
        raise ValueError(
            f"the {BATCH_METHOD!r} baseline must be built with planner_rollout_fn=None; "
            "with planner rollouts it is a second iterative run, not a baseline"
        )
