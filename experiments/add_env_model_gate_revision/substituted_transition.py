# SPDX-License-Identifier: MIT

"""Checks for an environment whose transition has been swapped for another one.

Planning with a fitted or re-implemented transition means building a model that
is the task in every respect except its dynamics. Three ways that goes wrong are
silent -- no exception, no shape error, just a study that measures something
other than what it claims -- and each has actually happened here:

**Only one dynamics path gets swapped.** An environment usually has three: the
single-step sampler, a batch path a particle filter uses, and the log-density a
belief update weights with. Batch paths are commonly written against the
concrete analytic transition rather than the stored one, so replacing the stored
transition leaves the planner searching the original dynamics through the filter
while the single-step path uses the new one.

**A carried channel gets driven.** The substitute is fitted over the channels
the dynamics move; everything else -- a latent drawn once per episode, a goal, a
terminal flag -- must be copied through untouched. A substitute that writes the
latent flattens the belief's variance over it, which is the quantity a
risk-sensitive planner is there to price.

**The observation widens.** A twin that builds its observation from the whole
state hands the planner the latent the task hides. It reads as a much better
planner, and it is not planning at all.

Functions:
    assert_transition_is_used_everywhere: Every dynamics path consults the substitute.
    assert_carried_channels_preserved: Non-driven channels survive a step unchanged.
    assert_observation_matches_reference: An observation is no wider than the task exposes.
    RecordingTransition: A transition that records which paths asked it for anything.
"""

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from POMDPPlanners.core.environment import TransitionModel


class RecordingTransition(TransitionModel):
    """A transition that answers plausibly and records that it was asked.

    The recording is the point: a path that never consults it is a path still
    wired to the environment's original dynamics.

    Attributes:
        calls: Method name to how many times it was called.

    Example:
        >>> import numpy as np
        >>> transition = RecordingTransition(dim=2)
        >>> _ = transition.sample_next_state(np.zeros(2), np.zeros(1))
        >>> transition.calls["sample_next_state"]
        1
    """

    def __init__(self, dim: int, shift: float = 7.0) -> None:
        """Initialize the recorder.

        Args:
            dim: Width of the state block this transition drives.
            shift: Constant added to every driven channel, chosen large enough
                that a successor produced by the original dynamics instead is
                obvious rather than merely different.
        """
        self.dim = int(dim)
        self.shift = float(shift)
        self.calls: Dict[str, int] = {}

    def _record(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        """Return ``state + shift``, recording the call."""
        del action
        self._record("sample_next_state")
        rows = np.atleast_2d(np.asarray(state, dtype=float)) + self.shift
        if rows.shape[0] == 1 and n_samples > 1:
            rows = np.repeat(rows, n_samples, axis=0)
        return rows[0] if (n_samples == 1 and np.asarray(state).ndim == 1) else rows

    def log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        """Return a flat log-density, recording the call."""
        del state, action
        self._record("log_probability")
        return np.zeros(np.atleast_2d(np.asarray(next_states, dtype=float)).shape[0])


def assert_transition_is_used_everywhere(
    build_model: Callable[[RecordingTransition], Any],
    state: np.ndarray,
    action: Any,
    dim: int,
    paths: Sequence[str] = ("sample_next_state", "sample_next_state_batch", "transition_log_probability"),
) -> None:
    """Assert every dynamics path of the model consults the substituted transition.

    Args:
        build_model: Called with a :class:`RecordingTransition` and expected to
            return the environment built to plan with it.
        state: A valid full-width state for that environment.
        action: An action the environment accepts.
        dim: Width of the block the substitute drives.
        paths: Model methods to exercise. A model that does not define one is
            skipped rather than failed -- not every environment has a batch path.

    Raises:
        AssertionError: If a path exists, runs, and never consults the substitute.
    """
    for path in paths:
        transition = RecordingTransition(dim=dim)
        model = build_model(transition)
        method = getattr(model, path, None)
        if method is None:
            continue
        if path == "sample_next_state":
            method(state, action)
        elif path == "sample_next_state_batch":
            method(np.atleast_2d(state), action)
        else:
            method(state, action, np.atleast_2d(state))
        assert transition.calls, (
            f"{path} produced a result without consulting the substituted transition, so it is "
            "still running the environment's own dynamics"
        )


def assert_carried_channels_preserved(
    model: Any,
    state: np.ndarray,
    action: Any,
    carried_indices: Sequence[int],
    num_samples: int = 16,
) -> None:
    """Assert the channels outside the driven block survive a step unchanged.

    Args:
        model: The environment built with the substituted transition.
        state: A full-width state whose carried channels are non-zero, so a
            substitute that zeroes them is caught.
        action: An action the environment accepts.
        carried_indices: Flat indices of the channels that must not move.
        num_samples: Successors to draw.

    Raises:
        AssertionError: If any carried channel changed in any successor.
    """
    indices = np.asarray(carried_indices, dtype=int)
    source = np.asarray(state, dtype=float).reshape(-1)[indices]
    successors = np.atleast_2d(model.sample_next_state(state, action, num_samples))
    assert np.allclose(successors[:, indices], source[np.newaxis, :]), (
        "a carried channel moved: the substitute is driving a block it was not fitted over, "
        "which flattens the belief's variance over it"
    )


def _observation_width(observation: Any) -> int:
    """Total numbers in an observation, whether it arrives as a dict or an array.

    Isaac models report a ``{channel: values}`` mapping; a vectorized twin reports
    a flat row. The quantity that matters is the same either way: how much the
    planner is told.
    """
    if isinstance(observation, dict):
        return int(sum(np.asarray(value, dtype=float).size for value in observation.values()))
    return int(np.atleast_2d(np.asarray(observation, dtype=float)).shape[-1])


def assert_observation_matches_reference(
    model: Any,
    reference: Any,
    state: np.ndarray,
    action: Optional[Any] = None,
) -> None:
    """Assert a model observes no more than the reference environment exposes.

    Args:
        model: The environment under test -- a twin, or one built with a
            substituted transition.
        reference: The environment whose observation defines what the task
            exposes.
        state: A full-width state both accept.
        action: An action, for models whose observation depends on one.

    Raises:
        AssertionError: If the model's observation carries more numbers than the
            reference's, which is the shape of handing the planner a hidden channel.
    """
    observed = _observation_width(
        model.sample_observation(state, action) if action is not None
        else model.sample_observation(state)
    )
    expected = _observation_width(
        reference.sample_observation(state, action) if action is not None
        else reference.sample_observation(state)
    )
    assert observed <= expected, (
        f"the model observes {observed} numbers where the task exposes {expected}; "
        "the extra channels are ones the planner should have to infer"
    )
