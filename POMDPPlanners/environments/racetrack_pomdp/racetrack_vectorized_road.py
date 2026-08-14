# SPDX-License-Identifier: MIT

"""Where the road bends, in torch: the curvature sources a vectorized model plugs in.

:class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_vectorized_model.RacetrackVectorizedModel`
is one implementation whatever the planner knows about the circuit. The one place that
knowledge enters is here, and it is asked twice:

* **Where does the road bend under each particle?** The torch counterpart of the scalar
  model's ``_curvature_for`` — a table lookup by arclength for a planner with a map, a single
  live estimate for one without. This drives the transition.
* **What should the camera's curvature channel read from there?** The torch counterpart of
  ``curvature_ahead``. This drives the likelihood, and it is the one term that scores a
  particle's arclength — but only for a planner holding a map, because a mapless one derives
  its answer from the very reading it is scoring.

Keeping them out of the model module is not only about length. A third kind of planner — one
holding a different map, or a different estimator — is an addition here and no edit there,
which is what ``curvature_source=`` on the model exists for.

Classes:
    TrackMapCurvature: Curvature by arclength lookup, for a planner holding a map.
    ObservedCurvature: One live estimate, for a planner taking the road from its camera.
"""

import numbers
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp import RacetrackModelPOMDP
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import EGO_ARCLENGTH_M
from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import TrackGeometry

# Given the ego block at the start of a substep, ``[N, EGO_STATE_WIDTH]``, return the signed
# curvature in 1/m under each row, ``[N]``. The torch counterpart of the scalar model's
# ``_curvature_for``, and the one thing that differs between a planner with a map and one
# without. A plain callable rather than a protocol, so a caller can pass a lambda; a source
# that can also look ahead exposes a ``curvature_ahead`` method, which
# :func:`curvature_ahead_of` finds by name.
CurvatureSource = Callable[[Tensor], Tensor]

# The public attribute a mapless scalar model exposes its per-step estimate on. Read by name
# rather than by class, so a model does not have to be one of the two shipped ones.
_CURVATURE_ESTIMATE_ATTR = "curvature_estimate"

# The method a curvature source exposes when it can predict the camera's channel, and the
# method a scalar model exposes for the same thing.
_CURVATURE_AHEAD_ATTR = "curvature_ahead"


class TrackMapCurvature:
    """Torch mirror of ``TrackGeometry.curvature_at``: a table lookup by arclength.

    Holds the profile as device tensors so the whole substep loop stays on one device. The
    NumPy original is not called here on purpose — see the module docstring — but the two
    agree by construction: the same floored modulo, the same ``searchsorted(..., right) - 1``
    and the same clamp, so a rollout on the map cannot diverge between the two models.

    Attributes:
        total_length_m: Length of one lap in metres, the modulus the arclength wraps on.
    """

    def __init__(
        self,
        geometry: TrackGeometry,
        device: torch.device,
        dtype: torch.dtype,
        lookahead_m: Any = (),
    ) -> None:
        """Move a curvature profile onto a device.

        Args:
            geometry: The lap's piecewise-constant curvature profile.
            device: Device the lookup's tensors live on.
            dtype: Floating dtype of the returned curvature.
            lookahead_m: Distances the camera's curvature channel reports at, used by
                :meth:`curvature_ahead`. Defaults to none, which makes that method return an
                empty channel.
        """
        # The starts are held in the *model's* dtype, matching the arclength they are
        # compared against, and that is deliberate. In a float32 model the arclength itself
        # is already rounded -- a boundary like 372.2208 m is off by up to 1.1e-5 m before
        # the lookup sees it -- so holding the starts in float64 cannot recover the
        # intended segment, and measurably makes it worse: on the shipped circuit's nine
        # boundaries, float32 starts agree with NumPy on all nine and float64 starts on
        # five. Matching dtypes is what makes a boundary compare equal.
        self._starts = torch.as_tensor(
            np.asarray(geometry.segment_starts, dtype=np.float64), dtype=dtype, device=device
        )
        self._curvatures = torch.as_tensor(
            np.asarray(geometry.segment_curvatures, dtype=np.float64), dtype=dtype, device=device
        )
        self._lookahead = torch.as_tensor(
            np.asarray(lookahead_m, dtype=np.float64).reshape(-1), dtype=dtype, device=device
        )
        self.total_length_m = float(geometry.total_length_m)

    def __call__(self, ego: Tensor) -> Tensor:
        return self._lookup(ego[:, EGO_ARCLENGTH_M])

    def curvature_ahead(self, ego: Tensor) -> Tensor:
        """Curvature at each lookahead distance past every row's arclength, ``[N, L]``."""
        return self._lookup(ego[:, EGO_ARCLENGTH_M][:, None] + self._lookahead[None, :])

    def _lookup(self, arclength: Tensor) -> Tensor:
        distance = torch.remainder(arclength, self.total_length_m)
        flat = distance.reshape(-1).contiguous()
        index = torch.searchsorted(self._starts, flat, right=True) - 1
        return self._curvatures[index.clamp_(0, self._curvatures.shape[0] - 1)].reshape(
            distance.shape
        )


class ObservedCurvature:
    """A single estimated curvature, refreshed each real step and shared by every row.

    The counterpart of :class:`TrackMapCurvature` for a planner with no map, which takes one
    curvature per step off the camera and has nothing better to assume further ahead. The
    value is read through a callable rather than copied in, because the estimate is replaced
    on every real step while this model is built once: caching it would freeze the planner on
    the corner it happened to start in.

    :meth:`curvature_ahead` holds that one value across the whole channel, which mirrors the
    scalar model exactly — and means the term cannot separate two particles, since it did not
    come from either of them.
    """

    def __init__(self, read_curvature: Callable[[], float], lookahead_count: int = 0) -> None:
        """Wrap a live per-step curvature estimate.

        Args:
            read_curvature: Zero-argument callable returning the current estimate in 1/m.
            lookahead_count: Number of samples the camera's curvature channel carries.
                Defaults to 0, an empty channel.
        """
        self._read_curvature = read_curvature
        self._lookahead_count = int(lookahead_count)

    def __call__(self, ego: Tensor) -> Tensor:
        return torch.full_like(ego[:, EGO_ARCLENGTH_M], float(self._read_curvature()))

    def curvature_ahead(self, ego: Tensor) -> Tensor:
        """The one estimate, held across the channel, ``[N, L]``."""
        return self(ego)[:, None].expand(-1, self._lookahead_count)


def curvature_ahead_of(source: CurvatureSource, ego: Tensor, lookahead_count: int) -> Tensor:
    """Ask a curvature source what the camera's channel should read, ``[N, L]``.

    Found by name rather than by type so a caller may pass a plain callable for the
    transition and get the honest fallback here: the curvature under the ego, held across the
    channel. A source that can do better says so by exposing ``curvature_ahead``.

    Args:
        source: The model's curvature source.
        ego: The ego block, ``[N, EGO_STATE_WIDTH]``.
        lookahead_count: Number of samples the channel carries.

    Returns:
        ``[N, lookahead_count]`` signed curvature in 1/m.
    """
    # Annotated Any rather than narrowed: getattr on a Callable gives pyright an unknown, and
    # declaring the intent once is cheaper than a type: ignore at the call.
    ahead: Any = getattr(source, _CURVATURE_AHEAD_ATTR, None)
    if ahead is None:
        return source(ego)[:, None].expand(-1, lookahead_count)
    return ahead(ego)


def resolve_curvature_source(
    env: RacetrackModelPOMDP, device: torch.device, dtype: torch.dtype
) -> CurvatureSource:
    """Pick the torch curvature lookup matching the scalar model's own ``_curvature_for``.

    Dispatches on what the model exposes rather than on its class, so a third subclass with
    a track map or a per-step estimate works without editing this module.

    Args:
        env: The scalar model whose curvature source is being mirrored.
        device: Device the lookup's tensors live on.
        dtype: Floating dtype of the returned curvature.

    Returns:
        A callable mapping an ego block to a per-row curvature in 1/m, which also knows what
        the camera's curvature channel should read.

    Raises:
        ValueError: If the model exposes neither a track map nor a per-step estimate.
            Rejected rather than defaulted to zero curvature: a silent straight-line model
            would still run, still look plausible, and drive through every corner.
    """
    lookahead = tuple(env.curvature_lookahead_m)
    geometry = getattr(env, "track_geometry", None)
    if isinstance(geometry, TrackGeometry):
        return TrackMapCurvature(geometry, device, dtype, lookahead_m=lookahead)
    if _has_curvature_estimate(env):
        return ObservedCurvature(
            lambda: float(getattr(env, _CURVATURE_ESTIMATE_ATTR)), len(lookahead)
        )
    raise ValueError(
        f"Cannot infer a curvature source from {type(env).__name__}: it exposes neither a "
        f"'track_geometry' nor a '{_CURVATURE_ESTIMATE_ATTR}'. Pass curvature_source= "
        f"explicitly. Defaulting to zero curvature would give a model that runs, looks "
        f"plausible, and drives straight through every corner."
    )


def _has_curvature_estimate(env: RacetrackModelPOMDP) -> bool:
    # numbers.Real rather than float, so an estimate held as a NumPy scalar -- the natural
    # thing to fall out of reading one off an observation -- is recognised too.
    return isinstance(getattr(env, _CURVATURE_ESTIMATE_ATTR, None), numbers.Real)


__all__ = [
    "CurvatureSource",
    "ObservedCurvature",
    "TrackMapCurvature",
    "curvature_ahead_of",
    "resolve_curvature_source",
]
