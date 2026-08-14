# SPDX-License-Identifier: MIT

"""The arithmetic behind "was there something there, and how sure are we?".

Shared by both arms of the matched pair and mirrored by the torch model, so the POMDP arm's
radar detections and the MDP arm's slot flags cannot drift apart on the questions they both
have to answer.

**Whether.** A model that scores only *where* a vehicle is has no way to charge a particle
whose slots are empty against a reading full of traffic — it scores its ego row and nothing
else, and comes out ahead for holding no hypothesis at all. :func:`detected_probabilities`
fixes that by pushing an occupancy ``q`` through the detection channel::

    p = q * (1 - miss) + (1 - q) * false_alarm

which is the textbook composition. ``false_alarm`` is what a particle carrying no vehicle
pays for a detection the observation reports; ``miss`` is what a particle carrying one pays
for a detection the observation does not.

Both rates are **zero by default**, because this world's detection decision is deterministic:
the range gate and the occlusion rule run on the vehicles' true positions and nothing is
randomly dropped or invented. So a particle whose visibility prediction contradicts the
reading is excluded rather than discounted, which is what Bayes says to do with a hypothesis
the data rules out. Nonzero rates model a lossy radar — a legitimate configuration, and not
this world.

:func:`bernoulli_log_prob` clips to :data:`PROBABILITY_EPS`, so a contradiction costs about
27.6 nats rather than ``-inf``. That is a numerical guard and not a sensor property: on a
finite particle set an all-zero weight vector is a crash, not an inference.

**What.** A bare false-alarm *rate* is not comparable with a matched detection's *density*,
and subtracting the two inverts the likelihood — measured on the MDP arm before its clutter
term existed, a state holding the observed vehicle scored 1.20 nats *worse* than one holding
nothing. So a false alarm reports a phantom drawn from :func:`cauchy_draw` and scored by
:func:`cauchy_log_prob`, as PDA and JPDA have done since the 1970s. Cauchy rather than the
usual uniform-over-the-field-of-view because these slots are ranked by range with no window
to be uniform over: the heavy tail keeps a genuinely distant report at a log cost instead of
a quadratic one, and the support is all of R so no observation can be impossible.

Functions:
    detected_probabilities: Push occupancy probabilities through the detection channel.
    bernoulli_log_prob: Log-likelihood of an observed boolean array under those.
    gaussian_log_prob: Log-density of a zero-mean isotropic Gaussian, summed.
    cauchy_draw: Zero-median Cauchy draws, by inverse transform.
    cauchy_log_prob: Log-density of a zero-median Cauchy, summed.
    pack_detections: Order detection rows by range and pad them into fixed slots.
    validate_detection_rates: Reject rates the composition above cannot use.
"""

from typing import Tuple

import numpy as np

from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    DETECTION_PRESENT,
    DETECTION_REL_X,
    DETECTION_SLOT_WIDTH,
)

# A numerical floor, not a sensor property. The shipped detection rates are 0, so a particle
# contradicting the reading's visibility genuinely has probability zero; clipping charges it
# about 27.6 nats instead, because a weight vector of all zeros crashes the filter's
# normalisation rather than telling it anything.
PROBABILITY_EPS = 1e-12

# What one detection report carries before the presence flag is prepended: the relative
# position and the full relative velocity, both in the ego body frame.
_REPORT_WIDTH = DETECTION_SLOT_WIDTH - 1


def detected_probabilities(
    occupancy: np.ndarray, miss_prob: float, false_alarm_prob: float
) -> np.ndarray:
    """Push occupancy probabilities through the detection channel.

    Args:
        occupancy: Probabilities that each slot really holds a vehicle, any shape.
        miss_prob: Rate at which a real vehicle fails to be reported.
        false_alarm_prob: Rate at which an empty slot is reported occupied.

    Returns:
        Probabilities of the same shape that the channel *reports* occupied.
    """
    return false_alarm_prob + (1.0 - miss_prob - false_alarm_prob) * np.asarray(
        occupancy, dtype=float
    )


def bernoulli_log_prob(probabilities: np.ndarray, observed: np.ndarray) -> float:
    """Log-likelihood of an observed boolean array under per-entry probabilities.

    Args:
        probabilities: Per-entry occupancy probabilities of any shape.
        observed: A boolean array of the same shape.

    Returns:
        The summed log-likelihood.
    """
    clipped = np.clip(
        np.asarray(probabilities, dtype=float), PROBABILITY_EPS, 1.0 - PROBABILITY_EPS
    )
    return float(np.sum(np.where(observed, np.log(clipped), np.log1p(-clipped))))


def gaussian_log_prob(deviation: np.ndarray, std: float) -> float:
    """Log-density of a zero-mean isotropic Gaussian at ``deviation``, summed over entries.

    Args:
        deviation: Residuals of any shape.
        std: Standard deviation shared by every entry.

    Returns:
        The summed log-density.
    """
    variance = float(std) ** 2
    flat = np.asarray(deviation, dtype=float).reshape(-1)
    return float(
        -0.5 * np.sum(flat**2) / variance - 0.5 * flat.size * np.log(2.0 * np.pi * variance)
    )


def cauchy_draw(shape: Tuple[int, int], scale: float) -> np.ndarray:
    """Zero-median Cauchy draws, by inverse transform.

    Args:
        shape: Shape of the array to draw.
        scale: Half-width at half-maximum, in the coordinates' own units.

    Returns:
        An array of the requested shape.
    """
    return scale * np.tan(np.pi * (np.random.random(shape) - 0.5))


def cauchy_log_prob(values: np.ndarray, scale: float) -> float:
    """Log-density of a zero-median Cauchy at ``values``, summed over entries.

    Args:
        values: Any array of coordinates.
        scale: Half-width at half-maximum, in the coordinates' own units.

    Returns:
        The summed log-density.
    """
    flat = np.asarray(values, dtype=float).reshape(-1)
    return float(-flat.size * np.log(np.pi * scale) - np.sum(np.log1p((flat / scale) ** 2)))


def pack_detections(reports: np.ndarray, max_detections: int) -> np.ndarray:
    """Order ``(D, 4)`` detection reports by range and pad them into fixed slots.

    Ordering is by *measured* range, which is the only range a sensor has, and it is also
    what the likelihood's rank association pairs against. Reports beyond ``max_detections``
    are dropped: the state carries that many slots, so a reading the model has no slot for is
    exactly what its false-alarm rate stands for.

    Args:
        reports: ``(D, 4)`` rows of ``[rel_x, rel_y, rel_vx, rel_vy]``.
        max_detections: Number of slots the packed reading carries.

    Returns:
        ``(max_detections, 5)`` rows of ``[detected, rel_x, rel_y, rel_vx, rel_vy]``, filled
        from slot zero, undetected slots left at zero.
    """
    rows = np.zeros((max_detections, DETECTION_SLOT_WIDTH), dtype=float)
    if len(reports) == 0:
        return rows
    order = np.argsort(np.linalg.norm(reports[:, :2], axis=1))[:max_detections]
    rows[: len(order), DETECTION_PRESENT] = 1.0
    rows[: len(order), DETECTION_REL_X : DETECTION_REL_X + _REPORT_WIDTH] = reports[order]
    return rows


def validate_detection_rates(miss_prob: float, false_alarm_prob: float) -> None:
    """Reject detection rates the composition ``q (1 - miss) + (1 - q) fa`` cannot use.

    Zero is the shipped setting: this world's detector misses nothing it can see and invents
    nothing, so hard presence agreement is the honest model of it. Above zero the pair models
    a lossy radar instead, in the sampler as well as in the density.

    Args:
        miss_prob: Rate at which a tracked vehicle fails to be reported.
        false_alarm_prob: Rate at which an empty slot is reported occupied.

    Raises:
        ValueError: If either rate is outside ``[0, 1)``, or they sum to 1 or more.
    """
    for name, value in (
        ("presence_miss_prob", miss_prob),
        ("presence_false_alarm_prob", false_alarm_prob),
    ):
        if not 0.0 <= value < 1.0:
            raise ValueError(
                f"{name} must be in [0, 1), got {value}. Zero is the shipped setting, since "
                f"this world's detection decision is deterministic; above zero models a "
                f"lossy radar."
            )
    if miss_prob + false_alarm_prob >= 1.0:
        raise ValueError(
            f"presence_miss_prob + presence_false_alarm_prob must stay below 1, got "
            f"{miss_prob} + {false_alarm_prob}. At or above 1 a tracked vehicle makes a slot "
            f"no more likely to be reported than an empty one, inverting the likelihood."
        )


__all__ = [
    "PROBABILITY_EPS",
    "bernoulli_log_prob",
    "cauchy_draw",
    "cauchy_log_prob",
    "detected_probabilities",
    "gaussian_log_prob",
    "pack_detections",
    "validate_detection_rates",
]
