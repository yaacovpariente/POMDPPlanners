# SPDX-License-Identifier: MIT

"""One environment variant per configuration switch.

The conformance suites used to run each environment on one pinned
configuration. An environment's logic branches on its switches -- a reward
model, an observation model, whether a hazard ends the episode -- and each
branch is written again in the batch path, in the vectorized belief and in the
torch model. A branch the suites never build is a branch where those copies can
disagree unseen.

This module finds every switch by reading the constructor, and nothing else:

* a parameter whose default is a ``bool`` -- the variant flips it;
* a parameter whose default or annotation is an :class:`enum.Enum` -- one
  variant per other member;
* a string parameter with a fixed set of values, which the constructor's
  signature cannot show -- listed by hand in :data:`EXTRA_SWITCH_VALUES`.

Each variant changes one switch from the pinned default and leaves the rest
alone, so a failure names the switch that caused it. A new switch is swept the
moment it is added to a constructor; it needs no registration here.

Several pinned configurations place no hazard on the map. There a hazard switch
-- the reward model, hazard-hit termination -- selects a branch nothing ever
enters. :data:`HAZARD_ARGUMENTS` gives those environments a hazard: every
variant on a hazard switch is built with it, and so is one extra ``[hazards]``
variant that keeps every switch at its default.

A variant the constructor rejects on its own gets the one extra argument it
needs from :data:`COMPANION_ARGUMENTS`. A variant that should not be built goes
on :data:`CONFIG_VARIANT_EXCLUSIONS` with its reason. A variant an existing hand-written registry entry already
builds (with extra arguments the sweep cannot guess) goes on
:data:`HAND_BUILT_VARIANTS`, so the two do not run twice.
"""

import enum
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.tests.test_environments._vectorized_config_contract import (
    discover_enum_params,
)

# (base env id, constructor parameter, value name).
VariantKey = Tuple[str, str, str]

# Constructor parameters that never change transitions, observations or rewards.
FRAMEWORK_OPTIONS = frozenset({"name", "output_dir", "debug", "use_queue_logger"})

# String switches: the constructor takes a plain ``str`` and validates it
# against a fixed set, so the signature does not show the alternatives.
# Keyed by (environment class name, parameter); the values exclude the default.
EXTRA_SWITCH_VALUES: Dict[Tuple[str, str], Tuple[Any, ...]] = {
    ("PacManPOMDP", "ghost_coordination"): ("coordinated", "mixed"),
}

# Variants a hand-written ENV_BUILDERS entry already builds, and that entry's
# id. Each needs an argument besides the switch, which the sweep cannot guess.
HAND_BUILT_VARIANTS: Dict[VariantKey, str] = {
    ("ChicheckInvadersPOMDP", "observation_mode", "FULL"): (
        "ChicheckInvadersPOMDP[fully_observable]"
    ),
    # Built with a wider range noise, so the truncation actually bites.
    ("OccupancyGridMappingPOMDP", "range_noise_model", "TRUNCATED_NORMAL"): (
        "OccupancyGridMappingPOMDP[truncated_normal]"
    ),
}

# Switches whose branches only run when the map holds a hazard.
HAZARD_SWITCHES = frozenset(
    {"reward_model_type", "is_dangerous_area_hit_terminal", "is_obstacle_hit_terminal"}
)

# Id suffix of the extra variant that adds the hazards and flips no switch.
HAZARDS_VARIANT = "hazards"


def _push_hazards() -> Dict[str, Any]:
    return {
        "obstacles": [(2.0, 2.0)],
        "obstacle_hit_probability": 0.5,
        "dangerous_areas": [(5.0, 5.0)],
        "dangerous_area_radius": 2.0,
        "dangerous_area_hit_probability": 0.5,
    }


def _continuous_push_hazards() -> Dict[str, Any]:
    return {**_push_hazards(), "obstacles": [(2.0, 2.0, 1.0)]}


# Hazards for the environments whose pinned configuration has none, as
# constructor arguments. Each sits where a short random rollout from the
# initial state enters it, and is hit with probability one half where the
# environment has such a probability, so both outcomes of the roll occur.
# A function per environment, so no two builds share a list.
HAZARD_ARGUMENTS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "ContinuousPushPOMDP": _continuous_push_hazards,
    "ContinuousPushPOMDPDiscreteActions": _continuous_push_hazards,
    "PacManPOMDP": lambda: {"dangerous_areas": {(1, 1)}},
    "PushPOMDP": _push_hazards,
    "RockSamplePOMDP": lambda: {
        "dangerous_areas": [(1, 1)],
        "dangerous_area_hit_probability": 0.5,
    },
}

# Variants the constructor rejects unless a second argument changes with the
# switch, and that argument. The variant keeps its swept id.
_NO_HIT_TERMINAL = {"is_obstacle_hit_terminal": False}
COMPANION_ARGUMENTS: Dict[VariantKey, Dict[str, Any]] = {
    # The shock reward has no hit probability to couple termination to, and
    # obstacle-hit termination is this environment's default.
    ("ContinuousLightDarkPOMDP", "reward_model_type", "ZERO_MEAN_HAZARD_SHOCK"): _NO_HIT_TERMINAL,
    (
        "ContinuousLightDarkPOMDPDiscreteActions",
        "reward_model_type",
        "ZERO_MEAN_HAZARD_SHOCK",
    ): _NO_HIT_TERMINAL,
}

# Variants the sweep finds and deliberately does not build, and why. The
# coverage test fails on an entry that no longer names a real variant.
CONFIG_VARIANT_EXCLUSIONS: Dict[VariantKey, str] = {}


@dataclass(frozen=True)
class ConfigVariant:
    """One switch of one environment, set to one non-default value.

    Attributes:
        base_id: ``ENV_BUILDERS`` id of the environment's pinned configuration.
        option: Constructor parameter the variant changes.
        value: The value it is set to.
    """

    base_id: str
    option: str
    value: Any

    @property
    def value_name(self) -> str:
        """The value as it appears in the variant id."""
        return self.value.name if isinstance(self.value, enum.Enum) else str(self.value)

    @property
    def key(self) -> VariantKey:
        """The key the exclusion and hand-built lists use."""
        return (self.base_id, self.option, self.value_name)

    @property
    def variant_id(self) -> str:
        """Test id: the hand-built entry's id when there is one."""
        return HAND_BUILT_VARIANTS.get(self.key, f"{self.base_id}[{self.option}={self.value_name}]")


def base_id_of(env_id: str) -> str:
    """The id of the pinned configuration a variant was derived from."""
    return env_id.split("[", 1)[0]


def discover_switches(env_class: Type[Any]) -> Dict[str, Tuple[Any, ...]]:
    """Every configuration switch of ``env_class`` and its non-default values.

    Args:
        env_class: The environment class whose constructor is read.

    Returns:
        ``{parameter: alternatives}`` in constructor order. A parameter with no
        alternative to its default is left out.
    """
    enum_params = discover_enum_params(env_class)
    switches: Dict[str, Tuple[Any, ...]] = {}
    for name, parameter in inspect.signature(env_class.__init__).parameters.items():
        if name == "self" or name in FRAMEWORK_OPTIONS:
            continue
        default = parameter.default
        alternatives: Tuple[Any, ...] = ()
        if isinstance(default, bool):
            alternatives = (not default,)
        elif isinstance(default, enum.Enum):
            alternatives = tuple(member for member in type(default) if member is not default)
        elif name in enum_params:
            # Enum-annotated, with a default that is not a member (None).
            alternatives = tuple(enum_params[name])
        alternatives += EXTRA_SWITCH_VALUES.get((env_class.__name__, name), ())
        if alternatives:
            switches[name] = alternatives
    return switches


def config_variants(base_id: str, env_class: Type[Any]) -> List[ConfigVariant]:
    """Every variant of one environment, excluded ones included."""
    return [
        ConfigVariant(base_id, option, value)
        for option, values in discover_switches(env_class).items()
        for value in values
    ]


def config_variant_builders(
    base_builders: Sequence[Tuple[str, Callable[..., Environment]]],
) -> List[Tuple[str, Callable[[], Environment]]]:
    """Builders for the variants the sweep adds to a registry.

    Args:
        base_builders: ``(base id, builder)`` for each environment's pinned
            configuration. The builder takes constructor overrides as keyword
            arguments.

    Returns:
        ``(variant id, builder)`` for every variant that is neither excluded
        nor already built by hand. Each pinned configuration is built once
        here, to read its class. No variant is: one the constructor rejects
        fails in the tests that build it.
    """
    builders: List[Tuple[str, Callable[[], Environment]]] = []
    for base_id, builder in base_builders:
        if base_id in HAZARD_ARGUMENTS:
            builders.append(
                (f"{base_id}[{HAZARDS_VARIANT}]", _build_with(builder, base_id, None, {}))
            )
        for variant in config_variants(base_id, type(builder())):
            if variant.key in CONFIG_VARIANT_EXCLUSIONS or variant.key in HAND_BUILT_VARIANTS:
                continue
            overrides = {**COMPANION_ARGUMENTS.get(variant.key, {}), variant.option: variant.value}
            builders.append(
                (variant.variant_id, _build_with(builder, base_id, variant.option, overrides))
            )
    return builders


def _build_with(
    builder: Callable[..., Environment],
    base_id: str,
    option: Optional[str],
    overrides: Dict[str, Any],
) -> Callable[[], Environment]:
    """A builder for one variant; ``option`` is ``None`` for the hazards-only variant."""

    def build() -> Environment:
        hazards: Dict[str, Any] = {}
        if base_id in HAZARD_ARGUMENTS and (option is None or option in HAZARD_SWITCHES):
            hazards = HAZARD_ARGUMENTS[base_id]()
        return builder(**{**hazards, **overrides})

    return build
