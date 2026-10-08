# SPDX-License-Identifier: MIT

"""The configuration sweep covers every switch, and its hand-kept lists are current.

:mod:`~POMDPPlanners.tests.test_environments._env_config_variants` reads each
environment's constructor and adds one variant per configuration switch to
``ENV_BUILDERS``, so every conformance suite runs on every branch an
environment can be configured into. The sweep itself needs no upkeep. Five
small lists beside it do: the string switches a signature cannot show, the
hazards given to hazard-free maps, the companion arguments, the variants built
by hand and the exclusions. An entry that no longer names a real switch covers
nothing while still reading as coverage, so each list is checked here against
the constructors.
"""

import inspect
import typing
from typing import Dict, FrozenSet, List, Set, Tuple

import pytest

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.tests.test_environments._env_config_variants import (
    COMPANION_ARGUMENTS,
    CONFIG_VARIANT_EXCLUSIONS,
    EXTRA_SWITCH_VALUES,
    FRAMEWORK_OPTIONS,
    HAND_BUILT_VARIANTS,
    HAZARD_ARGUMENTS,
    HAZARD_SWITCHES,
    HAZARDS_VARIANT,
    ConfigVariant,
    VariantKey,
    base_id_of,
    config_variants,
    discover_switches,
)
from POMDPPlanners.tests.test_environments.test_env_api_conformance import (
    ENV_BUILDERS,
    HAND_WRITTEN_ENV_BUILDERS,
    EnvBuilder,
)

_BUILDER_BY_ID: Dict[str, EnvBuilder] = dict(ENV_BUILDERS)

# The pinned configurations: the hand-written entries that are not variants.
_PINNED_IDS: List[str] = [env_id for env_id, _ in HAND_WRITTEN_ENV_BUILDERS if "[" not in env_id]


def _pinned_env(base_id: str) -> Environment:
    return _BUILDER_BY_ID[base_id]()


def _all_variants() -> List[ConfigVariant]:
    return [
        variant
        for base_id in _PINNED_IDS
        for variant in config_variants(base_id, type(_pinned_env(base_id)))
    ]


def _variant_ids() -> List[str]:
    """Every registered id that is a variant of a pinned configuration."""
    return [env_id for env_id, _ in ENV_BUILDERS if base_id_of(env_id) != env_id]


def test_every_config_switch_has_a_registered_variant() -> None:
    """Each switch value of each environment is built, or excluded with a reason.

    Purpose: A switch value with no registered variant is a branch of the
        environment -- and of its batch path, its vectorized belief and its
        torch model -- that no conformance test builds. ``ENV_BUILDERS`` is
        built from the same sweep, so for a swept value this holds by
        construction; what can break it is the hand-kept part: a hand-built
        id that is not registered, or a value both excluded and built. The
        sweep finds a bool switch by its default, so a parameter annotated
        ``bool`` with any other default would be missed on both sides; the
        last assertion reads the annotations to catch that.

    Given: Every bool, enum and listed string switch of every pinned
        environment, found by reading its constructor.
    When: Each non-default value is looked up in ``ENV_BUILDERS`` and in
        ``CONFIG_VARIANT_EXCLUSIONS``, and each ``bool``-annotated parameter
        in the swept switches.
    Then: Each value is in exactly one list, and every ``bool``-annotated
        parameter is a swept switch.

    Test type: unit
    """
    for base_id in _PINNED_IDS:
        env_class = type(_pinned_env(base_id))
        swept = set(discover_switches(env_class))
        try:
            hints = typing.get_type_hints(env_class.__init__)
        except Exception:  # pylint: disable=broad-exception-caught
            hints = {}
        unswept = sorted(
            name
            for name, hint in hints.items()
            if name not in FRAMEWORK_OPTIONS
            and (hint is bool or bool in typing.get_args(hint))
            and name not in swept
        )
        assert not unswept, (
            f"{base_id}: constructor parameters annotated bool that the sweep does not "
            f"find, because their default is not a bool: {unswept}"
        )
    registered = set(_BUILDER_BY_ID)
    for variant in _all_variants():
        is_registered = variant.variant_id in registered
        is_excluded = variant.key in CONFIG_VARIANT_EXCLUSIONS
        assert is_registered != is_excluded, (
            f"{variant.base_id}: {variant.option}={variant.value_name} must be either "
            f"registered in ENV_BUILDERS as {variant.variant_id!r} or listed in "
            "CONFIG_VARIANT_EXCLUSIONS, not both and not neither"
        )


def test_variant_lists_name_real_variants() -> None:
    """No hand-kept list entry outlives the switch it was written for.

    Purpose: A renamed parameter or a removed enum member leaves its list
        entry behind. The sweep then builds the renamed variant without the
        companion argument or the exclusion it needed, and the stale entry
        still reads as if it applied.

    Given: The exclusions, the hand-built variants and the companion arguments.
    When: Each key is looked up among the variants the sweep finds.
    Then: Every key names one, and every hand-built id is in ``ENV_BUILDERS``.

    Test type: unit
    """
    real: Set[VariantKey] = {variant.key for variant in _all_variants()}
    for name, keys in (
        ("CONFIG_VARIANT_EXCLUSIONS", set(CONFIG_VARIANT_EXCLUSIONS)),
        ("HAND_BUILT_VARIANTS", set(HAND_BUILT_VARIANTS)),
        ("COMPANION_ARGUMENTS", set(COMPANION_ARGUMENTS)),
    ):
        stale = sorted(keys - real)
        assert not stale, f"{name} names variants the sweep no longer finds: {stale}"
    missing = sorted(set(HAND_BUILT_VARIANTS.values()) - set(_BUILDER_BY_ID))
    assert not missing, f"HAND_BUILT_VARIANTS names ids that are not in ENV_BUILDERS: {missing}"


def test_extra_switch_values_name_real_string_parameters() -> None:
    """Each listed string switch is a constructor parameter with a string default.

    Purpose: String switches are the one kind the sweep cannot find, so the
        list is the only thing that covers them. An entry for a parameter that
        was renamed, or turned into an enum, covers nothing.

    Given: ``EXTRA_SWITCH_VALUES``.
    When: Each ``(class, parameter)`` is looked up among the pinned
        environments' constructors.
    Then: The class is registered, the parameter exists with a ``str``
        default, and the default is not repeated among the listed values.

    Test type: unit
    """
    classes = {
        type(_pinned_env(base_id)).__name__: type(_pinned_env(base_id)) for base_id in _PINNED_IDS
    }
    for (class_name, option), values in EXTRA_SWITCH_VALUES.items():
        assert (
            class_name in classes
        ), f"EXTRA_SWITCH_VALUES names an unregistered class {class_name}"
        parameters = inspect.signature(classes[class_name].__init__).parameters
        assert option in parameters, f"{class_name} has no constructor parameter {option!r}"
        default = parameters[option].default
        assert isinstance(default, str), (
            f"{class_name}.{option} defaults to {default!r}, not a string; the sweep finds "
            "bool and enum switches by itself"
        )
        assert default not in values, f"{class_name}.{option}: the default {default!r} is listed"


def test_hazard_arguments_apply_to_environments_with_a_hazard_switch() -> None:
    """Hazards are given only where a hazard switch can use them.

    Purpose: ``HAZARD_ARGUMENTS`` exists so a hazard switch selects a branch
        that something enters. An entry for an environment with no such switch
        builds a ``[hazards]`` variant and changes nothing else; an entry whose
        arguments the constructor does not take fails every variant at once.

    Given: ``HAZARD_ARGUMENTS``.
    When: Each environment's constructor is read.
    Then: The environment is pinned, takes every hazard argument, and its
        ``[hazards]`` variant is registered.

    Test type: unit
    """
    for base_id, arguments in HAZARD_ARGUMENTS.items():
        assert base_id in _PINNED_IDS, f"HAZARD_ARGUMENTS names an unregistered id {base_id}"
        parameters = inspect.signature(type(_pinned_env(base_id)).__init__).parameters
        unknown = sorted(set(arguments()) - set(parameters))
        assert not unknown, f"{base_id} takes no constructor arguments named {unknown}"
        assert f"{base_id}[{HAZARDS_VARIANT}]" in _BUILDER_BY_ID


# Groups of registered configurations that share one ``config_id`` although
# they differ. Each is a real collision: result caches and tables key on
# ``config_id``, so one configuration can be served the other's cached episodes.
#
# Continuous light-dark keeps no public attribute for its reward model type,
# and ``config_id`` hashes public attributes only. The constant-penalty and
# zero-mean-shock reward models hold the same fields, so with the hazard slot
# off -- the only way the shock model can be built -- the two share an id.
# Not fixed here: a public attribute would change the id of every continuous
# light-dark configuration and invalidate its cached results.
KNOWN_CONFIG_ID_COLLISIONS: Set[FrozenSet[str]] = {
    frozenset(
        {
            "ContinuousLightDarkPOMDP[is_obstacle_hit_terminal=False]",
            "ContinuousLightDarkPOMDP[reward_model_type=ZERO_MEAN_HAZARD_SHOCK]",
        }
    ),
    frozenset(
        {
            "ContinuousLightDarkPOMDPDiscreteActions[is_obstacle_hit_terminal=False]",
            "ContinuousLightDarkPOMDPDiscreteActions[reward_model_type=ZERO_MEAN_HAZARD_SHOCK]",
        }
    ),
}


def test_registered_configurations_have_distinct_config_ids() -> None:
    """No two registered configurations share a ``config_id``, beyond the recorded ones.

    Purpose: ``config_id`` is the key of the experiment cache and of every
        results table. Two configurations with one id are one configuration to
        the cache: the second to run is handed the first one's episodes. A
        switch whose value never reaches a public attribute collides this way,
        and nothing else notices, because the environments still behave
        differently and still compare unequal.

    Given: Every registered configuration and its ``config_id``.
    When: The ids are grouped.
    Then: The groups of more than one configuration are exactly the recorded
        collisions -- a new one fails, and so does a recorded one that has
        been fixed.

    Test type: unit
    """
    by_config_id: Dict[str, Set[str]] = {}
    for env_id, builder in ENV_BUILDERS:
        by_config_id.setdefault(builder().config_id, set()).add(env_id)
    collisions = {frozenset(ids) for ids in by_config_id.values() if len(ids) > 1}
    new = sorted(sorted(group) for group in collisions - KNOWN_CONFIG_ID_COLLISIONS)
    assert not new, f"registered configurations that share a config_id: {new}"
    fixed = sorted(sorted(group) for group in KNOWN_CONFIG_ID_COLLISIONS - collisions)
    assert not fixed, f"KNOWN_CONFIG_ID_COLLISIONS records collisions that are gone: {fixed}"


def _hazard_switch_owners() -> List[Tuple[str, List[str]]]:
    owners = []
    for base_id in _PINNED_IDS:
        switches = discover_switches(type(_pinned_env(base_id)))
        hazard_switches = sorted(set(switches) & HAZARD_SWITCHES)
        if hazard_switches:
            owners.append((base_id, hazard_switches))
    return owners


@pytest.mark.parametrize("env_id", _variant_ids())
def test_variant_differs_from_its_pinned_configuration(env_id: str) -> None:
    """A variant is a different configuration from the one it was derived from.

    Purpose: A variant only adds coverage if its switch reached the
        constructor. If a builder drops the override, or a pinned argument
        shadows it, the variant is the pinned configuration under another id
        and every test on it repeats a test that already ran.

    Given: A registered variant and its pinned configuration.
    When: Their ``config_id`` values are compared.
    Then: They differ.

    Test type: unit
    """
    variant = _BUILDER_BY_ID[env_id]()
    pinned = _pinned_env(base_id_of(env_id))
    assert type(variant) is type(pinned)
    assert variant.config_id != pinned.config_id, (
        f"{env_id} has the same config_id as {base_id_of(env_id)}: the changed argument "
        "did not reach the environment"
    )


@pytest.mark.parametrize(
    "base_id,hazard_switches",
    [pytest.param(base_id, switches, id=base_id) for base_id, switches in _hazard_switch_owners()],
)
def test_hazard_switch_variants_are_built_on_a_map_with_a_hazard(
    base_id: str, hazard_switches: List[str]
) -> None:
    """An environment with a hazard switch is swept on a map that has a hazard.

    Purpose: On a map with no hazard, the reward model and hazard-hit
        termination select branches nothing enters, and their variants pass
        every check by never running the code they were built to reach. The
        pinned Push, continuous Push, RockSample and PacMan maps hold none.

    Given: An environment with at least one hazard switch.
    When: Its pinned configuration and its hazard arguments are read.
    Then: The pinned configuration already places a hazard, or
        ``HAZARD_ARGUMENTS`` gives it one.

    Test type: unit
    """
    if base_id in HAZARD_ARGUMENTS:
        return
    pinned = _pinned_env(base_id)
    hazards = [
        getattr(pinned, name)
        for name in ("obstacles", "dangerous_areas")
        if getattr(pinned, name, None) is not None
    ]
    assert any(len(hazard) > 0 for hazard in hazards), (
        f"{base_id} has hazard switches {hazard_switches} but its pinned configuration "
        "places no obstacle and no dangerous area; add it to HAZARD_ARGUMENTS"
    )
