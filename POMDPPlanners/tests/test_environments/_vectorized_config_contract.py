# SPDX-License-Identifier: MIT

"""Reusable contract engine for vectorized-model config coverage tests.

A vectorized generative model is a hand-written torch duplicate of a *subset*
of its scalar environment's configuration space. Every such model guards the
configs it does not implement with :class:`NotImplementedError`. The risk is
silent drift: the scalar env grows a new enum member (a new opponent policy,
reward model, ghost-coordination mode, ...) and the vectorized model neither
implements it nor is forced to declare it out of scope, so a downstream planner
only discovers the gap at runtime.

This module provides :func:`assert_config_contract`, which auto-discovers every
``enum.Enum``-typed constructor parameter of an environment and, one axis at a
time, asserts that the vectorized model built from each enum value **either**:

* constructs successfully (the config is supported), **or**
* raises :class:`NotImplementedError` **and** the ``(param, member)`` pair is on
  a small, reviewed ``expected_declines`` allowlist.

The forcing function is the allowlist: a newly added enum member is swept
automatically, so it must be either implemented or consciously added to the
allowlist — it can never slip through unnoticed. The allowlist is also kept
honest in the other direction: a member on the allowlist that the model now
builds fails the contract, so implementing a previously declined config forces
its removal from the allowlist.

Scope note: this contract verifies *coverage and triage*, not *numeric
correctness*. That a supported config builds does not prove its kernels match
the scalar env — each env's dedicated parity test owns that check.
"""

import enum
import inspect
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Set, Tuple, Type

# A single swept axis value, identified by (constructor-param name, enum-member
# name). Used as the allowlist key so declines read as e.g. ("reward_model_type",
# "DISTANCE_DECAYED_HAZARD_PENALTY").
DeclineKey = Tuple[str, str]


@dataclass
class VectorizedContractSpec:
    """Declarative spec pinning one environment's vectorized-model contract.

    Attributes:
        name: Human-readable label used in assertion messages and test ids.
        env_class: The scalar environment class whose ``__init__`` is
            introspected for ``Enum``-typed parameters.
        build_env: Callable that builds a valid environment from keyword
            overrides (the engine passes exactly one ``{param: member}`` axis
            override per call).
        build_model: Callable that builds the vectorized model from an
            environment instance.
        expected_declines: The reviewed allowlist of ``(param, member)`` pairs
            the vectorized model is expected to reject with
            ``NotImplementedError``.
    """

    name: str
    env_class: Type[Any]
    build_env: Callable[..., Any]
    build_model: Callable[[Any], Any]
    expected_declines: Set[DeclineKey] = field(default_factory=set)


def _resolve_annotation(annotation: Any) -> Any:
    # Unwrap Optional[Enum] / Union[Enum, None] to the underlying Enum class so
    # `opponent_policy: Optional[OpponentPolicy]` is still discovered.
    if isinstance(annotation, type):
        return annotation
    if typing.get_origin(annotation) is typing.Union:
        enum_args = [
            arg
            for arg in typing.get_args(annotation)
            if isinstance(arg, type) and issubclass(arg, enum.Enum)
        ]
        if len(enum_args) == 1:
            return enum_args[0]
    return annotation


def discover_enum_params(env_class: Type[Any]) -> Dict[str, Type[enum.Enum]]:
    """Map each ``Enum``-typed ``__init__`` parameter to its ``Enum`` class.

    Args:
        env_class: The environment class to introspect.

    Returns:
        An ordered mapping ``{param_name: enum_class}`` for every constructor
        parameter annotated (directly or as ``Optional``) with an ``enum.Enum``
        subclass. Empty if the environment exposes no enum-typed config axis.
    """
    try:
        hints = typing.get_type_hints(env_class.__init__)
    except Exception:  # pylint: disable=broad-exception-caught
        # Fall back to raw annotations when forward refs cannot be resolved.
        hints = {
            name: param.annotation
            for name, param in inspect.signature(env_class.__init__).parameters.items()
            if param.annotation is not inspect.Parameter.empty
        }
    discovered: Dict[str, Type[enum.Enum]] = {}
    for name, annotation in hints.items():
        resolved = _resolve_annotation(annotation)
        if isinstance(resolved, type) and issubclass(resolved, enum.Enum):
            discovered[name] = resolved
    return discovered


def assert_config_contract(spec: VectorizedContractSpec) -> Dict[str, int]:
    """Assert the vectorized model supports-or-declines every enum config axis.

    For each ``Enum``-typed constructor parameter of ``spec.env_class`` and each
    of its members, builds the environment with that single override and then
    the vectorized model, asserting the model either constructs or raises
    ``NotImplementedError`` for an allowlisted ``(param, member)`` pair.

    Args:
        spec: The environment's contract specification.

    Returns:
        A summary ``{"supported": n, "declined": m}`` count for reporting.

    Raises:
        AssertionError: If a config is declined without being allowlisted, if an
            allowlisted config unexpectedly builds, or if the environment
            exposes no enum-typed config axis to sweep.
    """
    enum_params = discover_enum_params(spec.env_class)
    assert enum_params, (
        f"{spec.name}: no Enum-typed constructor parameters were discovered; "
        "remove this env from the contract or check its type annotations."
    )
    supported, declined = 0, 0
    for param_name, enum_cls in enum_params.items():
        for member in enum_cls:
            supported, declined = _check_member(spec, param_name, member, supported, declined)
    return {"supported": supported, "declined": declined}


def _check_member(
    spec: VectorizedContractSpec,
    param_name: str,
    member: enum.Enum,
    supported: int,
    declined: int,
) -> Tuple[int, int]:
    key: DeclineKey = (param_name, member.name)
    try:
        env = spec.build_env(**{param_name: member})
    except (ValueError, NotImplementedError):
        # The scalar env itself rejects this value, so it is not a config the
        # vectorized model is ever asked to model — nothing to check.
        return supported, declined
    try:
        model = spec.build_model(env)
    except NotImplementedError:
        assert key in spec.expected_declines, (
            f"{spec.name}: {param_name}={member.name} is declined by the vectorized model "
            f"but is not in expected_declines. Implement it, or add it to the allowlist "
            f"as a reviewed out-of-scope decision."
        )
        return supported, declined + 1
    assert key not in spec.expected_declines, (
        f"{spec.name}: {param_name}={member.name} is listed in expected_declines but the "
        f"vectorized model built successfully — remove it from the allowlist."
    )
    assert model is not None
    return supported + 1, declined
