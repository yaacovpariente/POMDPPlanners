# SPDX-License-Identifier: MIT

"""One record per evaluation episode, read from the episode traces.

The chart builder's aggregated mode plots what a run logged: one mean, with
its interval, per planner. The spread behind that mean -- is a mean of 8 two
episodes of 16 and two of 0, or four of 8? -- is only in the episodes. Each
trace's envelope carries the episode's returns, length and ending, and each
step carries the environment's ``step_info`` channels, so a record per
episode is read from there rather than from a second log.

Environment metrics such as ``task_completion_rate`` are reductions the
environment declares over those channels (any, sum, max ...). The reporting
site does not construct environments to learn the reductions, so a record
holds each channel's *total over the episode* under ``<channel> (sum)``; for
an indicator channel such as ``terminal_state`` that is 1 or 0, which is the
per-episode value of the metric built on it.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from POMDPPlanners.core.simulation.traces import ArtifactKind
from POMDPPlanners.reporting.store import EnvironmentView

#: The fields every record has, in the order a table or a CSV shows them.
BASE_FIELDS = ("episode", "return", "discounted_return", "steps", "ended")


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def episode_record(path: Path, episode: int) -> Optional[Dict[str, Any]]:
    """Read one trace into one record.

    Args:
        path: The trace file.
        episode: The episode index.

    Returns:
        The record, or ``None`` when the file is not a readable trace.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "schema_version" not in data:
        return None
    steps = data.get("steps") if isinstance(data.get("steps"), list) else []
    rewards = [_number(step.get("reward")) for step in steps if isinstance(step, dict)]
    total = _number(data.get("total_reward"))
    if total is None and rewards:
        total = float(sum(r for r in rewards if r is not None))
    terminal = data.get("reach_terminal_state")
    record: Dict[str, Any] = {
        "episode": episode,
        "return": total,
        "discounted_return": _number(data.get("discounted_return")),
        # Steps that took an action. The trace's num_steps also counts the
        # final bookkeeping step, which has none, so it runs one higher than
        # the logged average_actual_num_steps.
        "steps": (
            float(
                sum(
                    1 for step in steps if isinstance(step, dict) and step.get("action") is not None
                )
            )
            if steps
            else _number(data.get("num_steps"))
        ),
        "ended": (
            ("terminal" if terminal else "out of steps") if isinstance(terminal, bool) else None
        ),
    }
    channels: Dict[str, float] = {}
    for step in steps:
        info = step.get("info") if isinstance(step, dict) else None
        if not isinstance(info, dict):
            continue
        for name, value in info.items():
            number = _number(value)
            if number is not None:
                channels[name] = channels.get(name, 0.0) + number
    for name in sorted(channels):
        record[f"{name} (sum)"] = channels[name]
    return record


def episode_records(
    env: EnvironmentView, policies: Optional[Sequence[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Every episode of each planner on one environment, as records.

    Args:
        env: The environment within an evaluation run.
        policies: Planner names to read; all of them when ``None``.

    Returns:
        Planner name to its records, in episode order.
    """
    found: Dict[str, List[Dict[str, Any]]] = {}
    for policy in env.policies:
        if policies is not None and policy.name not in policies:
            continue
        records = []
        for index, artifacts in sorted(policy.episodes.items()):
            trace = next((a for a in artifacts if a.kind is ArtifactKind.TRACE), None)
            if trace is None:
                continue
            record = episode_record(trace.path, index)
            if record is not None:
                records.append(record)
        found[policy.name] = records
    return found
