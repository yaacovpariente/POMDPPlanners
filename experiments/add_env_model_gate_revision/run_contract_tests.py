# SPDX-License-Identifier: MIT
"""Run the LightDark contract suite and save its machine-readable verdict."""

import json
import subprocess
import sys
import time
from pathlib import Path


OUT = Path("results/add-env-model-gate-revision")
TEST = "experiments/add_env_model_gate_revision/test_contract_light_dark.py"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    junit = OUT / "contract-junit.xml"
    started = time.time()
    command = [sys.executable, "-m", "pytest", TEST, "-q", f"--junitxml={junit}"]
    completed = subprocess.run(command, check=False)
    artifact = {
        "pass": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
        "test_file": TEST,
        "junit_xml": str(junit),
        "wall_seconds": time.time() - started,
        "note": (
            "LightDark batch parity over the truth, doubled-noise and learned transitions. "
            "Of the substituted-transition helpers, assert_transition_is_used_everywhere and "
            "assert_observation_matches_reference run on the wired model with real data; "
            "assert_carried_channels_preserved runs with an empty carried-index list, because "
            "this world has no carried channel, so it asserts nothing about values."
        ),
    }
    (OUT / "contract.json").write_text(json.dumps(artifact, indent=2))
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
