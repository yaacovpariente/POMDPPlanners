"""Compare fixed four-state discrete LaserTag GIF saves after one warm-up."""

import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from PIL import Image

from POMDPPlanners.core.simulation import StepData
import POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_visualizer
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_visualizer import LaserTagVisualizer

ROOT = Path(__file__).resolve().parent


class Belief:
    def to_unique_support_distribution(self):
        return SimpleNamespace(
            values=[np.array([1, 1, 5, 3, 0]), np.array([2, 1, 4, 3, 0])],
            probs=np.array([0.6, 0.4]),
        )


def fixed_history():
    states = ([1, 1, 5, 3, 0], [2, 1, 4, 3, 0], [2, 2, 2, 3, 0], [2, 2, 2, 2, 1])
    actions = (1, 2, 4, None)
    return [
        StepData(np.asarray(state), action, None, None, None, Belief())
        for state, action in zip(states, actions)
    ]


def load_baseline():
    continuous_spec = importlib.util.spec_from_file_location(
        "POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_visualizer",
        ROOT / "baseline_renderer.py",
    )
    continuous = importlib.util.module_from_spec(continuous_spec)
    current = sys.modules[continuous_spec.name]
    sys.modules[continuous_spec.name] = continuous
    continuous_spec.loader.exec_module(continuous)
    try:
        spec = importlib.util.spec_from_file_location("baseline_discrete", ROOT / "baseline_discrete.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.LaserTagVisualizer
    finally:
        sys.modules[continuous_spec.name] = current


def make_visualizer(cls):
    return cls((7, 5), {(3, 2)}, [(1, 3)], 0.8)


if __name__ == "__main__":
    result = {}
    for name, cls in (("before", load_baseline()), ("after", LaserTagVisualizer)):
        visualizer = make_visualizer(cls)
        path = ROOT / f"discrete-{name}.gif"
        visualizer.create_visualization(fixed_history(), path)
        times, sizes, hashes = [], [], []
        for _ in range(5):
            start = perf_counter()
            visualizer.create_visualization(fixed_history(), path)
            times.append(perf_counter() - start)
            sizes.append(path.stat().st_size)
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        with Image.open(path) as image:
            frames, dimensions = image.n_frames, image.size
            durations = []
            for index in range(image.n_frames):
                image.seek(index)
                durations.append(image.info["duration"])
            image.seek(2)
            image.convert("RGB").save(ROOT / f"discrete-{name}.png")
        result[name] = {
            "seconds": times,
            "median_seconds": median(times),
            "bytes": sizes,
            "sha256": hashes,
            "frames": frames,
            "dimensions": dimensions,
            "durations_ms": durations,
        }
    (ROOT / "benchmark-discrete.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
