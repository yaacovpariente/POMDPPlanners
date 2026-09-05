"""Compare six-state LaserTag episodes after one warm-up per renderer."""
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from PIL import Image
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_visualizer import ContinuousLaserTagVisualizer

ROOT = Path(__file__).resolve().parent


class Belief:
    def to_unique_support_distribution(self):
        return SimpleNamespace(values=[
            np.array([2.3, 2.2, 7.2, 4.1, 0.]),
            np.array([2.7, 2.5, 7.6, 4.4, 0.]),
        ])


def fixed_history():
    states = [
        [1.25, 1.5, 8.2, 4.5, 0.], [1.8, 1.5, 7.7, 4.2, 0.],
        [2.1, 2.0, 7., 4., 0.], [3., 2., 3.8, 2., 0.],
        [3., 2., 3.8, 2., 1.], [3., 2., 3.8, 2., 1.],
    ]
    actions = [np.array([1., 0., 0.]), "up", "tag", np.array([0., 0., 1.]), None, None]
    return [StepData(np.array(s), a, None, None, None, Belief()) for s, a in zip(states, actions)]


def make_visualizer(cls=ContinuousLaserTagVisualizer):
    return cls(np.array([11., 7.]), np.array([[5., 3., .5, 1.]]), .3, .3, [(3., 5.)], 1.)


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("baseline_renderer", ROOT / "baseline_renderer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    history = fixed_history()
    result = {}
    for name, cls in (("before", module.ContinuousLaserTagVisualizer), ("after", ContinuousLaserTagVisualizer)):
        visualizer = make_visualizer(cls)
        path = ROOT / f"{name}.gif"
        visualizer.create_visualization(history, path)
        times, sizes, hashes = [], [], []
        for _ in range(5):
            start = perf_counter()
            visualizer.create_visualization(history, path)
            times.append(perf_counter() - start)
            sizes.append(path.stat().st_size)
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        with Image.open(path) as image:
            frames = image.n_frames
            durations = []
            for index in range(frames):
                image.seek(index)
                durations.append(image.info['duration'])
            image.seek(2)
            image.convert("RGB").save(ROOT / f"{name}.png")
            dimensions = image.size
        result[name] = dict(seconds=times, median_seconds=median(times), bytes=sizes,
                            sha256=hashes, frames=frames, dimensions=dimensions,
                            durations_ms=durations)
    (ROOT / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
