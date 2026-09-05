"""One fresh-process LaserTag benchmark arm and actual decoded GIF preview."""
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np
from PIL import Image
import continuous_inputs as continuous
import discrete_inputs as discrete

variant, arm = sys.argv[1:3]
root = Path(__file__).resolve().parent
np.random.seed(42)
if variant == "discrete":
    cls = discrete.load_baseline() if arm == "before" else discrete.LaserTagVisualizer
    make = discrete.make_visualizer
    history = discrete.fixed_history()
else:
    cls = continuous.ContinuousLaserTagVisualizer
    if arm == "before":
        spec = importlib.util.spec_from_file_location("baseline",root/"baseline_renderer.py")
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls=module.ContinuousLaserTagVisualizer
    make=continuous.make_visualizer
    history=continuous.fixed_history()
    if variant == "continuous_discrete":
        history=[step._replace(action=action) for step,action in zip(history,["right","up","tag","tag",None,None])]
path=root/f"{variant}-{arm}.gif"
start=perf_counter()
visualizer=make(cls)
visualizer.create_visualization(history,path)
first=perf_counter()-start
times=[]
for _ in range(5):
    start=perf_counter()
    visualizer.create_visualization(history,path)
    times.append(perf_counter()-start)
with Image.open(path) as gif:
    durations=[]
    for index in range(gif.n_frames):
        gif.seek(index)
        durations.append(gif.info["duration"])
    gif.seek(0)
    gif.convert("RGB").save(root/f"{variant}-{arm}-decoded.png")
    frames,dimensions=gif.n_frames,gif.size
result=dict(variant=variant,arm=arm,seed=42,first_save_seconds=first,seconds=times,median_seconds=median(times),bytes=path.stat().st_size,frames=frames,dimensions=dimensions,durations_ms=durations,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(root/f"{variant}-{arm}.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result))
