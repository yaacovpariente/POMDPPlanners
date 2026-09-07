# CartPole recorded episode replay

The before GIFs come from the existing standalone diagnostic tool. Develop had
no CartPole package exporter. The after GIFs use the new package episode hook.
Both replay the same saved histories; no planner was rerun.

Each seed pair has 20 frames, 800×500 pixels, and 500 ms per frame. Seed 42
history SHA256 is `ea717c24f74d6eea861f41ab33aa535280393b9c43c5f3db3264ded5f3c9fcbe`;
seed 43 is `b11ba6fa177258707acbe4bdbfa3f448d0b0f05c70c3417408d8566ce5184cc3`.
The captured histories come from the existing environment visualization audit.

Measurements ran on ubuntu64, one CPU thread, through the shared execution
gate. Each version ran in a fresh process: the first complete save was the
warm-up, followed by five timed complete saves for seed 42. The first-save
values below are single samples, not distributions.

| Seed 42 | Standalone diagnostic | Package renderer |
| --- | ---: | ---: |
| Median warm save, seconds | 1.284816 | 0.319430 |
| Warm range, seconds | 1.235620–1.341575 | 0.318370–0.322115 |
| First save, seconds | 1.260997 | 0.340735 |
| GIF bytes | 264326 | 231263 |
| Seed 43 GIF bytes | 257382 | 241452 |

These are absolute export costs and a diagnostic-tool comparison, not a
speedup over an earlier package renderer.

The artwork supplies the lab and blue cart. Recorded position and angle draw
the cart and full 1 m pole with one physical scale. Action 0 points left and
action 1 right, independently of velocity. Pre-action truth, stored particle
support, and realised noisy observations/rewards have separate labels.
