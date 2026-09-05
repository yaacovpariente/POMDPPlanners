# Push cached visualization artifacts

These fixed four-frame episodes show the renderer before and after the direct
Pillow change. Each history includes a push, an obstacle collision, reward
updates, and terminal success.

| Variant | Before | After |
| --- | --- | --- |
| Discrete Push | [Matplotlib GIF](discrete_before.gif) | [Pillow GIF](discrete_after.gif) |
| Continuous Push | [Matplotlib GIF](continuous_before.gif) | [Pillow GIF](continuous_after.gif) |

The benchmark used seed 42, one warm-up, then five timed exports on
`Yaacovs-MacBook-Air.local` with one logical thread.

| Variant | Baseline median | Pillow median | Baseline bytes | Pillow bytes |
| --- | ---: | ---: | ---: | ---: |
| Discrete Push | 0.333598 s | 0.070443 s | 102,525 | 120,559 |
| Continuous Push | 0.341632 s | 0.071982 s | 101,207 | 131,300 |
