# Push cached visualization artifacts

These fixed four-frame episodes show the renderer before and after the direct
Pillow change. Each history includes a push, an obstacle collision, reward
updates, and terminal success.

| Variant | Before | After |
| --- | --- | --- |
| Discrete Push | [Matplotlib GIF](discrete_before.gif) | [Pillow GIF](discrete_after.gif) |
| Continuous Push | [Matplotlib GIF](continuous_before.gif) | [Pillow GIF](continuous_after.gif) |

The benchmark used seed 42, one warm-up, then five timed exports on
`ubuntu64` with one logical thread, after review corrections. Both renderers used
the same Python environment and histories. Initial Mac measurements remain in
the task report as historical evidence.

| Variant | Baseline median | Pillow median | Baseline bytes | Pillow bytes |
| --- | ---: | ---: | ---: | ---: |
| Discrete Push | 0.547151 s | 0.083899 s | 101,747 | 120,559 |
| Continuous Push | 0.555873 s | 0.083978 s | 104,190 | 131,161 |
