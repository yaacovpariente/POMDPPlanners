# Push cached visualization artifacts

These fixed four-frame episodes compare the `develop` Matplotlib renderer with
the redesigned Pillow renderer: textured stone terrain, shaded blue rover,
wooden crate, golden goal and framed HUD. Each history includes a push, an
obstacle collision, rewards and terminal success. All pairs use seed 42,
1200×1000 frames and 1250 ms per frame.

| Variant | Before | After |
| --- | --- | --- |
| Discrete Push | [Matplotlib GIF](discrete_before.gif) | [Pillow GIF](discrete_after.gif) |
| Continuous Push | [Matplotlib GIF](continuous_before.gif) | [Pillow GIF](continuous_after.gif) |
| Continuous Push, discrete actions | [Matplotlib GIF](continuous_discrete_before.gif) | [Pillow GIF](continuous_discrete_after.gif) |

The benchmark used seed 42, one warm-up, then five timed exports on
`ubuntu64` with one logical thread, after review corrections. Both renderers used
the same Python environment and histories. Initial Mac measurements remain in
the task report as historical evidence.

| Variant | Baseline median | Pillow median | Baseline bytes | Pillow bytes |
| --- | ---: | ---: | ---: | ---: |
| Discrete Push | 0.539515 s | 0.126826 s | 101,747 | 2,475,202 |
| Continuous Push | 0.549183 s | 0.126635 s | 104,190 | 2,440,146 |
| Continuous Push, discrete actions | 0.552328 s | 0.126694 s | 101,739 | 2,445,033 |

The first discrete export in a fresh process, including texture and sprite
initialization, took 0.576440 s before and 0.273945 s after (one sample each).
The table reports warmed medians, not first-save times.

Textures increase GIF size; cached terrain and sprites still make these exports
more than four times faster than baseline. Final GIFs were decoded for visual
review. Both golden references passed in the CI Docker image.

The packaged transparent sprite sheet was generated with OpenAI's built-in
image tool, then cropped and resized by the renderer. The prompt requested a
top-down cobalt-blue four-wheel rover, a cross-braced wooden crate and a faceted
gold star in three equal cells, with true transparency and no text or floor.
The asset is loaded from the installed package; rendering needs no service call.
