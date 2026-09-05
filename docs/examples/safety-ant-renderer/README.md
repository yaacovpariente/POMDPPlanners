# Safety Ant GIF export

The renderer copies a cached Pillow background for each frame, avoiding repeated Matplotlib layout and drawing. The public API stays the same. Terminal states with a `None` reward now render as `0.0`.

The delivery run measured the same seed-42 four-state history with one warmup and five saved exports on ubuntu64, using one CPU thread and no GPU through the host gate. These timings include GIF encoding and saving. The terminal status banner sits above the plot so the step and action remain readable.

| Measurement | Matplotlib before | Pillow after |
| --- | ---: | ---: |
| Median export, seconds | 0.704784 | 0.193154 |
| File bytes | 137,807 | 177,883 |
| Frames | 4 | 4 |
| Dimensions | 1600×800 | 1600×800 |
| Frame duration, milliseconds | 1,250 | 1,250 |

The measured median fell by 72.6%. Timing will vary by machine and episode length. The scene keeps trajectory and speed panels, equal world units, velocity and force arrows, safety thresholds, rewards, and terminal status. Captions follow wide and tall world plots; the status header stays fully visible.

Before:

![Matplotlib export](before.gif)

After:

![Pillow export](after.gif)

The focused Safety Ant suite passed all 35 tests and focused pyright reported no errors. The regenerated golden GIF passed its hash comparison in the CI Docker base on ubuntu64. Black and whitespace checks pass. Two Claude reviews were completed; confirmed layout, golden, and formatting findings were fixed.

The original full-suite run had 5,749 passes, 37 failures, and 5 errors outside Safety Ant, including sandbox-denied sockets and downloads. That run does not establish a clean full-suite baseline.
