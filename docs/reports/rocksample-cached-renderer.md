# RockSample cached renderer evidence

The Pillow renderer cut the warmed six-frame export median from 0.3590 seconds
to 0.0500 seconds on `Yaacovs-MacBook-Air.local`. Both runs used the same fixed
history, one warm-up, five saved exports, and one CPU thread.

| Renderer | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median | GIF size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matplotlib baseline | 0.3523 s | 0.3590 s | 0.3532 s | 0.3840 s | 0.3664 s | 0.3590 s | 85,291 B |
| Cached Pillow | 0.0504 s | 0.0503 s | 0.0500 s | 0.0498 s | 0.0499 s | 0.0500 s | 110,253 B |

The fixed history has six states: east movement, a rock sensor check, a
successful sample, a failed sample after the rock changes, east movement
toward the exit, and a terminal state. It also includes a danger area and a
second bad rock.

- [Before GIF](artifacts/rocksample-renderer-before.gif)
- [After GIF](artifacts/rocksample-renderer-after.gif)

The baseline GIF SHA-256 is
`c882e6730605be778c9fb63fd1f610c92e970a670c049a8fd8d7a538a0572bc2`.
The Pillow GIF SHA-256 is
`1ccfb1f8ebf1bdc912f84317c9687e584324e78bd175a6473daa6c87d1a2d233`.
