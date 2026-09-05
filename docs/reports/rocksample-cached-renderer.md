# RockSample cached renderer evidence

The reviewed Pillow renderer cut the warmed six-frame saved-export median from
0.5857 seconds to 0.0736 seconds (87.4% lower) on `ubuntu64`. Both runs used the
same fixed history, one warm-up, five exports, and one CPU thread, with Pillow
12.0.0. Earlier local measurements were repeated after the clipping fix.

| Renderer | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median | GIF size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matplotlib baseline | 0.584140 s | 0.583967 s | 0.585696 s | 0.591760 s | 0.593965 s | 0.585696 s | 81,570 B |
| Cached Pillow | 0.073689 s | 0.073631 s | 0.073685 s | 0.073581 s | 0.073594 s | 0.073631 s | 110,253 B |

The fixed history has six states: east movement, a rock sensor check, a
successful sample, a failed sample after the rock changes, east movement
toward the exit, and a terminal state. It also includes a danger area and a
second bad rock.

- [Before GIF](artifacts/rocksample-renderer-before.gif)
- [After GIF](artifacts/rocksample-renderer-after.gif)

The baseline GIF SHA-256 is
`cdb8fae69d909a1fbc2352b278de016cb501a4647f2e1b3000c85503f33537e6`.
The Pillow GIF SHA-256 is
`1ccfb1f8ebf1bdc912f84317c9687e584324e78bd175a6473daa6c87d1a2d233`.
