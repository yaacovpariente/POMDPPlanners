# MountainCar saved-history replays

The new episode hook writes `agent_path_<episode>.gif`. Before this change,
MountainCar's package hook did nothing. The before GIFs here come from the
standalone `create_diagnostic_gif` audit tool, not an old package exporter.

Each pair replays the same saved seed-42 or seed-43 history. All four GIFs are
800 × 500 pixels, with 20 frames at 500 ms each. No planner was run. Rows show
the true pre-action state, selected action, and recorded noisy next-state
observation. Gray dots show binned belief position mass; brightness is relative
to the largest bin. Gaussian beliefs instead show the mean and two standard
deviations, clipped to position bounds.

The vista, rock material, and blue car use packaged generated artwork. Code
places the track at `h(x) = 0.45 sin(3x) + 0.55`, rotates the car to the displayed
tangent, and places the goal at the environment's configured goal position.
Backgrounds, fonts, car rotations, and the GIF palette are cached.

## Complete saved-export measurements

Measured on ubuntu64 through its shared host gate, one CPU thread, Python
3.12.3 and Pillow 12.0.0. Each warmed median uses five complete saves after one
untimed warm-up. First saves are single samples from fresh processes, excluding
module imports and history loading. The diagnostic comparison is tooling
context; it is not a speedup over a former package renderer.

| Seed | Exporter | First save (s) | Warm median (s) | Bytes |
| --- | --- | ---: | ---: | ---: |
| 42 | Before: diagnostic | 1.675589 | 1.556353 | 230708 |
| 42 | After: package renderer | 0.294821 | 0.135659 | 1242546 |
| 43 | Before: diagnostic | 1.559594 | 1.544462 | 217164 |
| 43 | After: package renderer | 0.297641 | 0.135715 | 1264965 |

The detailed artwork increases file size. The exporter retains one indexed
image per recorded row, so memory still grows with episode length; it does not
also retain a full RGB copy of the episode.

## Provenance and checksums

Trusted histories came from the native visual audit's `MountainCarPOMDP`
`episode_42.pkl` and `episode_43.pkl`, captured before this renderer was written.
Matching diagnostic and package replays use these exact bytes:

| Artifact | SHA-256 |
| --- | --- |
| History 42 | `b085cce2a303d38be329fcbc019998939abf960bb413e0d638db83f3607794b4` |
| History 43 | `186f43ed330b3e786dc1242469e1d0853e8f4648fc8579ae6a528b546a156936` |
| before-42-diagnostic.gif | `37e537f8d073cbefe5fd767aa0613a0fab199cd62bd29fee19006a1736bf090a` |
| after-42.gif | `2c52b219bbfad3f4774390e646fa5946d9a527cef4a49593f97fdb4b42dad471` |
| before-43-diagnostic.gif | `461166b78f5cf23cdb19dd7c662749d2fc179b236ddbc611e15ebc1fc80fb026` |
| after-43.gif | `085979c1bfa807ee5a1a5e2153ab4b9ba03afe9e9d1ec034a73a65e2f0a3caf5` |
