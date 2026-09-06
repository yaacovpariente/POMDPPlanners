# PacMan saved-GIF redesign

Dark stone tiles, raised blue walls, shaded characters, and glowing pellets replace the flat maze. Each simulated ghost now draws one ghost, rather than the old four-character atlas.

## Matched exports

Both sides replay the same saved seed-42 history through `PacManPOMDP(discount_factor=.95)`: 20 frames, 1000 ms per frame, loop forever. Native states, beliefs, and actions are unchanged. The history ends at the recording horizon, so the new final row keeps its selected action instead of falsely saying Terminal.

The baseline renderer is unchanged source from develop `8973840781a16e78d0cc56c0e4ecb09d3f52e646`; its PacMan source is also identical at `4da132e7`. The same explicit tile size is used on both sides. Seed 43 was also replayed for visual coverage.

| Tile size | Before | After | Dimensions |
| --- | --- | --- | --- |
| 96 px | [Before GIF](before-96.gif) | [After GIF](after-96.gif) | 672 × 752 |
| 32 px (unchanged default) | [Before GIF](before-32.gif) | [After GIF](after-32.gif) | 224 × 304 |

The PNG previews were decoded from saved GIFs, so they include palette conversion.

## Complete-save timings

Measured on macOS 26.5.1 arm64, Python 3.12.14, Pillow 12.3.0. Ubuntu64 became unreachable, so these use the approved local fallback with one thread and no GPU. Each row ran sequentially in a fresh process. First-save samples include renderer construction and art preparation, but exclude imports and environment construction; warmed medians cover five complete saves on the same renderer. A separate one-CPU Docker build was active during this local comparison. Raw samples and output hashes are in the adjacent JSON files.

| Tile size | First save, before → after (s; one sample) | Warmed median, before → after (s) | Bytes, before → after |
| --- | --- | --- | --- |
| 96 px | 0.18266 → 0.12542 | 0.16811 → 0.08268 | 143,996 → 1,934,583 |
| 32 px | 0.04828 → 0.05605 | 0.03729 → 0.02536 | 59,820 → 256,667 |

Static maze composition, sized sprites, fonts, and palette are cached. All frames still encode on every save. The extra texture increases file size; default first-save startup is slightly slower in this single sample.

## Inputs and artwork

Saved-history SHA-256:

- Seed 42: `5d06f093266ad77336ddfa07f19db618f50f64d9b4c0c0b9c77b6f5097efd512`
- Seed 43: `b8462e72e96aac12227eafcf12626ca1c617ae7c8d25c4ed1049aeea1f610504`

The two compact RGBA character assets were generated once from a prompt for a shaded gold player and one red ghost on black. Offline matte extraction removed the black background and preserved soft edges. Runtime only loads packaged PNGs; it never generates images or accesses machine-specific paths. Stone tiles and pellet light are deterministic cached Pillow/NumPy artwork.

Coordinates, wall cells, pellets, true-state scores, collision status, and relative ghost-belief mass retain their meaning. Danger and belief overlays stay below entities and cannot paint the status panel. Dynamics and observations are untouched.

## Verification

- Native Mac: 110 existing/new renderer and environment tests passed, then the added one-pixel and multi-ghost/exact-score regressions passed in the final Linux run. Pyright reports zero errors and warnings on all three changed Python files.
- Local Colima, emulated Linux amd64, Python 3.10.20, one CPU and 3 GB: 21 focused/golden tests passed; two fresh renderers produced identical bytes for the actual golden fixture. The checked-in golden was generated here. This is not a claim that GitHub CI has already run.
- Black format/check passed. A built wheel contained both new PNGs, and both loaded from a temporary install outside the checkout. The existing package-data glob required no change.
- Own review and independent Claude review completed. Review fixes restored distinct ghost colors, exact score text, and font fallback. Decoded final GIFs passed visual review at both sizes.
