# Light-Dark sprites

These original RGBA assets were generated for this renderer with the built-in image generation tool on 2026-09-05. They are packaged with the library; rendering never calls an image service. Generated 1254 × 1254 sources were reduced to 192 × 192 with Lanczos filtering for packaging, retaining true alpha while avoiding large PNG decodes for sprites drawn at 64 pixels or less. Runtime resize and heading caches keep repeated saves fast.

Prompt briefs:

- `rover.png`: exactly overhead red mechanical rover facing right, four dark treaded wheels, blue glass cabin, metal bolts and pipes, bright front headlights, isolated on transparent alpha.
- `beacon.png`: overhead bronze-and-steel circular lamp, warm ivory lens, protective cross cage, four bolts, small cobalt navigation badge, transparent alpha.
- `goal.png`: raised green enamel star on a round bronze-and-steel base, bolts and dimensional highlights, transparent alpha.

The art follows the Light-Dark proposal's materials and symbols. It does not add world objects or simulated vehicle pose; rover heading still comes from the existing action/trajectory mapping.
