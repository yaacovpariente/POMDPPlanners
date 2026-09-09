# SPDX-License-Identifier: MIT
"""Reference artwork geometry copied from highway-env 1.12.1 lane parameters.

These are world coordinates, not the planner's re-integrated curvature map.
The regression test compares every sampled edge with the actual simulator.
This module is rendering-only and never participates in transitions/collisions.
"""

import numpy as np

# Straight: start, end, width, line types. Arc: centre, radius, start/end
# phase (radians), clockwise, width, line types. Preserve source seam overlaps.
_LANES = [
    ("straight", [42, 0], [100, 0], 5, [2, 1]),
    ("straight", [42, 5], [100, 5], 5, [1, 2]),
    ("arc", [100, -20], 20, 1.5707963267948966, -0.017453292519943295, False, 5, [2, 0]),
    ("arc", [100, -20], 25, 1.5707963267948966, -0.017453292519943295, False, 5, [1, 2]),
    ("straight", [120, -20], [120, -30], 5, [2, 0]),
    ("straight", [125, -20], [125, -30], 5, [1, 2]),
    ("arc", [105, -30], 15, 0.0, -3.1590459461097367, False, 5, [2, 0]),
    ("arc", [105, -30], 20, 0.0, -3.1590459461097367, False, 5, [1, 2]),
    ("arc", [70, -30], 20, 0.0, 2.3736477827122884, True, 5, [2, 1]),
    ("arc", [70, -30], 15, 0.0, 2.3911010752322315, True, 5, [0, 2]),
    ("straight", [55.7, -15.7], [35.7, -35.7], 5, [2, 0]),
    ("straight", [59.3934, -19.2], [39.3934, -39.2], 5, [1, 2]),
    ("arc", [18.1, -18.1], 25, 5.497787143782138, 2.9670597283903604, False, 5, [2, 0]),
    ("arc", [18.1, -18.1], 30, 5.497787143782138, 2.8797932657906435, False, 5, [1, 2]),
    ("arc", [18.1, -18.1], 25, 2.9670597283903604, 0.9773843811168246, False, 5, [2, 0]),
    ("arc", [18.1, -18.1], 30, 2.9670597283903604, 1.0122909661567112, False, 5, [1, 2]),
    ("arc", [43.2, 23.4], 23.5, 4.1887902047863905, 4.71238898038469, True, 5, [2, 1]),
    ("arc", [43.2, 23.4], 18.5, 4.153883619746504, 4.6774823953448035, True, 5, [0, 2]),
]


def reference_track_lanes() -> list[dict]:
    """Sample each exact lane edge at at most 0.25 m intervals for drawing."""
    lanes = []
    for spec in _LANES:
        width, line_types = spec[-2:]
        if spec[0] == "straight":
            start, end = np.asarray(spec[1], dtype=float), np.asarray(spec[2], dtype=float)
            length = float(np.linalg.norm(end - start))
            direction = (end - start) / length
            normal = np.array([-direction[1], direction[0]])
            distance = np.linspace(0, length, int(np.ceil(length / 0.25)) + 1)
            centre = start + distance[:, None] * direction
            edges = [centre + lateral * normal for lateral in (-width / 2, width / 2)]
        else:
            _, centre, radius, start, end, clockwise, _, _ = spec
            direction = 1 if clockwise else -1
            length = radius * (end - start) * direction
            phase = np.linspace(start, end, int(np.ceil(length / 0.25)) + 1)
            vectors = np.column_stack((np.cos(phase), np.sin(phase)))
            edges = [
                np.asarray(centre) + (radius - lateral * direction) * vectors
                for lateral in (-width / 2, width / 2)
            ]
        lanes.append(
            {
                "polygon": np.vstack((edges[0], edges[1][::-1])),
                "lines": [{"points": edge, "type": kind} for edge, kind in zip(edges, line_types)],
            }
        )
    return lanes
