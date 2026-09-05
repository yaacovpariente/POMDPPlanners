# SPDX-License-Identifier: MIT
"""Pillow sprites shared by both LaserTag renderers."""

import numpy as np
from PIL import Image, ImageDraw


def _render_robot_pillow(size_px: int = 64) -> np.ndarray:
    """Draw the red humanoid, including its visor and antenna."""
    image = Image.new("RGBA", (64, 64))
    draw = ImageDraw.Draw(image)
    red, edge = "#D32F2F", "#B71C1C"
    draw.line((32, 10, 32, 2), fill=edge, width=2)
    draw.ellipse((30, 0, 34, 4), fill="#FF5252")
    draw.ellipse((24, 7, 40, 23), fill=red, outline=edge, width=2)
    draw.rounded_rectangle((23, 12, 41, 17), radius=2, fill="#FFCDD2", outline=edge)
    for box in (
        (20, 26, 44, 45),
        (10, 27, 18, 44),
        (46, 27, 54, 44),
        (21, 47, 29, 62),
        (35, 47, 43, 62),
    ):
        draw.rounded_rectangle(box, radius=3, fill=red, outline=edge, width=2)
    return np.asarray(image.resize((size_px, size_px), Image.Resampling.LANCZOS))


def _render_opponent_pillow(size_px: int = 64) -> np.ndarray:
    """Draw the blue rover, including its sensor, wheels and antenna."""
    image = Image.new("RGBA", (64, 64))
    draw = ImageDraw.Draw(image)
    edge = "#0D47A1"
    draw.line((44, 25, 50, 9), fill=edge, width=2)
    draw.polygon(((47, 9), (53, 9), (50, 3)), fill="#FF8F00")
    draw.ellipse((24, 10, 40, 26), fill="#90CAF9", outline=edge, width=2)
    draw.ellipse((29, 15, 35, 21), fill="white", outline=edge)
    draw.rounded_rectangle((8, 26, 56, 47), radius=5, fill="#1976D2", outline=edge, width=2)
    for x in (16, 32, 48):
        draw.ellipse((x - 6, 46, x + 6, 58), fill="#455A64", outline="#263238", width=2)
        draw.ellipse((x - 2, 50, x + 2, 54), fill="#90A4AE")
    return np.asarray(image.resize((size_px, size_px), Image.Resampling.LANCZOS))
