"""Replay captured JSON without running a planner: python replay.py after 42."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.tiger_visualizer import TigerVisualizer


def diagnostic(history, target):
    """Tiger branch of the original standalone visual-audit diagnostic."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = []
    for index, step in enumerate(history):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        ax.set_title(f"TigerPOMDP — diagnostic baseline — frame {index + 1}/{len(history)}")
        ax.set_xlim(-0.2, 2.2)
        ax.set_ylim(0, 1.7)
        ax.axis("off")
        for x, label in ((0.2, "LEFT"), (1.2, "RIGHT")):
            ax.add_patch(plt.Rectangle((x, 0.1), 0.8, 1.1, facecolor="#8D6E63", edgecolor="black"))
            ax.text(x + 0.4, 0.65, label, ha="center", va="center", color="white", weight="bold")
        hidden_x = 0.6 if step.state == "tiger_left" else 1.6
        ax.text(
            hidden_x,
            1.35,
            "TIGER — hidden ground truth",
            ha="center",
            color="#C62828",
            weight="bold",
        )
        counts = {"tiger_left": 0, "tiger_right": 0}
        for particle in step.belief.particles:
            if particle in counts:
                counts[particle] += 1
        total = max(sum(counts.values()), 1)
        ax.text(0.6, 0.02, f"belief tiger-left {counts['tiger_left']/total:.2f}", ha="center")
        ax.text(1.6, 0.02, f"belief tiger-right {counts['tiger_right']/total:.2f}", ha="center")
        ax.text(
            0.01,
            0.99,
            f"action={step.action!r}   observation={step.observation!r}   reward={step.reward!r}",
            transform=ax.transAxes,
            va="top",
        )
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(
            Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            ).convert("RGB")
        )
        plt.close(fig)
    frames[0].save(target, save_all=True, append_images=frames[1:], duration=500, loop=0)


if __name__ == "__main__":
    arm, seed = sys.argv[1], int(sys.argv[2])
    root = Path(__file__).resolve().parent
    rows = json.loads((root / f"episode_{seed}.json").read_text())["history"]
    history = [
        StepData(
            row["state"],
            row["action"],
            row["next_state"],
            row["observation"],
            row["reward"],
            WeightedParticleBelief(
                row["belief"]["particles"], np.asarray(row["belief"]["log_weights"])
            ),
        )
        for row in rows
    ]
    output = root / f"replayed-{arm}-{seed}.gif"
    if arm == "before":
        diagnostic(history, output)
    elif arm == "after":
        TigerVisualizer().create_visualization(history, output)
    else:
        raise ValueError("arm must be before or after")
