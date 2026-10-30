"""Depth sweep figure: eval return vs number of core steps T, one dot per seed, one line per graph.

Reads the ``depth_sweep`` and per-seed eval returns written by ``scripts/summarize_results.py``.
The line joins the means over seeds; the dots show every seed, so a single collapsed seed is
visible instead of hidden in the mean.

Usage:
    python scripts/plot_depth.py results/main_results.json --out results/depth_sweep.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_curves import GRID, INK, MUTED, SURFACE, color_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/depth_sweep.png"))
    parser.add_argument("--title", default="Eval return vs core steps T (dots: seeds, line: mean)")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    variants, sweep = summary["variants"], summary["depth_sweep"]
    if not sweep:
        raise SystemExit(f"{args.summary} has no depth sweep")
    steps = sorted({int(t) for entries in sweep.values() for t in entries})
    position = {t: i for i, t in enumerate(steps)}

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    used: list[str] = []
    width = 0.24
    for g, (graph, entries) in enumerate(sweep.items()):
        color = color_for(graph, used)
        used.append(color)
        offset = (g - (len(sweep) - 1) / 2) * width
        xs, means = [], []
        for t, name in sorted(entries.items(), key=lambda item: int(item[0])):
            x = position[int(t)] + offset
            per_seed = variants[name]["eval_per_seed"]
            ax.scatter([x] * len(per_seed), per_seed, s=36, color=color, edgecolors=SURFACE,
                       linewidths=2, zorder=3)
            xs.append(x)
            means.append(variants[name]["eval_mean"])
        ax.plot(xs, means, color=color, linewidth=2, label=graph, zorder=2)

    ax.set_xticks(range(len(steps)), [f"T = {t}" for t in steps])
    ax.set_xlim(-0.6, len(steps) - 0.4)
    ax.set_ylabel("eval return (20 unseen tracks)")
    ax.set_title(args.title, color=INK, loc="left")
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
