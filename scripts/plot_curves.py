"""Learning curves from ``episodes.csv``: smoothed return vs agent steps, one line per experiment.

Each experiment is a directory ``runs/<exp_name>/`` holding one ``seed*/`` folder per seed. The
line is the mean over seeds of the per-seed curve (mean return of episodes in each step bin);
the band spans the min and max over seeds. With a single seed there is no band.

Usage:
    python scripts/plot_curves.py runs/mlp_baseline [runs/other ...] --out results/curves.png
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed categorical order: a variant keeps its color whatever subset of runs is plotted.
VARIANT_COLORS = {
    "connectome": "#2a78d6",
    "rewired": "#eb6834",
    "random": "#1baf7a",
    "block": "#6250d6",
    "mlp": "#eda100",
    "nocore": "#e87ba4",
}
FALLBACK_COLORS = ["#008300", "#e34948"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def color_for(name: str, used: list[str]) -> str:
    for key, color in VARIANT_COLORS.items():
        if name.startswith(key):
            return color
    return FALLBACK_COLORS[len(used) % len(FALLBACK_COLORS)]


def spread_labels(ys: list[float], gap: float) -> list[float]:
    """Heights for labels meant at ``ys``, at least ``gap`` apart; overlapping labels are stacked
    as a group centred on the mean of their lines."""
    def span(group: list[int]) -> tuple[float, float]:
        centre = float(np.mean([ys[i] for i in group]))
        return centre - (len(group) - 1) / 2 * gap, centre + (len(group) - 1) / 2 * gap

    groups: list[list[int]] = []
    for i in np.argsort(ys):
        groups.append([int(i)])
        while len(groups) > 1 and span(groups[-2])[1] + gap > span(groups[-1])[0]:
            top = groups.pop()
            groups[-1] += top
    heights = [0.0] * len(ys)
    for group in groups:
        low = span(group)[0]
        for k, i in enumerate(group):
            heights[i] = low + k * gap
    return heights


def seed_curve(csv: Path, bins: np.ndarray) -> np.ndarray:
    df = pd.read_csv(csv)
    idx = np.digitize(df["step"], bins) - 1
    curve = np.full(len(bins) - 1, np.nan)
    for b in range(len(curve)):
        sel = df["return"][idx == b]
        if len(sel):
            curve[b] = sel.mean()
    return curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/curves.png"))
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--title", default="CarRacing-v3: episode return during training")
    args = parser.parse_args()

    max_step = max(
        pd.read_csv(csv)["step"].max() for exp in args.experiments for csv in exp.glob("seed*/episodes.csv")
    )
    bins = np.linspace(0, max_step, args.bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    used: list[str] = []
    ends: list[tuple[str, float, float]] = []
    for exp in args.experiments:
        curves = np.array([seed_curve(csv, bins) for csv in sorted(exp.glob("seed*/episodes.csv"))])
        color = color_for(exp.name, used)
        used.append(color)
        n = len(curves)
        label = f"{exp.name} ({n} seed{'s' if n > 1 else ''})"
        with warnings.catch_warnings():  # bins past the end of a shorter run are all-NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(curves, axis=0)
        ax.plot(centers, mean, color=color, linewidth=2, label=label)
        if n > 1:
            ax.fill_between(centers, np.nanmin(curves, 0), np.nanmax(curves, 0), color=color, alpha=0.15, linewidth=0)
        last = np.flatnonzero(~np.isnan(mean))[-1]
        ends.append((exp.name, centers[last], mean[last]))

    low, high = ax.get_ylim()
    heights = spread_labels([y for _, _, y in ends], gap=0.045 * (high - low))
    for (name, x, _), y in zip(ends, heights, strict=True):
        ax.annotate(name, (x, y), xytext=(4, 0), textcoords="offset points", va="center", color=INK, fontsize=9)

    ax.set_xlabel("agent steps (1 step = 4 frames)")
    ax.set_ylabel("episode return")
    ax.set_title(args.title, color=INK, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
    if len(args.experiments) > 1:  # a single line is already named by its direct label
        # Curves rise and then plateau, so the middle right is usually empty (lower right can
        # hold a variant that never learned).
        ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.1)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
