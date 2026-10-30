"""Results table of an experiment plan: return, sample efficiency and cost per variant.

For every plan entry with finished runs (trained and evaluated):

- eval return: mean over seeds of each seed's mean return on the 20 unseen evaluation tracks
  (``eval.json``), with the standard deviation over seeds (the unit of replication is the seed);
- final training return: mean return of the episodes in the last 100k training steps;
- steps to a threshold: first agent step at which the mean of the last 20 training episodes
  reaches it, per seed; the table shows the median over the seeds that reached it;
- comparison with a reference entry: difference of eval means and ratio of median steps to
  each threshold (when every seed of both entries reached it), with Welch's t-test over seeds.
  An entry with a suffix is compared with the reference of the same suffix when there is one
  (``rewired_T4`` with ``connectome_T4``);
- depth sweep: eval return per graph and number of core steps T, for entries named
  ``<graph>_T<steps>`` next to the ``<graph>`` entry they vary;
- cost, from ``scripts/profile_core.py --plan`` (one machine, each entry configured as in
  training): rollout forward time, update time per minibatch, peak GPU memory, estimated hours
  per 1M-step run.

Writes ``<out>.json`` and ``<out>.md``.

Usage:
    python scripts/summarize_results.py configs/experiments/main.yaml --only connectome mlp_matched nocore
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from connectome_driver.data.graphs import build_graph
from connectome_driver.envs.car_racing import observation_shape
from connectome_driver.models.policy import build_policy, needs_graph
from connectome_driver.utils.config import load_config
from connectome_driver.utils.plan import load_plan

NUM_ACTIONS = 5
FINAL_WINDOW = 100_000  # agent steps
ROLLING_EPISODES = 20  # as the training log's "return(last 20)"
DEPTH_ENTRY = re.compile(r"^(?P<graph>.+)_T(?P<steps>\d+)$")


def steps_to(episodes: pd.DataFrame, threshold: float) -> int | None:
    rolling = episodes["return"].rolling(ROLLING_EPISODES).mean()
    hit = np.flatnonzero(rolling.to_numpy() >= threshold)
    return int(episodes["step"].iloc[hit[0]]) if len(hit) else None


def summarize_entry(run_dirs: list[Path], thresholds: list[float]) -> dict:
    seeds, evals, finals, reach = [], [], [], {t: [] for t in thresholds}
    for d in run_dirs:
        episodes = pd.read_csv(d / "episodes.csv")
        seeds.append(int(d.name.removeprefix("seed")))
        evals.append(json.loads((d / "eval.json").read_text())["mean"])
        finals.append(float(episodes.loc[episodes.step > episodes.step.max() - FINAL_WINDOW, "return"].mean()))
        for t in thresholds:
            reach[t].append(steps_to(episodes, t))
    evals_arr = np.array(evals)
    return {
        "seeds": seeds,
        "eval_per_seed": evals,
        "eval_mean": float(evals_arr.mean()),
        "eval_sd": float(evals_arr.std(ddof=1)) if len(evals) > 1 else None,
        "final_train_return": float(np.mean(finals)),
        "steps_to": {str(t): steps for t, steps in reach.items()},
    }


def reference_for(name: str, reference: str, rows: dict) -> str | None:
    """Entry ``name`` is compared with; None for the reference and its own variants."""
    suffix = name[name.index("_"):] if "_" in name else ""
    if name in (reference, reference + suffix):
        return None
    return reference + suffix if reference + suffix in rows else reference if reference in rows else None


def depth_sweep(rows: dict, configs: dict) -> dict[str, dict[int, str]]:
    """Graph -> {core steps: entry}, for ``<graph>_T<steps>`` entries and the ``<graph>`` entry."""
    sweep: dict[str, dict[int, str]] = {}
    for name in rows:
        m = DEPTH_ENTRY.match(name)
        if m and m["graph"] in rows:
            sweep.setdefault(m["graph"], {})[int(m["steps"])] = name
    for graph, entries in sweep.items():
        entries[int(configs[graph].model.core.steps)] = graph
    return {graph: dict(sorted(entries.items())) for graph, entries in sweep.items()}


def parameter_counts(cfg) -> dict[str, int]:
    graph = build_graph(cfg.model.graph, seed=cfg.seed) if needs_graph(cfg.model) else None
    return build_policy(cfg.model, observation_shape(cfg.env), NUM_ACTIONS, graph).parameter_counts()


def fmt_steps(steps: list[int | None]) -> str:
    reached = [s for s in steps if s is not None]
    if not reached:
        return f"not reached (0/{len(steps)})"
    return f"{np.median(reached) / 1000:.0f}k ({len(reached)}/{len(steps)})"


def fmt_p(p: float) -> str:
    return f"{p:.2f}" if p >= 0.01 else "< 0.01"


def to_markdown(rows: dict, cost: dict, compare: dict, depth: dict, thresholds: list[float]) -> str:
    lines = ["### Performance", ""]
    head = ["Variant", "Seeds", "Eval return", "Per seed", "Final training return"]
    head += [f"Steps to {t:g}" for t in thresholds]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in rows.items():
        sd = f" ± {r['eval_sd']:.0f}" if r["eval_sd"] is not None else ""
        cells = [name, str(len(r["seeds"])), f"{r['eval_mean']:.0f}{sd}",
                 " / ".join(f"{e:.0f}" for e in r["eval_per_seed"]), f"{r['final_train_return']:.0f}"]
        cells += [fmt_steps(r["steps_to"][str(t)]) for t in thresholds]
        lines.append("| " + " | ".join(cells) + " |")

    if compare:
        head = ["Variant", "Compared with", "Eval return difference", "p"]
        for t in thresholds:
            head += [f"Steps to {t:g}, ratio of medians", "p"]
        lines += ["", "### Comparison with the reference (Welch's t-test over seeds)", "",
                  "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for name, c in compare.items():
            cells = [name, c["reference"], f"{c['difference']:+.0f}", fmt_p(c["p"])]
            for t in thresholds:
                s = c["steps_to"].get(str(t))
                cells += [f"{s['ratio_of_medians']:.2f}x", fmt_p(s["p"])] if s else ["n/a", ""]
            lines.append("| " + " | ".join(cells) + " |")

    if depth:
        steps = sorted({t for entries in depth.values() for t in entries})
        lines += ["", "### Depth sweep: eval return by number of core steps T (per seed in brackets)", "",
                  "| Graph | " + " | ".join(f"T = {t}" for t in steps) + " |", "|---|" + "---|" * len(steps)]
        for graph, entries in depth.items():
            cells = [graph]
            for t in steps:
                r = rows[entries[t]] if t in entries else None
                cells.append(f"{r['eval_mean']:.0f} ({' / '.join(f'{e:.0f}' for e in r['eval_per_seed'])})"
                             if r else "")
            lines.append("| " + " | ".join(cells) + " |")

    if cost:
        lines += ["", "### Cost (one GPU, configured as in training)", "",
                  ("| Variant | Core params | Total params | Rollout forward (ms, 8 envs) "
                   "| Update (ms per 256-sample minibatch) | Peak GPU memory (MB) | Est. hours per 1M steps |"),
                  "|---|---|---|---|---|---|---|"]
        for name, c in cost.items():
            lines.append(
                f"| {name} | {c['core_params']:,} | {c['total_params']:,} | {c['forward_ms']:.1f} "
                f"| {c['update_ms']:.0f} | {c['peak_mem_mb']:.0f} | {c['est_hours_per_run']:.1f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--only", nargs="*", default=None, help="plan entries to include")
    parser.add_argument("--reference", default="connectome", help="entry the others are compared with")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[500, 700, 800])
    parser.add_argument("--profile", type=Path, default=Path("results/profile_main.json"))
    parser.add_argument("--out", type=Path, default=Path("results/main_results"))
    args = parser.parse_args()

    run_dir = Path(load_config("configs/train.yaml").run_dir)
    runs = [r for r in load_plan(args.plan) if args.only is None or r.exp_name in args.only]
    names = list(dict.fromkeys(r.exp_name for r in runs))
    rows, configs = {}, {}
    for name in names:
        done = [r for r in runs if r.exp_name == name and r.status(run_dir) == "done"]
        if not done:
            print(f"{name}: no finished runs, skipped")
            continue
        rows[name] = summarize_entry(sorted(r.dir(run_dir) for r in done), args.thresholds)
        configs[name] = load_config(done[0].dir(run_dir) / "config.yaml")

    compare = {}
    for name, r in rows.items():
        ref_name = reference_for(name, args.reference, rows)
        if ref_name is None or len(r["eval_per_seed"]) < 2 or len(rows[ref_name]["eval_per_seed"]) < 2:
            continue
        ref = rows[ref_name]
        t, p = stats.ttest_ind(r["eval_per_seed"], ref["eval_per_seed"], equal_var=False)
        compare[name] = {"reference": ref_name, "difference": r["eval_mean"] - ref["eval_mean"],
                         "t": float(t), "p": float(p), "steps_to": {}}
        for thr in map(str, args.thresholds):  # only when every seed of both reached it
            a, b = r["steps_to"][thr], ref["steps_to"][thr]
            if None not in a and None not in b:
                t, p = stats.ttest_ind(a, b, equal_var=False)
                compare[name]["steps_to"][thr] = {"ratio_of_medians": float(np.median(a) / np.median(b)),
                                                  "t": float(t), "p": float(p)}
    depth = depth_sweep(rows, configs)

    cost = {}
    if args.profile.exists():
        profiled = {row["exp"]: row for row in json.loads(args.profile.read_text())["rows"] if "exp" in row}
        for name in rows:
            if name in profiled:
                counts = parameter_counts(configs[name])
                p = profiled[name]
                cost[name] = {"core_params": counts["core"], "total_params": sum(counts.values()),
                              **{k: p[k] for k in ("forward_ms", "update_ms", "micro_batch",
                                                    "peak_mem_mb", "est_hours_per_run")}}
    else:
        print(f"{args.profile} not found: no cost columns (run scripts/profile_core.py --plan first)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {"plan": args.plan.as_posix(), "thresholds": args.thresholds, "reference": args.reference,
              "variants": rows, "comparison": compare, "depth_sweep": depth, "cost": cost}
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2))
    markdown = to_markdown(rows, cost, compare, depth, args.thresholds)
    args.out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"written to {args.out}.json and .md")


if __name__ == "__main__":
    main()
