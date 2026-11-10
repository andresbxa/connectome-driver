# Task: whole-brain runs on the second GPU (Phase 5)

**Owner:** Joshue · **Depends on:** `main` with the null models merged (`phase-4-nulls`)
**Out of scope:** changing code, configs or hyperparameters; if something looks wrong, open an
issue or comment on the PR instead, so every run of the experiment uses the same setup.

## Why this matters

The main comparison runs on the k = 3 path subgraph (13,261 neurons) on Andrés's laptop:
connectome vs degree-preserving rewired, random and block-rewired graphs, a parameter-matched
MLP and a no-core ablation, 3 seeds each. Your runs answer one question: **does the result hold
on the complete connectome** (139,255 neurons, 3.7M edges)?

One thing to keep in mind when reading them. On the subgraph, rewiring puts most descending
neurons one hop away from the photoreceptors (823 of 1,014 in the rewired graph, 2 in the real
one), because high out-degree photoreceptors get wired straight to high in-degree descending
neurons. A null can therefore differ from the connectome just because the visual signal arrives
sooner. `results/null_stats.json` has the same numbers for the full brain: within 2 hops of the
photoreceptors are 12% of the descending neurons in the connectome, 98% in the rewired graph,
100% in the random one and 62% in the block null. The block null keeps the most of the real
path structure, which is why it is in the plan even though it adds two runs.

## 1. Setup

```bash
git clone https://github.com/andresbxa/connectome-driver.git
cd connectome-driver
python -m venv .venv
.venv/Scripts/activate            # Windows; `source .venv/bin/activate` elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
nvidia-smi                        # tells us which GPU you have
```

Download the FlyWire v783 files with your own Codex account (see README, section Data), then:

```bash
python scripts/preprocess_connectome.py
python scripts/graph_stats.py
pytest
```

## 2. Add your email as coauthor (first commit)

You are already listed as coauthor; only your email is missing.

1. `pyproject.toml`: change your entry to
   `{ name = "Joshue Calero Tamayo", email = "you@example.com" }`.
2. `CITATION.cff`: add `email: you@example.com` under your entry (same indentation as
   `family-names`).
3. Set your git identity in the repo so your commits carry your name:
   `git config user.name "Joshue Calero Tamayo"` and `git config user.email "you@example.com"`
   (use your GitHub email, or the `...@users.noreply.github.com` one if you keep it private).

Open a small PR to `main` with this commit.

## 3. Review the null models (optional, ~30 min)

A second pair of eyes on `src/connectome_driver/data/nulls.py` and `tests/test_nulls.py` is
welcome: the whole comparison depends on the nulls keeping what they claim to keep. The
specification they follow is [`null-models.md`](null-models.md). Leave comments on the PR or in
an issue.

## 4. Run the whole-brain plan

`configs/experiments/whole_brain.yaml`: connectome / rewired / random / block at T = 4, 2 seeds
each (8 runs, ~4.8 h each on an 8 GB RTX 3070 Ti Laptop; slower or faster on your card). On an
AMD card use `whole_brain_rocm.yaml` instead, after the validation in [`docs/rocm.md`](../rocm.md)
(~12 h per run on an RX 7600). The runner goes seed by seed, so after the first four runs every
graph already has one seed.

1. **Check that it fits.** Send the output of `nvidia-smi`. With less than 8 GB, change
   `ppo.micro_batch_size=128` to `64` in the plan's `common` line (the update is identical,
   only slower). On Windows a run that does not fit does not crash: the driver silently
   spills to system RAM and training becomes several times slower, so check the `sps` column
   of the first log lines.
2. **Keep the machine awake and plugged in.** The runner asks Windows not to idle-sleep, but
   closing the lid or running on battery still suspends or throttles it (a sleeping laptop
   froze our first overnight sweep for 8 hours).
3. **Run it**, one run at a time (a full-brain run uses most of the GPU):

   ```bash
   python scripts/run_experiments.py configs/experiments/whole_brain.yaml --status
   python scripts/run_experiments.py configs/experiments/whole_brain.yaml
   ```

   Finished runs are skipped, so after an interruption run the same command again. A run
   that was cut halfway starts again from zero (there is no mid-run resume). Each run
   trains for 1M steps and then evaluates on 20 unseen tracks.
4. **Send back** each `runs/<name>/seed<k>/` folder (`config.yaml`, `episodes.csv`,
   `iterations.csv`, `eval.json`, `DONE`, `policy.pt`) and the `seed<k>.log` files, e.g. as a
   zip. Do not commit `runs/`. Wall-clock times from your machine are not used as a cost
   metric (all timings come from one machine), but say which GPU you used.

## Done when

- your email is in `pyproject.toml` and `CITATION.cff` (merged PR);
- the 8 runs of the plan show `done` in `--status` and their folders are shared.

## Conventions

- Code, comments and docstrings in English.
- `ruff check src scripts tests` and `pytest` must pass.
- Never commit anything under `data/` or `runs/`.
- Small commits with descriptive messages; open PRs to `main`.
