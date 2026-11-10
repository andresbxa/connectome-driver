# Task: whole-brain runs on the second GPU (Phase 5)

**Owner:** Joshue · **Depends on:** `main` with the null models merged (`phase-4-nulls`)
**Status:** sections 1-3 done (email in PR #9, null review and fix in PR #8). The plan is split
by seed: Andrés runs seed 1 of the four graphs on the RTX 3070 Ti, you run seed 2 on the
RX 7600 with the ROCm backend (section 4).
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

## 4. Run seed 2 of the whole-brain plan

The plan has connectome / rewired / random / block at T = 4, 2 seeds each (8 runs).
Splitting by seed keeps each seed's four graphs on one machine and one backend, so the
hardware difference falls equally on every graph instead of being confused with one of them.
Your part is seed 2: four runs of `configs/experiments/whole_brain_rocm.yaml`, about 12 h each
on the RX 7600 (~2 days in total). Andrés's seed 1 runs with `whole_brain.yaml` (same
configuration, PyTorch backend, micro-batches of 128).

1. **Validate the block null first.** Your validation in `results/rocm_validation.json` covers
   connectome, rewired and random; run the same output and gradient check for `block` with
   seeds 1 and 2 (see [`docs/rocm.md`](../rocm.md)) and add it to that file.
2. **Use the performance power mode and keep the machine plugged in, lid open.** The runner
   asks Windows not to idle-sleep, but cannot stop a suspend. A quiet or "Silent" power mode
   slowed our first whole-brain run 2.3x (GPU at 210 MHz); check that the `sps` of the first
   log lines matches your validation (~23 steps/s).
3. **Run it**, one run at a time (a full-brain run uses most of the GPU):

   ```powershell
   $env:CONNECTOME_TUNABLEOP_CACHE = (Resolve-Path .venv/rocm-gemm.csv).Path
   python scripts/run_experiments.py configs/experiments/whole_brain_rocm.yaml --seeds 2 --status
   python scripts/run_experiments.py configs/experiments/whole_brain_rocm.yaml --seeds 2
   ```

   Finished runs are skipped, so after an interruption run the same command again. A run
   that was cut halfway starts again from zero (there is no mid-run resume), so if you need
   the machine, stop right after a run finishes. Each run trains for 1M steps and then
   evaluates on 20 unseen tracks.
4. **Send back** each `runs/<name>/seed2/` folder (`config.yaml`, `runtime.json`,
   `episodes.csv`, `iterations.csv`, `eval.json`, `eval_runtime.json`, `DONE`, `policy.pt`) and
   the `seed2.log` files, e.g. as a zip. Do not commit `runs/`. Wall-clock times from your
   machine are not used as a cost metric (all timings come from one machine).

**What a failed run looks like.** Some seeds learn to drive off the map: the reward clip
turns its -100 penalty into -1, which can beat a slow lap. Episodes then last ~230-350 frames
instead of 1,000 and the return stays near -40. It happened to the connectome at T = 4 with
seed 1 on the subgraph. The whole-brain connectome run with seed 1 did the same for its first
250k steps, then learned to drive within 50k steps and ended at 765 on the evaluation tracks
(`results/whole_brain_results.md`). Do not stop or restart such a run: a collapse can end on
its own, it is part of the result either way, and every graph gets the same treatment.

## Done when

- the block null passes the ROCm check for seeds 1 and 2;
- the 4 seed-2 runs of `whole_brain_rocm.yaml` show `done` in `--status` and their folders
  are shared.

## Conventions

- Code, comments and docstrings in English.
- `ruff check src scripts tests` and `pytest` must pass.
- Never commit anything under `data/` or `runs/`.
- Small commits with descriptive messages; open PRs to `main`.
