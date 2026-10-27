# Task: null-model graphs (Phase 4)

**Branch:** `phase-4-nulls` (from `main`) · **Depends on:** `main` (Phases 1-4 are merged)
**Out of scope:** model, training and PPO code; they live in other phases.

**Two parts:** (1) the null models, below; (2) once they are merged, the whole-brain runs on
your GPU (see "After the nulls: whole-brain runs").

## Why this matters

The experiment asks whether the *wiring* of the fly connectome helps a driving policy. To
answer it, the real graph is compared against graphs that keep some of its properties and
destroy the rest. Every variant uses the same encoder, decoder, PPO and hyperparameters; only
the graph changes. If the nulls are wrong (e.g. they silently change the degree sequence), the
whole comparison is meaningless, so correctness and tests come first.

## What you get from Phase 1

`connectome_driver.data.graph.Connectome` (see `src/connectome_driver/data/graph.py`):

| Field | Shape / type | Meaning |
|---|---|---|
| `neurons` | DataFrame, N rows | row i is node i; columns include `super_class`, `class`, `side`, `nt_type`, `sign` (+1/-1), `role`, `is_visual_afferent`, `is_readout` |
| `src`, `dst` | (E,) int64 | presynaptic / postsynaptic node of each edge |
| `syn_count` | (E,) int32 | synapses of the edge (>= 5) |
| `neuropil` | (E,) int16 | code of the neuropil holding most of the edge's synapses |
| `neuropil_names` | list[str] | vocabulary for `neuropil` |

Real v783 graph: N = 139,255 nodes, E = 3,732,460 edges, no self-loops, no duplicate edges.
Load it with `load_connectome("data/processed")` after running the Phase 1 preprocessing.

**Which graph the nulls are applied to.** The main experiment trains on the k = 3 path
subgraph (13,261 neurons, 107,764 edges: neurons on a path of at most 3 hops from the
photoreceptors to the descending neurons). `build_graph` cuts the subgraph first and then
applies the null to it, with `seed` = the run's seed (1, 2, 3), so every seed trains on a
different null sample. The nulls therefore run on graphs of ~10^5 edges in every training run
(must take seconds) and on the full brain only for the whole-brain check (minutes).

**Signs are a node property, not an edge property.** The sign of an edge is computed later as
`neurons["sign"][src]` (Dale's law: one transmitter per presynaptic neuron). Nulls must never
store per-edge signs; they only move edges, and the sign follows the new presynaptic neuron
automatically.

## What to build

`src/connectome_driver/data/nulls.py` with three functions. All return a new `Connectome` with
the **same `neurons` table** and new edge arrays, are deterministic for a given `seed`, and use
vectorized NumPy (networkx does not scale to millions of edges).

### 1. `rewire_degree_preserving(graph, seed, swaps_per_edge=10.0) -> Connectome`

Directed double edge swap: pick two edges `a->b` and `c->d`, replace them with `a->d` and
`c->b`. Each edge keeps its source and its `syn_count` (and `neuropil`) and only its target
changes, so every neuron keeps its out-degree, its in-degree and the total weight it sends.

Reject a proposed swap if it would create a self-loop (`a == d` or `c == b`) or an edge that
already exists. Keep going until `swaps_per_edge * E` swaps have been **accepted**.

Suggested vectorized scheme (you may do better):
1. Pair edges by a random permutation, so each edge is in at most one proposal per batch.
2. Encode edges as `key = src * N + dst` (int64) and keep the current keys sorted.
3. For all proposals at once: compute the two new keys, reject self-loops, reject keys already
   present (`np.searchsorted` on the sorted keys, or `np.isin`), and reject proposals whose new
   keys collide with each other inside the batch (`np.unique` with counts).
4. Apply the accepted swaps, rebuild the sorted keys, repeat.

Log the acceptance rate and, at the end, the fraction of edges whose target differs from the
original (it should be close to 1 after enough swaps).

### 2. `random_graph(graph, seed) -> Connectome`

Same N and E, edges drawn uniformly at random among all ordered pairs `(i, j)` with `i != j`,
without duplicates. `syn_count` (with its `neuropil`) is a random permutation of the real values,
so the weight distribution is preserved and the sign follows the new presynaptic neuron.

### 3. `rewire_within_blocks(graph, blocks, seed, swaps_per_edge=10.0) -> Connectome`

Same as (1), but a swap is only allowed between two edges with the same
`(blocks[src], blocks[dst])` pair. This preserves the degrees **and** the number of edges
between every pair of blocks, i.e. the coarse region-to-region architecture, while scrambling
the wiring inside it. It separates "the fine wiring matters" from "the coarse architecture is
enough".

`blocks` is an int array of length N. Add a helper
`neuron_blocks(graph, by: str) -> np.ndarray` with at least:
- `by="super_class"`: one block per value of `neurons["super_class"]`;
- `by="neuropil"`: each neuron's dominant **output** neuropil (the `neuropil` of its out-edges,
  weighted by `syn_count`); neurons without out-edges get their own block `-1`.

Hint: run (1) independently inside each block pair, or sort edges by block pair and only pair
edges within the same group.

### Where it plugs in (already done)

`src/connectome_driver/data/graphs.py` (`apply_variant`) already calls these four functions
with exactly the signatures above, and `configs/variant/{rewired,random,block}.yaml` select
them. As soon as `connectome_driver/data/nulls.py` exists, training works with
`--model configs/model/connectome.yaml --model configs/variant/rewired.yaml`, and the
integration tests in `tests/test_controls.py` stop being skipped. If you need to change a
signature, change `apply_variant` in the same PR.

## Tests (`tests/test_nulls.py`)

They must pass on the toy brain in `tests/conftest.py` and on random graphs of a few thousand
edges, and run in seconds without FlyWire data (CI has no data).

- In- and out-degree of every node identical to the original (rewire and block rewire).
- Same E; no self-loops; no duplicate `(src, dst)`.
- Multiset of `syn_count` unchanged; each node's total outgoing `syn_count` unchanged (rewire).
- Block rewire: the count matrix of edges per `(block_src, block_dst)` is unchanged.
- Random graph: N and E unchanged, no self-loops or duplicates, `syn_count` multiset unchanged.
- Same seed gives the same graph; different seeds give different graphs.
- Rewiring actually mixes: on a random test graph, most targets change after 10 swaps/edge.
- `neurons` table is untouched (so signs, roles and readout neurons are unchanged).

## Report (`scripts/null_stats.py`)

For the real graph and one sample of each null (`seed=0`), on both the k = 3 subgraph (what
the main experiment trains on) and the full brain, write `results/null_stats.json` with:
- runtime of each null (target: seconds on the subgraph, minutes on the full brain);
- acceptance rate and fraction of rewired edges;
- `stats.reachability(...)` from visual afferents to readout neurons with hop budgets
  `[2, 3, 4, 8, 12]` (already implemented in `src/connectome_driver/data/stats.py`).

The last point matters: random graphs have much shorter paths than the real one, so a null
could beat the connectome just by delivering the visual signal to the descending neurons in
fewer steps. We need these numbers to interpret the results.

## After the nulls: whole-brain runs on your GPU

The main comparison (k = 3 subgraph) runs on Andrés's laptop. Your GPU runs the check on the
complete connectome: `configs/experiments/whole_brain.yaml`, connectome / rewired / random at
T = 4, 2 seeds each (6 runs, ~4.8 h each on an 8 GB RTX 3070 Ti Laptop; slower or faster on
yours). Start once your nulls PR and the experiment runner are both on `main`.

1. **Check that it fits.** Send the output of `nvidia-smi`. With less than 8 GB, change
   `ppo.micro_batch_size=128` to `64` in the plan's `common` line (the update is identical,
   only slower). On Windows a run that does not fit does not crash: the driver silently
   spills to system RAM and training becomes several times slower, so check the `sps` column
   of the first log lines.
2. **Keep the machine awake and plugged in.** The runner asks Windows not to idle-sleep, but
   closing the lid or running on battery still suspends or throttles it (a sleeping laptop
   froze our first overnight sweep for 8 hours).
3. **Run the plan**, one run at a time (a full-brain run uses most of the GPU):

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

## First: add your email as coauthor

You are already listed as coauthor; only your email is missing. In your first commit:

1. `pyproject.toml`: change your entry to
   `{ name = "Joshue Calero Tamayo", email = "you@example.com" }`.
2. `CITATION.cff`: add `email: you@example.com` under your entry (same indentation as
   `family-names`).
3. Set your git identity in the repo so your commits carry your name:
   `git config user.name "Joshue Calero Tamayo"` and `git config user.email "you@example.com"`
   (use your GitHub email, or the `...@users.noreply.github.com` one if you keep it private).

## Setup

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

## Conventions

- Code, comments and docstrings in English; match the style of `src/connectome_driver/data/`.
- `ruff check src scripts tests` and `pytest` must pass.
- Never commit anything under `data/`.
- Small commits with descriptive messages; open a PR to `main` when done.
