**Partial: 1 of 8 runs** (connectome, seed 1). Seed 1 of the three nulls runs next on the same laptop; seed 2 of all four graphs runs on the RX 7600 with ROCm (`docs/tasks/whole-brain-runs.md`). This file is regenerated when they finish.

The run drove off the map (return near -40, median episode of 259 frames) for its first 250k steps, then learned to drive within 50k steps (`results/whole_brain_curves.png`).

### Performance

| Variant | Seeds | Eval return | Per seed | Final training return | Steps to 500 | Steps to 700 | Steps to 800 |
|---|---|---|---|---|---|---|---|
| connectome_full_T4 | 1 | 765 | 765 | 762 | 280k (1/1) | 292k (1/1) | 312k (1/1) |
