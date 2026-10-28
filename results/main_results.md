### Performance

| Variant | Seeds | Eval return | Per seed | Final training return | Steps to 500 | Steps to 700 | Steps to 800 |
|---|---|---|---|---|---|---|---|
| connectome | 3 | 782 ± 41 | 806 / 806 / 735 | 815 | 128k (3/3) | 163k (3/3) | 267k (3/3) |
| mlp_matched | 3 | 739 ± 49 | 686 / 748 / 784 | 768 | 308k (3/3) | 328k (3/3) | 429k (3/3) |
| nocore | 3 | -13 ± 50 | -41 / -43 / 45 | -9 | not reached (0/3) | not reached (0/3) | not reached (0/3) |

### Compared with connectome (Welch's t-test over seeds)

| Variant | Eval return difference | p | Steps to 500, ratio of medians | p | Steps to 700, ratio of medians | p | Steps to 800, ratio of medians | p |
|---|---|---|---|---|---|---|---|---|
| mlp_matched | -43 | 0.31 | 2.40x | 0.11 | 2.01x | 0.07 | 1.61x | 0.32 |
| nocore | -795 | < 0.01 | n/a |  | n/a |  | n/a |  |

### Cost (one GPU, configured as in training)

| Variant | Core params | Total params | Rollout forward (ms, 8 envs) | Update (ms per 256-sample minibatch) | Peak GPU memory (MB) | Est. hours per 1M steps |
|---|---|---|---|---|---|---|
| connectome | 338,417 | 2,538,887 | 12.4 | 106 | 1571 | 1.7 |
| mlp_matched | 336,832 | 2,537,302 | 3.3 | 11 | 123 | 1.0 |
| nocore | 527,280 | 2,727,750 | 3.1 | 12 | 124 | 1.0 |
