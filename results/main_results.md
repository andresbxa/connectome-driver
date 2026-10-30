### Performance

| Variant | Seeds | Eval return | Per seed | Final training return | Steps to 500 | Steps to 700 | Steps to 800 |
|---|---|---|---|---|---|---|---|
| connectome | 3 | 782 ± 41 | 806 / 806 / 735 | 815 | 128k (3/3) | 163k (3/3) | 267k (3/3) |
| rewired | 3 | 786 ± 38 | 792 / 746 / 820 | 821 | 104k (3/3) | 125k (3/3) | 219k (3/3) |
| random | 3 | 798 ± 33 | 767 / 796 / 832 | 804 | 91k (3/3) | 111k (3/3) | 176k (3/3) |
| block | 3 | 806 ± 31 | 790 / 787 / 842 | 808 | 104k (3/3) | 118k (3/3) | 178k (3/3) |
| mlp_matched | 3 | 739 ± 49 | 686 / 748 / 784 | 768 | 308k (3/3) | 328k (3/3) | 429k (3/3) |
| nocore | 3 | -13 ± 50 | -41 / -43 / 45 | -9 | not reached (0/3) | not reached (0/3) | not reached (0/3) |
| connectome_T4 | 2 | 383 ± 600 | -41 / 808 | 392 | 196k (1/2) | 210k (1/2) | 345k (1/2) |
| connectome_T12 | 2 | 792 ± 0 | 793 / 792 | 806 | 106k (2/2) | 127k (2/2) | 230k (2/2) |
| rewired_T4 | 2 | 758 ± 89 | 821 / 695 | 811 | 104k (2/2) | 120k (2/2) | 204k (2/2) |
| rewired_T12 | 2 | 809 ± 0 | 809 / 809 | 818 | 170k (2/2) | 202k (2/2) | 316k (2/2) |
| random_T4 | 2 | 809 ± 25 | 791 / 827 | 814 | 104k (2/2) | 131k (2/2) | 284k (2/2) |
| random_T12 | 2 | 817 ± 5 | 820 / 814 | 821 | 126k (2/2) | 166k (2/2) | 255k (2/2) |

### Comparison with the reference (Welch's t-test over seeds)

| Variant | Compared with | Eval return difference | p | Steps to 500, ratio of medians | p | Steps to 700, ratio of medians | p | Steps to 800, ratio of medians | p |
|---|---|---|---|---|---|---|---|---|---|
| rewired | connectome | +4 | 0.91 | 0.81x | 0.99 | 0.76x | 0.39 | 0.82x | 0.20 |
| random | connectome | +16 | 0.62 | 0.71x | 0.23 | 0.68x | 0.05 | 0.66x | 0.25 |
| block | connectome | +24 | 0.46 | 0.81x | 0.45 | 0.72x | 0.06 | 0.67x | 0.31 |
| mlp_matched | connectome | -43 | 0.31 | 2.40x | 0.11 | 2.01x | 0.07 | 1.61x | 0.32 |
| nocore | connectome | -795 | < 0.01 | n/a |  | n/a |  | n/a |  |
| rewired_T4 | connectome_T4 | +375 | 0.54 | n/a |  | n/a |  | n/a |  |
| rewired_T12 | connectome_T12 | +16 | < 0.01 | 1.60x | 0.20 | 1.59x | 0.20 | 1.37x | 0.33 |
| random_T4 | connectome_T4 | +425 | 0.50 | n/a |  | n/a |  | n/a |  |
| random_T12 | connectome_T12 | +24 | 0.09 | 1.19x | 0.67 | 1.31x | 0.37 | 1.11x | 0.73 |

### Depth sweep: eval return by number of core steps T (per seed in brackets)

| Graph | T = 4 | T = 8 | T = 12 |
|---|---|---|---|
| connectome | 383 (-41 / 808) | 782 (806 / 806 / 735) | 792 (793 / 792) |
| rewired | 758 (821 / 695) | 786 (792 / 746 / 820) | 809 (809 / 809) |
| random | 809 (791 / 827) | 798 (767 / 796 / 832) | 817 (820 / 814) |

### Cost (one GPU, configured as in training)

| Variant | Core params | Total params | Rollout forward (ms, 8 envs) | Update (ms per 256-sample minibatch) | Peak GPU memory (MB) | Est. hours per 1M steps |
|---|---|---|---|---|---|---|
| connectome | 338,417 | 2,538,887 | 13.9 | 107 | 1571 | 1.9 |
| rewired | 338,417 | 2,538,887 | 12.6 | 103 | 1571 | 1.8 |
| random | 338,417 | 2,538,887 | 12.3 | 106 | 1571 | 1.8 |
| block | 338,417 | 2,538,887 | 11.8 | 104 | 1571 | 1.8 |
| mlp_matched | 336,832 | 2,537,302 | 3.9 | 11 | 123 | 1.1 |
| nocore | 527,280 | 2,727,750 | 3.1 | 9 | 124 | 1.1 |
| connectome_T4 | 338,417 | 2,538,887 | 8.5 | 58 | 947 | 1.5 |
| connectome_T12 | 338,417 | 2,538,887 | 15.5 | 153 | 2195 | 2.1 |
| rewired_T4 | 338,417 | 2,538,887 | 9.5 | 60 | 947 | 1.5 |
| rewired_T12 | 338,417 | 2,538,887 | 17.5 | 152 | 2195 | 2.2 |
| random_T4 | 338,417 | 2,538,887 | 8.9 | 61 | 947 | 1.5 |
| random_T12 | 338,417 | 2,538,887 | 16.8 | 154 | 2195 | 2.2 |
