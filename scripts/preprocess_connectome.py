"""Build the processed connectome (neuron table + edge list) from the raw Codex files.

Usage:
    python scripts/preprocess_connectome.py [key=value ...]
    python scripts/preprocess_connectome.py signs.glut=1 edges.min_syn_count=10
"""

from __future__ import annotations

import argparse

from connectome_driver.data.preprocess import run
from connectome_driver.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. signs.glut=1")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    report = run(cfg)
    width = max(map(len, report))
    for key, value in report.items():
        print(f"{key:<{width}}  {value:>12,}")
    print(f"\nwritten to {cfg.processed_dir}/")


if __name__ == "__main__":
    main()
