"""Print the real schema of the raw FlyWire Codex files before anything relies on it.

For each file: shape, columns, dtypes, a few rows, and value counts of the columns the
pipeline uses. Run this first after downloading the data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from connectome_driver.utils.config import load_config

# Columns whose value counts are worth seeing in full (or as a top-N list).
CATEGORICAL = ["flow", "super_class", "class", "sub_class", "nt_type", "neuropil", "side", "nerve"]


def describe(path: Path, max_levels: int) -> None:
    print(f"\n{'=' * 80}\n{path.name}  ({path.stat().st_size / 1e6:.1f} MB)\n{'=' * 80}")
    df = pd.read_csv(path)
    print(f"rows: {len(df):,}   columns: {len(df.columns)}")
    print("\ndtypes:")
    print(df.dtypes.to_string())
    print("\nhead:")
    print(df.head(5).to_string())
    print("\nmissing values per column:")
    print(df.isna().sum().to_string())

    for col in CATEGORICAL:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        print(f"\n[{col}] {len(counts)} distinct values")
        print(counts.head(max_levels).to_string())

    for col in df.columns:
        if "syn" in col and pd.api.types.is_numeric_dtype(df[col]):
            print(f"\n[{col}] summary")
            print(df[col].describe().to_string())

    if "root_id" in df.columns:
        print(f"\nunique root_id: {df['root_id'].nunique():,}")
    if {"pre_root_id", "post_root_id"} <= set(df.columns):
        pairs = df[["pre_root_id", "post_root_id"]].drop_duplicates()
        print(f"\nunique (pre, post) pairs: {len(pairs):,}")
        print(f"autapse rows (pre == post): {(df['pre_root_id'] == df['post_root_id']).sum():,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--max-levels", type=int, default=40)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg.raw_dir)
    for name in cfg.files.values():
        path = raw_dir / name
        if path.exists():
            describe(path, args.max_levels)
        else:
            print(f"\nMISSING: {path}")


if __name__ == "__main__":
    main()
