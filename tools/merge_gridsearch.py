#!/usr/bin/env python3
"""Merge the per-optimizer grid-search CSVs into folds.csv and results.csv.

    uv run tools/merge_gridsearch.py [out-dir]

Run once the Slurm array from cluster/run_pcn_gridsearch.slurm has finished.
Parts are read oldest-first and de-duplicated on row identity, so re-running
one optimizer supersedes its rows instead of double-counting them and skewing
the ranking.
"""

import sys
from pathlib import Path

import pandas as pd

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/gridsearch")

# (stem, row identity, sort column, ascending)
for stem, identity, sort_key, ascending in (
    ("folds", ["config_id", "fold"], "config_id", True),
    ("results", ["config_id"], "mean_val_accuracy", False),
):
    parts = sorted(out_dir.glob(f"{stem}_*.csv"), key=lambda part: part.stat().st_mtime)
    if not parts:
        raise SystemExit(f"no {stem}_*.csv files in {out_dir}")

    table = pd.concat([pd.read_csv(part) for part in parts], ignore_index=True)
    merged = table.drop_duplicates(identity, keep="last").sort_values(sort_key, ascending=ascending)
    merged.reset_index(drop=True).to_csv(out_dir / f"{stem}.csv", index=False)

    superseded = f", {len(table) - len(merged)} superseded" if len(merged) < len(table) else ""
    print(f"{len(parts)} parts -> {len(merged)} rows{superseded} in {out_dir / f'{stem}.csv'}")
