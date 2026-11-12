"""Result logging: a flat CSV of per-trial summaries plus one full JSON per trial."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

# The hyperparameters differ per model now, so the summary row carries the ones
# a reader of results.csv will want to group or filter by. The rest of the
# config (and every metric) is in the per-trial JSON.
_CSV_COLUMNS = [
    "run_idx", "model", "seed", "timestamp", "device", "n_params",
    "optimizer", "lr", "batch_size", "best_epoch", "epochs_run",
    "train_time_s", "eval_time_s", "test_time_s", "total_time_s",
    "test_accuracy", "test_cross_entropy", "test_f1_macro", "test_n",
]


def log_trial(result: dict, run_idx: int, out_dir: Path) -> Path:
    """Append the trial's summary row to ``out_dir/results.csv`` and write its
    full detail (the model's config, complete test metrics, epoch history) to
    ``out_dir/runs/<model>_run<run_idx>_seed<seed>.json``. Returns the JSON path.
    """
    out_dir = Path(out_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    test, config = result["test"], result["config"]
    row = {
        "run_idx": run_idx,
        "model": result["model"],
        "seed": result["seed"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": result["device"],
        "n_params": result["n_params"],
        "optimizer": config["optimizer"],
        "lr": config["lr"],
        "batch_size": config["batch_size"],
        "best_epoch": result["best_epoch"],
        "epochs_run": result["epochs_run"],
        "train_time_s": result["train_time_s"],
        "eval_time_s": result["eval_time_s"],
        "test_time_s": result["test_time_s"],
        "total_time_s": result["total_time_s"],
        "test_accuracy": test.get("accuracy"),
        "test_cross_entropy": test.get("cross_entropy"),
        "test_f1_macro": test.get("f1_macro"),
        "test_n": test.get("n"),
    }

    csv_path = out_dir / "results.csv"
    _check_header(csv_path)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)

    json_path = runs_dir / f"{result['model']}_run{run_idx:02d}_seed{result['seed']}.json"
    with json_path.open("w") as f:
        json.dump({**row, "config": config, "test_full": test, "history": result["history"]}, f, indent=2)

    return json_path


def _check_header(csv_path: Path) -> None:
    """Refuse to append to a table written with different columns.

    Appending regardless would file each value under whatever column happened
    to sit in that position, and the mismatch would only show up much later as
    nonsense in the analysis. Move or delete the old table instead.
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with csv_path.open(newline="") as f:
        header = next(csv.reader(f), [])
    if header != _CSV_COLUMNS:
        raise ValueError(
            f"{csv_path} was written with different columns: {header}. "
            f"Move it aside (or point --out-dir elsewhere) before running again."
        )
