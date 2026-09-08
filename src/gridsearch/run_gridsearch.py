"""Search PCN hyperparameters with a scrambled-Sobol space-filling design,
score each candidate by 5-fold cross-validation on the MNIST *train* split,
and write both a per-fold table and a summary table.

Each fold trains for up to ``--num-epochs`` and stops once its own validation
accuracy has gone ``--patience`` epochs without improving, so folds that
converge early stop early. The fold row reports its best epoch.

``--n-samples`` is the number of configurations *per optimizer*. With the
three configured optimizers, the command evaluates three times that many
configurations in total. Powers of two keep the Sobol balance properties.

Test data is never loaded by this command; see :mod:`gridsearch.data`. Every
candidate is scored on the same folds with the same per-fold seeding (see
:func:`gridsearch.search.cross_validate`), so the ranking reflects the
hyperparameters rather than which candidate drew easier folds.

Two files land in ``out/gridsearch/``:

``folds.csv``
    One row per (config, fold), with every validation metric and timing left
    raw. This is the long-format table to model from: real fold-level
    observations, rather than points resampled from an assumed normal.
``results.csv``
    One row per config, with mean, std, min and max for each metric, sorted
    by mean validation accuracy.

Both carry a ``config_id`` column, so a summary row joins back onto its folds.

``--optimizer`` runs one optimizer's configurations, which is how a Slurm job
array splits the search: one task per optimizer, defaulting to
``OPTIMIZERS[SLURM_ARRAY_TASK_ID]`` so the array needs no arguments. Each task
writes its own ``folds_<optimizer>.csv`` and ``results_<optimizer>.csv``;
``tools/merge_gridsearch.py`` folds those into the two tables above once the
array is done.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .data import load_mnist_train
from .search import aggregate, cross_validate
from .space import OPTIMIZERS, dims_for_config, select_configs

REPO_ROOT = Path(__file__).resolve().parents[2]


def _optimizer_for_array_task():
    """Map this array task to an optimizer, so ``--array=0-2`` takes no arguments."""
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        return None
    if int(task_id) >= len(OPTIMIZERS):
        raise SystemExit(
            f"array task {task_id} has no optimizer; "
            f"{len(OPTIMIZERS)} optimizers means --array=0-{len(OPTIMIZERS) - 1}"
        )
    return OPTIMIZERS[int(task_id)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "out" / "gridsearch")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZERS,
        default=_optimizer_for_array_task(),
        help="evaluate only this optimizer's configs (defaults to this array task's optimizer)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = select_configs(args.n_samples, args.seed, args.optimizer)

    # A whole-design run keeps writing the two tables directly.
    suffix = f"_{args.optimizer}" if args.optimizer else ""

    # An array too short to cover OPTIMIZERS silently drops one, which only
    # shows up as a missing chunk of the merged table hours later.
    tasks = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if tasks is not None and int(tasks) != len(OPTIMIZERS):
        print(f"warning: --array covers {tasks} tasks but there are {len(OPTIMIZERS)} optimizers")

    dataset = load_mnist_train(args.data_dir)
    print(
        f"device={args.device} n_samples_per_optimizer={args.n_samples} "
        f"optimizer={args.optimizer or 'all'} configs={len(configs)} "
        f"num_epochs={args.num_epochs} patience={args.patience} "
        f"train_size={len(dataset)} (5-fold CV, train data only)"
    )

    fold_frames, summary_rows = [], []
    for done, (config_id, config) in enumerate(tqdm(configs, desc="configs", unit="cfg"), 1):
        fold_rows = cross_validate(
            config, dataset, args.num_epochs, args.device, args.seed, args.patience
        )

        labelled = {
            "config_id": config_id,
            "dims": str(dims_for_config(config, 784)),
            **config,
        }
        fold_frames.append(pd.DataFrame([{**labelled, **row} for row in fold_rows]))
        summary_rows.append({**labelled, **aggregate(fold_rows)})

        summary = summary_rows[-1]
        print(
            f"[{done}/{len(configs)}] config_id={config_id} {config}\n"
            f"  val_accuracy={summary['mean_val_accuracy']:.4f} +/- {summary['std_val_accuracy']:.4f}"
            f"  val_f1_macro={summary['mean_val_f1_macro']:.4f} +/- {summary['std_val_f1_macro']:.4f}"
            f"  best_epoch={summary['mean_best_epoch']:.1f}"
            f"  fit={summary['mean_fit_time_s']:.1f}s"
        )

        # Written every config, so an interrupted search keeps what it finished.
        folds = pd.concat(fold_frames, ignore_index=True)
        folds.to_csv(args.out_dir / f"folds{suffix}.csv", index=False)
        table = (
            pd.DataFrame(summary_rows)
            .sort_values("mean_val_accuracy", ascending=False)
            .reset_index(drop=True)
        )
        table.to_csv(args.out_dir / f"results{suffix}.csv", index=False)

    print(f"\nbest config:\n{table.iloc[0]}")
    print(f"\n{len(folds)} fold rows -> {args.out_dir / f'folds{suffix}.csv'}")
    print(f"{len(table)} config rows -> {args.out_dir / f'results{suffix}.csv'}")


if __name__ == "__main__":
    main()
