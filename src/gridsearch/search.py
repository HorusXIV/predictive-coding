"""5-fold cross-validated scoring of one PCN hyperparameter configuration."""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from comparison.models import build_pcn, count_params
from comparison.seeding import seed_everything

from .space import dims_for_config

INPUT_DIM = 784
OUTPUT_DIM = 10
N_SPLITS = 5

# Columns that describe a fold rather than measure it, so aggregation skips them.
_NON_METRIC_KEYS = {"fold", "n_val", "n_params"}

# Validation metric the early stop watches. Accuracy is what the search ranks
# on, so stopping on anything else would optimise for a different target.
_MONITOR = "accuracy"


def _build_optimizer(config, model):
    optimizer = config["optimizer"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    if optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    if optimizer == "muon":
        return torch.optim.Muon(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    raise ValueError(f"Unknown optimizer: {optimizer!r}")


def cross_validate(config, dataset, num_epochs, device, seed, patience) -> list[dict]:
    """Train and validate a PCN built from ``config`` on each of 5 folds.

    Returns one row per fold, carrying every metric
    ``PredictiveCodingNetwork.evaluate`` reports (prefixed ``val_``) plus fit
    and evaluation timings. Nothing is averaged away here, so the caller keeps
    the raw fold-level distribution for later modelling instead of only a mean
    and a standard deviation.

    Each fold trains for at most ``num_epochs`` and stops once its own
    validation accuracy has failed to improve for ``patience`` consecutive
    epochs, so a fold that converges early stops without waiting for the
    others. The row reports the best epoch's metrics rather than the last
    epoch's, which is the score a restored best-weights checkpoint would give.
    Selecting the epoch on the same fold the row reports makes ``val_accuracy``
    mildly optimistic; that is inherent to early stopping on a validation
    split, and every config pays it equally.

    ``dataset`` must already be the train split; this function never touches a
    test set. ``seed`` fixes the fold split and each fold's init/data order.
    Passing the same seed for every sampled config (as
    :mod:`gridsearch.run_gridsearch` does) puts all configs on identical
    folds, so differences between them come from the hyperparameters.
    """
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    dims = dims_for_config(config, INPUT_DIM)

    fold_rows = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dataset)))):
        generator = seed_everything(seed * 100 + fold_idx)

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=config["batch_size"],
            shuffle=True,
            generator=generator,
        )
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=256, shuffle=False)

        model = build_pcn(dims, OUTPUT_DIM, negative_slope=config["negative_slope"]).to(device)
        optimizer = _build_optimizer(config, model)

        fit_time_s = eval_time_s = 0.0
        best_metrics, best_epoch, stale = None, 0, 0

        for epoch in range(1, num_epochs + 1):
            # One epoch at a time: repeated single-epoch calls advance the
            # loader and optimizer exactly as one num_epochs call would.
            t0 = time.perf_counter()
            model.fit(
                train_loader,
                num_epochs=1,
                eta_infer=config["eta_infer"],
                T_infer=config["T_infer"],
                optimizer=optimizer,
                device=device,
                record_history=False,
                progress=False,
            )
            fit_time_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            metrics = model.evaluate(
                val_loader, eta_infer=config["eta_infer"], T_infer=config["T_infer"], device=device
            )
            eval_time_s += time.perf_counter() - t0

            # A diverged fold scores NaN, which never compares greater, so it
            # runs out its patience and stops instead of burning every epoch.
            if best_metrics is None or metrics[_MONITOR] > best_metrics[_MONITOR]:
                best_metrics, best_epoch, stale = metrics, epoch, 0
            else:
                stale += 1
                if stale >= patience:
                    break

        fold_rows.append({
            "fold": fold_idx,
            "n_val": best_metrics.pop("n"),
            "n_params": count_params(model),
            **{f"val_{k}": v for k, v in best_metrics.items()},
            "best_epoch": best_epoch,
            "epochs_run": epoch,
            "fit_time_s": fit_time_s,
            "eval_time_s": eval_time_s,
            "total_time_s": fit_time_s + eval_time_s,
        })

    return fold_rows


def aggregate(fold_rows: list[dict]) -> dict:
    """Reduce per-fold rows to mean/std/min/max per metric.

    Every numeric metric present in the folds is summarised, so accuracy, the
    F1 scores, cross-entropy, the free energies and the timings all get the
    same treatment and none of them is left with a mean but no spread.
    """
    summary = {"n_folds": len(fold_rows), "n_params": fold_rows[0]["n_params"]}
    for key in fold_rows[0]:
        if key in _NON_METRIC_KEYS:
            continue
        values = np.array([row[key] for row in fold_rows], dtype=float)
        summary[f"mean_{key}"] = float(values.mean())
        summary[f"std_{key}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[f"min_{key}"] = float(values.min())
        summary[f"max_{key}"] = float(values.max())
    return summary
