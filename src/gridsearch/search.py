"""5-fold cross-validated scoring of one PCN hyperparameter configuration."""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from comparison.models import build_pcn, count_params
from comparison.seeding import seed_everything
from pcn.batched import fit_epoch

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

    The folds train together rather than one after another: they share an
    architecture, so :func:`pcn.batched.fit_epoch` stacks them and spends one
    set of kernel launches on all five. Each fold still owns its weights, its
    optimizer and its data, so the scores match training them separately.

    Each fold stops once its own validation accuracy has failed to improve for
    ``patience`` consecutive epochs, and drops out of the stack at that point
    while the others carry on. The row reports the best epoch's metrics rather
    than the last epoch's, which is the score a restored best-weights
    checkpoint would give. Selecting the epoch on the same fold the row reports
    makes ``val_accuracy`` mildly optimistic; that is inherent to early stopping
    on a validation split, and every config pays it equally.

    ``fit_time_s`` is a shared epoch cost split evenly across the folds
    training in it, so the five rows still sum to the wall time spent fitting.

    ``dataset`` must already be the train split; this function never touches a
    test set. ``seed`` fixes the fold split and each fold's init/data order.
    Passing the same seed for every sampled config (as
    :mod:`gridsearch.run_gridsearch` does) puts all configs on identical
    folds, so differences between them come from the hyperparameters.
    """
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    dims = dims_for_config(config, INPUT_DIM)

    models, optimizers, train_loaders, val_loaders = [], [], [], []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dataset)))):
        # Seeded per fold before its weights are drawn, so a fold gets the same
        # init and the same batch order whether or not it shares a stack.
        generator = seed_everything(seed * 100 + fold_idx)

        train_loaders.append(DataLoader(
            Subset(dataset, train_idx),
            batch_size=config["batch_size"],
            shuffle=True,
            generator=generator,
        ))
        # 4096: evaluation only accumulates per-batch totals, so a wide batch
        # changes nothing but float accumulation order, and the GPU is
        # launch-bound below it.
        val_loaders.append(DataLoader(Subset(dataset, val_idx), batch_size=4096, shuffle=False))

        model = build_pcn(dims, OUTPUT_DIM, negative_slope=config["negative_slope"]).to(device)
        models.append(model)
        optimizers.append(_build_optimizer(config, model))

    best = [None] * N_SPLITS
    best_epoch, epochs_run, stale = [0] * N_SPLITS, [0] * N_SPLITS, [0] * N_SPLITS
    fit_time_s, eval_time_s = [0.0] * N_SPLITS, [0.0] * N_SPLITS
    training = list(range(N_SPLITS))

    for epoch in range(1, num_epochs + 1):
        if not training:
            break

        t0 = time.perf_counter()
        fit_epoch(
            [models[f] for f in training],
            [optimizers[f] for f in training],
            [train_loaders[f] for f in training],
            eta_infer=config["eta_infer"],
            T_infer=config["T_infer"],
            slope=config["negative_slope"],
            device=device,
        )
        shared = (time.perf_counter() - t0) / len(training)

        for fold_idx in list(training):
            fit_time_s[fold_idx] += shared

            t0 = time.perf_counter()
            metrics = models[fold_idx].evaluate(
                val_loaders[fold_idx],
                eta_infer=config["eta_infer"],
                T_infer=config["T_infer"],
                device=device,
            )
            eval_time_s[fold_idx] += time.perf_counter() - t0
            epochs_run[fold_idx] = epoch

            # A diverged fold scores NaN, which never compares greater, so it
            # runs out its patience and leaves instead of burning every epoch.
            if best[fold_idx] is None or metrics[_MONITOR] > best[fold_idx][_MONITOR]:
                best[fold_idx], best_epoch[fold_idx], stale[fold_idx] = metrics, epoch, 0
            else:
                stale[fold_idx] += 1
                if stale[fold_idx] >= patience:
                    training.remove(fold_idx)

    return [{
        "fold": fold_idx,
        "n_val": best[fold_idx].pop("n"),
        "n_params": count_params(models[fold_idx]),
        **{f"val_{k}": v for k, v in best[fold_idx].items()},
        "best_epoch": best_epoch[fold_idx],
        "epochs_run": epochs_run[fold_idx],
        "fit_time_s": fit_time_s[fold_idx],
        "eval_time_s": eval_time_s[fold_idx],
        "total_time_s": fit_time_s[fold_idx] + eval_time_s[fold_idx],
    } for fold_idx in range(N_SPLITS)]


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
