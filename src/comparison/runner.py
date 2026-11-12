"""Single-trial training + evaluation for the PCN vs. MLP comparison."""

from __future__ import annotations

import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

from pcn.model import _f1_from_confusion

from .data import build_mnist_loaders
from .models import build_mlp, build_optimizer, build_pcn, count_params
from .seeding import seed_everything

# Validation metric the early stop watches, for both models. The comparison is
# reported on accuracy, so stopping on anything else would select epochs
# against a different target.
_MONITOR = "accuracy"


def _mlp_metrics(model, loader, device):
    """Same metric set/definitions as ``PredictiveCodingNetwork.evaluate``, so
    PCN and MLP results are directly comparable (``pcn.model._f1_from_confusion``
    is reused rather than reimplemented)."""
    model.eval()
    n, correct, ce_sum, confusion = 0, 0, 0.0, None
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            ce_sum += F.cross_entropy(logits, y, reduction="sum").item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            n += y.numel()
            num_classes = logits.shape[1]
            counts = torch.bincount(
                y * num_classes + preds, minlength=num_classes**2
            ).reshape(num_classes, num_classes)
            confusion = counts if confusion is None else confusion + counts

    metrics = {"n": n, "accuracy": correct / max(1, n), "cross_entropy": ce_sum / max(1, n)}
    metrics.update(_f1_from_confusion(confusion))
    return metrics


def train_with_early_stopping(
    model, train_epoch, evaluate, num_epochs, patience, eval_train=None
):
    """Run epochs until validation accuracy stops improving, keeping the best weights.

    ``train_epoch`` trains one epoch, ``evaluate`` scores the validation split,
    and ``eval_train`` (optional) scores the train split for the history curve.
    Both models go through this same function, so neither gets a stopping rule
    the other did not.

    Training stops once ``patience`` consecutive epochs have failed to beat the
    best validation accuracy so far. The model is then restored to that best
    epoch's weights, which is the model the test split gets scored on. A run
    that diverges scores NaN, which never compares greater, so it runs out its
    patience and stops rather than burning the full budget.

    Returns ``(history, summary)``. ``history`` holds the per-epoch metrics;
    ``summary`` carries ``best_epoch``, ``epochs_run``, and the wall time split
    into fitting (``train_time_s``) and scoring (``eval_time_s``).
    """
    history = {"epoch": [], "train": [], "validation": []}
    best_metrics, best_state, best_epoch, stale = None, None, 0, 0
    train_time_s, eval_time_s = 0.0, 0.0
    epoch = 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.perf_counter()
        train_epoch()
        train_time_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        metrics = evaluate()
        if eval_train is not None:
            history["train"].append(eval_train())
        eval_time_s += time.perf_counter() - t0

        history["epoch"].append(epoch)
        history["validation"].append(metrics)

        if best_metrics is None or metrics[_MONITOR] > best_metrics[_MONITOR]:
            best_metrics, best_epoch, stale = metrics, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return history, {
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "train_time_s": train_time_s,
        "eval_time_s": eval_time_s,
    }


def run_pcn_trial(cfg, seed, device):
    """Train and test one PCN on MNIST for this seed. Returns a result dict."""
    pcn = cfg.pcn
    generator = seed_everything(seed)
    train_loader, val_loader, test_loader = build_mnist_loaders(
        cfg.data_dir, pcn.batch_size, cfg.val_size, generator
    )

    model = build_pcn(pcn.dims, cfg.output_dim, negative_slope=pcn.negative_slope).to(device)
    optimizer = build_optimizer(pcn.optimizer, model.parameters(), pcn.lr, pcn.weight_decay)
    infer = {"eta_infer": pcn.eta_infer, "T_infer": pcn.T_infer, "device": device}

    def train_epoch():
        # One epoch at a time so the early stop owns the loop. ``fit`` keeps no
        # state between calls apart from the model and the optimizer it is
        # handed, so N calls of one epoch train exactly as one call of N would.
        model.fit(
            train_loader,
            num_epochs=1,
            optimizer=optimizer,
            record_history=False,
            progress=False,
            **infer,
        )

    history, summary = train_with_early_stopping(
        model,
        train_epoch,
        lambda: model.evaluate(val_loader, **infer),
        pcn.num_epochs,
        pcn.patience,
        eval_train=(lambda: model.evaluate(train_loader, **infer)) if cfg.eval_train else None,
    )

    t0 = time.perf_counter()
    test_metrics = model.evaluate(test_loader, **infer)
    test_time_s = time.perf_counter() - t0

    return _result("pcn", asdict(pcn), model, seed, device, history, summary, test_metrics, test_time_s)


def run_mlp_trial(cfg, seed, device):
    """Train and test one MLP on MNIST for this seed. Returns a result dict."""
    mlp = cfg.mlp
    generator = seed_everything(seed)
    train_loader, val_loader, test_loader = build_mnist_loaders(
        cfg.data_dir, mlp.batch_size, cfg.val_size, generator
    )

    model = build_mlp(mlp.dims, cfg.output_dim, dropout=mlp.dropout).to(device)
    optimizer = build_optimizer(mlp.optimizer, model.parameters(), mlp.lr, mlp.weight_decay)

    def train_epoch():
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            optimizer.step()

    history, summary = train_with_early_stopping(
        model,
        train_epoch,
        lambda: _mlp_metrics(model, val_loader, device),
        mlp.num_epochs,
        mlp.patience,
        eval_train=(lambda: _mlp_metrics(model, train_loader, device)) if cfg.eval_train else None,
    )

    t0 = time.perf_counter()
    test_metrics = _mlp_metrics(model, test_loader, device)
    test_time_s = time.perf_counter() - t0

    return _result("mlp", asdict(mlp), model, seed, device, history, summary, test_metrics, test_time_s)


def _result(name, config, model, seed, device, history, summary, test_metrics, test_time_s):
    """Assemble the trial record both trials return and ``log_trial`` writes out.

    ``train_time_s`` is fitting alone. Per-epoch validation (and the optional
    train scoring) sits in ``eval_time_s``, so a timing comparison between the
    two models can use the cost of training or the cost of the whole trial
    without one silently including the other.
    """
    return {
        "model": name,
        "seed": seed,
        "device": str(device),
        "config": config,
        "n_params": count_params(model),
        **summary,
        "test_time_s": test_time_s,
        "total_time_s": summary["train_time_s"] + summary["eval_time_s"] + test_time_s,
        "test": test_metrics,
        "history": history,
    }
