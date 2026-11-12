"""Checks for the PCN vs. MLP comparison: early stopping, config loading, logging."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from comparison.config import MLPConfig, PCNConfig, pcn_config_from_gridsearch
from comparison.logging_utils import log_trial
from comparison.models import build_mlp
from comparison.runner import train_with_early_stopping

RESULTS_HEADER = (
    "config_id,dims,eta_infer,T_infer,lr,weight_decay,negative_slope,batch_size,"
    "hidden_width,n_hidden_layers,optimizer,mean_val_accuracy\n"
)
RESULTS_ROWS = (
    "7,\"(784, 256, 128)\",0.02,40,0.001,0.01,0.05,64,256,2,adam,0.83\n"
    "3,\"(784, 512)\",0.1,25,0.005,0.2,0.2,128,512,1,muon,0.91\n"
)


def _scores(*accuracies):
    """A stand-in for a validation pass: hands back the given accuracies in order."""
    remaining = list(accuracies)
    return lambda: {"accuracy": remaining.pop(0)}


def test_training_stops_once_patience_runs_out():
    model = build_mlp((4, 3), 2)
    validate = _scores(0.5, 0.7, 0.6, 0.65, 0.4)

    history, summary = train_with_early_stopping(
        model, lambda: None, validate, num_epochs=10, patience=3
    )

    assert summary["best_epoch"] == 2, "the best epoch is the one with the highest accuracy"
    assert summary["epochs_run"] == 5, "three stale epochs after the best one, then stop"
    assert history["epoch"] == [1, 2, 3, 4, 5]
    assert history["train"] == [], "train scoring is off unless eval_train is passed"


def test_the_best_epochs_weights_are_what_gets_tested():
    """Early stopping is worth nothing if the tested model is the last epoch's."""
    model = build_mlp((4, 3), 2)
    snapshots = []

    def train_epoch():
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
        snapshots.append({k: v.clone() for k, v in model.state_dict().items()})

    train_with_early_stopping(
        model, train_epoch, _scores(0.5, 0.9, 0.6, 0.6), num_epochs=4, patience=2
    )

    best = snapshots[1]  # epoch 2 scored highest
    for key, value in model.state_dict().items():
        assert torch.equal(value, best[key]), f"{key} is not the best epoch's"


def test_a_diverged_run_stops_instead_of_burning_the_budget():
    model = build_mlp((4, 3), 2)
    nan = float("nan")

    history, summary = train_with_early_stopping(
        model, lambda: None, _scores(nan, nan, nan, nan), num_epochs=50, patience=2
    )

    assert summary["epochs_run"] == 3, "NaN never improves, so patience runs out"
    assert summary["best_epoch"] == 1, "the first epoch is kept, so there are weights to test"
    assert len(history["validation"]) == 3


def test_gridsearch_row_becomes_a_pcn_config():
    with TemporaryDirectory() as tmp:
        results = Path(tmp) / "results.csv"
        results.write_text(RESULTS_HEADER + RESULTS_ROWS)

        best = pcn_config_from_gridsearch(results)
        picked = pcn_config_from_gridsearch(results, config_id=7, num_epochs=3)

    assert best.dims == (784, 512) and best.optimizer == "muon", "the best row wins, not the first"
    assert (best.T_infer, best.batch_size) == (25, 128), "ints stay ints"
    assert picked.dims == (784, 256, 128) and picked.optimizer == "adam"
    assert picked.num_epochs == 3, "an override beats the default"
    assert picked.patience == PCNConfig().patience, "unsearched fields keep their default"


def test_the_two_models_are_configured_apart():
    """The point of the rework: shared hyperparameters would hide the difference."""
    pcn, mlp = PCNConfig(), MLPConfig()

    assert (pcn.lr, pcn.batch_size, pcn.optimizer) != (mlp.lr, mlp.batch_size, mlp.optimizer)


def test_a_logged_trial_records_which_config_produced_it():
    result = {
        "model": "mlp", "seed": 1, "device": "cpu", "n_params": 42,
        "config": {"optimizer": "adam", "lr": 1e-3, "batch_size": 128, "dropout": 0.2},
        "best_epoch": 2, "epochs_run": 4,
        "train_time_s": 1.0, "eval_time_s": 0.5, "test_time_s": 0.25, "total_time_s": 1.75,
        "test": {"accuracy": 0.98, "cross_entropy": 0.06, "f1_macro": 0.97, "n": 10000},
        "history": {"epoch": [1, 2], "train": [], "validation": [{"accuracy": 0.9}, {"accuracy": 0.95}]},
    }

    with TemporaryDirectory() as tmp:
        json_path = log_trial(result, run_idx=0, out_dir=Path(tmp))
        row = next(iter(csv.DictReader((Path(tmp) / "results.csv").open())))
        record = json.loads(json_path.read_text())

    assert row["optimizer"] == "adam" and row["batch_size"] == "128"
    assert row["best_epoch"] == "2" and row["epochs_run"] == "4"
    assert record["config"]["dropout"] == 0.2, "the full config survives in the JSON"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
