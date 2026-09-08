"""Checks for the grid-search design and optimizer selection."""

from collections import Counter

import torch

from torch.utils.data import TensorDataset

from comparison.models import build_pcn
from gridsearch.search import N_SPLITS, _build_optimizer, cross_validate
from gridsearch.space import OPTIMIZERS, sample_configs, select_configs


def test_sobol_draws_n_samples_per_optimizer_and_is_reproducible():
    configs = sample_configs(n_samples=7, seed=123)

    assert len(configs) == 7 * len(OPTIMIZERS)
    assert Counter(config["optimizer"] for config in configs) == {optimizer: 7 for optimizer in OPTIMIZERS}
    assert configs == sample_configs(n_samples=7, seed=123)


def test_one_array_task_per_optimizer_covers_the_design_exactly_once():
    tasks = [select_configs(7, 123, optimizer=optimizer) for optimizer in OPTIMIZERS]
    ids = [config_id for task in tasks for config_id, _ in task]

    assert [len(task) for task in tasks] == [7] * len(OPTIMIZERS)
    assert len(set(ids)) == len(ids), "a config must not run in two array tasks"
    assert sorted(ids) == list(range(7 * len(OPTIMIZERS))), "the tasks must cover every config"


def test_config_ids_are_global_so_task_rows_join_across_optimizers():
    configs = sample_configs(7, 123)

    for optimizer in OPTIMIZERS:
        for config_id, config in select_configs(7, 123, optimizer=optimizer):
            assert config["optimizer"] == optimizer
            assert configs[config_id] == config


def test_muon_is_not_silently_treated_as_sgd():
    model = build_pcn((8, 4), output_dim=3)
    optimizer = _build_optimizer({"optimizer": "muon", "lr": 1e-3, "weight_decay": 1e-2}, model)

    assert isinstance(optimizer, torch.optim.Muon)


def test_unknown_optimizer_is_rejected():
    model = build_pcn((8, 4), output_dim=3)

    try:
        _build_optimizer({"optimizer": "not-an-optimizer", "lr": 1e-3, "weight_decay": 1e-2}, model)
    except ValueError as error:
        assert "Unknown optimizer" in str(error)
    else:
        raise AssertionError("Unknown optimizers must not fall back to SGD")


def test_each_fold_stops_after_its_own_patience_runs_out():
    torch.manual_seed(0)
    dataset = TensorDataset(torch.rand(60, 784), torch.randint(0, 10, (60,)))
    config = {
        "eta_infer": 0.05, "T_infer": 3, "lr": 1e-3, "weight_decay": 1e-3,
        "negative_slope": 0.1, "batch_size": 16, "hidden_width": 32,
        "n_hidden_layers": 1, "optimizer": "adam",
    }
    num_epochs, patience = 8, 2

    rows = cross_validate(config, dataset, num_epochs, "cpu", seed=0, patience=patience)

    assert len(rows) == N_SPLITS
    for row in rows:
        assert 1 <= row["best_epoch"] <= row["epochs_run"] <= num_epochs
        assert row["epochs_run"] == min(num_epochs, row["best_epoch"] + patience)
    assert any(row["epochs_run"] < num_epochs for row in rows), "no fold stopped early"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
