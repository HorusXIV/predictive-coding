"""Hyperparameters for the PCN vs. MLP comparison, one setting per model.

Earlier versions ran both models off one shared config. That handicapped
whichever model the shared values suited less, and made the comparison a
statement about one arbitrary point in hyperparameter space rather than about
the two training mechanisms. Each model now gets the settings that work for it:

``PCNConfig``
    The winner of the grid search (see :mod:`gridsearch.run_gridsearch`), with
    its provenance recorded below. Edit the defaults after a new search, or
    load a row straight from a results table with
    :func:`pcn_config_from_gridsearch`.
``MLPConfig``
    A published MNIST baseline; the comment on each field names its source.

Everything the comparison holds fixed lives on ``ExperimentConfig``: the data,
the train/val split, the output dimension. Both models see the same images in
the same split under the same seed, and both stop on the same rule, so what is
left to differ is the training mechanism and the hyperparameters each one was
given.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

# --------------------------------------------------------------------------
# Provenance of the PCN defaults below. Update this block together with the
# values, so a result in out/ can always be traced back to the search that
# produced its hyperparameters.
#
#   source      out/gridsearch/results.csv, search of 2026-09-08
#   design      351 configs (Sobol, 128 per optimizer), 5-fold CV on the
#               MNIST train split, up to 10 epochs, patience 4
#   winner      config_id=70
#   score       mean val accuracy 0.8702 +/- 0.0050, f1_macro 0.8683
#   note        mean best epoch 9.4 of a 10-epoch cap, i.e. the config was
#               still improving when the search cut it off, which is why
#               num_epochs here is well above what the search allowed.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PCNConfig:
    """Predictive-coding network: architecture, inference, and optimizer."""

    dims: tuple[int, ...] = (784, 968, 484)
    negative_slope: float = 0.0158762522780799
    eta_infer: float = 0.0175371115385876
    T_infer: int = 23
    optimizer: str = "muon"
    lr: float = 0.000129956114041
    weight_decay: float = 0.0055186641323977
    batch_size: int = 124
    num_epochs: int = 50
    patience: int = 4



@dataclass(frozen=True)
class MLPConfig:
    """Backprop MLP, set up the way the MNIST literature sets one up.

    The classic Keras ``mnist_mlp.py`` example reaches 98.40% test accuracy
    with two hidden layers, dropout 0.2 between them, and batch size 128;
    published MLP baselines that swap its RMSprop for Adam at lr=1e-3 land in
    the same 98% range with hidden widths of 512/256. Those are the values
    here. Dropout and the early stop do the regularizing, so weight decay
    stays at 0 as in the reference recipes.

    Unlike the PCN, this network carries bias terms and plain ReLU: there is no
    relaxation whose stability depends on the leak, and nothing to gain from
    matching the PCN's parameter count now that the two configs differ anyway.
    """

    dims: tuple[int, ...] = (784, 512, 256)
    dropout: float = 0.2
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    # An MLP epoch on MNIST costs seconds, so the cap is generous and the
    # early stop decides where training actually ends.
    num_epochs: int = 50
    patience: int = 5


@dataclass(frozen=True)
class ExperimentConfig:
    """What the two models share, plus the two per-model configs."""

    pcn: PCNConfig = field(default_factory=PCNConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    output_dim: int = 10
    val_size: int = 5000
    data_dir: str = "data"
    # Scoring the full train split every epoch is what gives the notebooks
    # their train curves and the train/val gap. For the PCN it costs about as
    # much as the epoch itself, since every train image has to be relaxed
    # again label-free. Set False to buy that time back.
    eval_train: bool = True


# Grid-search column -> PCNConfig field. Only these columns are read, so extra
# metric columns in a results table are ignored.
_GRIDSEARCH_FIELDS = (
    "eta_infer", "T_infer", "lr", "weight_decay", "negative_slope", "batch_size", "optimizer",
)


def pcn_config_from_gridsearch(results_csv, config_id=None, **overrides) -> PCNConfig:
    """Build a :class:`PCNConfig` from one row of a grid-search results table.

    Reads ``out/gridsearch/results.csv`` (or a per-optimizer table) and takes
    the row with the highest ``mean_val_accuracy``, or the row whose
    ``config_id`` is given. ``num_epochs`` and ``patience`` are not searched
    over, so they come from the defaults above unless passed in ``overrides``.

    This is the shortcut for trying a fresh search's winner without editing
    code. Once a config is settled on, write it into :class:`PCNConfig` so that
    a run no longer depends on a file in ``out/`` that the next search
    overwrites.
    """
    with Path(results_csv).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{results_csv} has no rows")

    if config_id is None:
        row = max(rows, key=lambda r: float(r["mean_val_accuracy"]))
    else:
        row = next((r for r in rows if int(r["config_id"]) == config_id), None)
        if row is None:
            raise ValueError(f"{results_csv} has no config_id={config_id}")

    types = {f.name: f.type for f in fields(PCNConfig)}
    values = {"dims": tuple(ast.literal_eval(row["dims"]))}
    for name in _GRIDSEARCH_FIELDS:
        values[name] = row[name] if types[name] == "str" else _number(row[name], types[name])
    return replace(PCNConfig(), **values, **overrides)


def _number(text, type_name):
    return int(text) if type_name == "int" else float(text)
