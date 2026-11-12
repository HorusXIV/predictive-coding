"""Train and test a PCN and an MLP ``n_runs`` times each on MNIST, logging
every trial to ``out/`` (see :mod:`comparison.logging_utils`).

The two models are trained under their own hyperparameters: the PCN under the
grid search's winner, the MLP under a published MNIST baseline (both in
:mod:`comparison.config`, which is also where to change them). Giving both the
same settings, as this script used to, only measured which mechanism happened
to suit one arbitrary configuration.

What stays paired is the data. Each run index gets its own seed, and that seed
drives both trials: the same train/val split and the same image order, drawn
before either model touches it. So the 10 paired runs support a paired test on
accuracy, or on ``total_time_s``, between the two model groups of
``out/results.csv``.

Both models train until validation accuracy has gone ``patience`` epochs
without improving, then fall back to their best epoch's weights before the test
split is scored. Each model's own patience and epoch cap live in its config,
because an MLP epoch costs seconds and a PCN epoch costs minutes.

Nothing here forces CUDA/cuDNN into deterministic mode, so GPU training keeps
its own run-to-run kernel-selection noise on top of the seeded randomness --
by design, since that is itself one of the sources of variation these repeated
runs are meant to capture.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch
from tqdm import tqdm

from .config import ExperimentConfig, pcn_config_from_gridsearch
from .logging_utils import log_trial
from .runner import run_mlp_trial, run_pcn_trial

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "out")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--pcn-from-gridsearch",
        type=Path,
        metavar="RESULTS_CSV",
        help="take the PCN hyperparameters from a grid-search results table "
             "instead of comparison.config (default: its best-scoring row)",
    )
    parser.add_argument(
        "--pcn-config-id",
        type=int,
        help="with --pcn-from-gridsearch, use this config_id rather than the best row",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.pcn_from_gridsearch:
        cfg = dataclasses.replace(
            cfg, pcn=pcn_config_from_gridsearch(args.pcn_from_gridsearch, args.pcn_config_id)
        )
    elif args.pcn_config_id is not None:
        parser.error("--pcn-config-id needs --pcn-from-gridsearch")

    print(
        f"device={args.device} n_runs={args.n_runs} base_seed={args.base_seed} "
        f"out_dir={args.out_dir} output_dim={cfg.output_dim} val_size={cfg.val_size} "
        f"eval_train={cfg.eval_train}\n"
        f"  pcn: {cfg.pcn}\n"
        f"  mlp: {cfg.mlp}"
    )

    for run_idx in tqdm(range(args.n_runs), desc="running trials", unit="run"):
        seed = args.base_seed + run_idx

        for trial in (run_pcn_trial, run_mlp_trial):
            result = trial(cfg, seed, args.device)
            log_trial(result, run_idx, args.out_dir)
            print(
                f"run {run_idx:02d} {result['model']} seed={seed} "
                f"accuracy={result['test']['accuracy']:.4f} "
                f"best_epoch={result['best_epoch']}/{result['epochs_run']} "
                f"time={result['total_time_s']:.1f}s"
            )

    print(f"done -- results in {args.out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
