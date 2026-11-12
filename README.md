# Predictive Coding Networks

A PyTorch implementation of a predictive-coding network (PCN) with a supervised readout, plus tooling to compare it against a separately tuned MLP and to search its hyperparameters.

## Setup

```bash
uv sync
```

## Layout

```
src/pcn/          the network, a 3D inference visualizer, and batched fold training
src/comparison/   PCN vs. a separately tuned MLP, run repeatedly on MNIST
src/gridsearch/   PCN hyperparameter search
test/             regression tests for src/pcn, the grid search, and the comparison
tools/            one-off scripts (merging the grid-search tables)
experiments/      notebooks: PCN on HerzCheck, MNIST, CIFAR-10
notebooks/        grid-search result analysis
docs/             engineering notes (GPU parallelisation)
cluster/          Slurm submission script
containers/       Singularity definition for the cluster environment
data/             datasets (gitignored)
ressources/       background papers (gitignored)
out/              results from comparison/ and gridsearch/ (gitignored)
```

## src/pcn

`PredictiveCodingNetwork` (in `model.py`) holds one weight matrix per generative layer, used both ways: top-down as `x_hat_l = f(W_l @ x_{l+1})` to predict the layer below, bottom-up through its error to drive inference above. Training alternates two phases per batch: relax every latent state with the input clamped, then take one weight step on the result. Gradients come from the analytic free-energy expressions rather than autograd, but land in `.grad`, so any `torch.optim` optimizer can run the weight phase.

`PCN3DVisualizer` (in `visualizer.py`) renders a recorded inference trajectory as a static 3D plot, a matplotlib animation, or an interactive Plotly figure.

Run the tests with:

```bash
uv run python test/test_pcn.py
```

## experiments/

Three notebooks, each with its own imports and its own data loading:

- `pcn_HerzCheck.ipynb`: heart-disease classification on `data/herzcheck_*.csv`.
- `pcn_MNIST.ipynb`: digit classification and class-conditional generation.
- `pcn_CIFAR.ipynb`: CIFAR-10 classification and class-conditional generation.

## src/comparison

The two models no longer share hyperparameters. `config.py` holds one dataclass each, and the comment above `PCNConfig` records the search that produced its values.

`PCNConfig` carries the grid search's winner (`config_id=70` of 351 candidates): Muon, `dims=(784, 727, 363)`, `T_infer=70`, `eta_infer=0.0104`, `lr=2.4e-4`, `weight_decay=0.091`, batch 49, 837,499 parameters, scoring 0.8702 +/- 0.0050 mean validation accuracy over the five folds.

`MLPConfig` carries a published MNIST baseline: hidden widths 512 and 256, ReLU, dropout 0.2, bias terms, Adam at `lr=1e-3`, batch 128, 535,818 parameters. Keras' `mnist_mlp.py` reaches 98.40% test accuracy with two hidden layers, dropout 0.2 and batch 128, and baselines that swap its RMSprop for Adam at 1e-3 land in the same range at 512/256 widths. Dropout and the early stop regularise, so weight decay stays at 0.

An earlier version ran both models off one shared config. That asked which mechanism suited one arbitrary point in hyperparameter space. Each mechanism now gets its own best-known setup instead, and `n_params` is logged per trial so the size difference stays visible.

After a new search, either edit `PCNConfig` or point a run straight at a results table:

```bash
uv run pcn-comparison              # 10 runs each, results go to out/
uv run pcn-comparison --n-runs 3 --device cpu
uv run pcn-comparison --pcn-from-gridsearch out/gridsearch/results.csv
uv run pcn-comparison --pcn-from-gridsearch out/gridsearch/results.csv --pcn-config-id 279
```

The flag takes the table's best-scoring row unless `--pcn-config-id` names another, and prints the config it resolved to. Use it to try a candidate; write the values into `PCNConfig` once one settles, so a run stops depending on a file the next search overwrites.

Both models train until validation accuracy has gone `patience` epochs without improving, then load the best epoch's weights before the test split is scored. Each model brings its own epoch cap and patience (30/4 for the PCN, 50/5 for the MLP), because an MLP epoch costs seconds and a PCN epoch costs minutes. `train_with_early_stopping` in `runner.py` runs both loops, so neither model gets a stopping rule the other does not.

Each run index gets one seed, and that seed drives both trials: same train/val split, same image order. Batching differs (49 against 128) because that is one of the things tuned. Across the ten runs this supports a paired comparison rather than twenty independent draws.

Every trial's summary lands in `out/results.csv`, with the optimizer, learning rate, batch size, best epoch and epochs run alongside the metrics. The model's full config, its test metrics and its per-epoch history go to `out/runs/<model>_run<idx>_seed<seed>.json`. Tables written before this change carry different columns, so `log_trial` refuses to append to one and asks you to move it aside.

`ExperimentConfig.eval_train` scores the full train split every epoch, which is what fills the train curves in `notebooks/analyze_results.ipynb`. For the PCN that costs about as much as the epoch itself, since every train image has to be relaxed again label-free; set it to `False` when the validation curve is enough.

Run its tests with:

```bash
uv run python test/test_comparison.py
```

## src/gridsearch

The notebooks above use hand-picked values for `eta_infer`, `T_infer`, `lr`, and `weight_decay`. This searches them instead.

```bash
uv run pcn-gridsearch                          # 128 samples per optimizer, 5-fold CV
uv run pcn-gridsearch --n-samples 20 --num-epochs 5 --patience 2
```

`space.py` draws a separate scrambled Sobol design for each optimizer over the eight non-categorical dimensions: `eta_infer`, `T_infer`, `lr`, `weight_decay`, `negative_slope`, `batch_size`, `hidden_width`, and `n_hidden_layers`. `--n-samples` counts configurations per optimizer, so with Adam, SGD, and Muon, the default `--n-samples 128` evaluates 384 in total. Keep it a power of two: Sobol points lose their balance properties at other sizes, and scipy warns. `search.py` scores each candidate with 5-fold cross-validation, all folds shared across every candidate. `data.py` loads only the MNIST train split.

Each fold trains for up to `--num-epochs` (15) and stops once its own validation accuracy has gone `--patience` (4) epochs without improving. The stop is per fold, so a fold that converges in 6 epochs stops there while its siblings keep going. The fold row reports the best epoch's metrics, which is the score a restored best-weights checkpoint would give, and records `best_epoch` and `epochs_run` alongside. Picking the epoch on the same fold the row reports makes `val_accuracy` mildly optimistic, which every config pays equally. A diverged fold scores NaN, never improves, and so gives up after `--patience` epochs instead of burning all 15.

The five folds of a configuration share an architecture, so `search.py` trains them stacked, spending one set of kernel launches on all five instead of five sets on one each. Each fold keeps its own weights, optimizer and data, and a fold that stops early leaves the stack while the others carry on. On top of that, one configuration still leaves an H200 mostly idle, so `--n-workers` (8) scores that many at once in one process pool. Results do not depend on it: every fold re-seeds from `seed` and its fold index, so a configuration scores the same whichever worker takes it. Keep `--n-workers` at or below `--cpus-per-task`, which binds before the GPU does. [docs/parallelisation.md](docs/parallelisation.md) has the measurements behind this, and the two larger changes still on the table.

Two tables land in `out/gridsearch/`, joinable on `config_id`:

- `folds.csv`: one row per (config, fold), every metric raw. Model from this one.
- `results.csv`: one row per config, with mean, std, min and max for accuracy, macro F1, cross-entropy, the free energies and the timings.

## notebooks/

`gridsearch.ipynb` reads those two tables and plots the search: the winning configurations and their fold spread, the best score against the number of configurations drawn, per-optimizer Spearman correlations between each hyperparameter and accuracy, the `lr`/`eta_infer` region where training diverges, and median accuracy across the width and depth grid. Run the search first, then the notebook.

## Cluster

Build the locked grid-search environment once, then submit the search from the repository root:

```bash
singularity build containers/pcn_uv.sif containers/pcn_uv.def
```

The Slurm job binds the checkout at `/workspace`, uses the image's pre-synced CUDA 12.8 dependencies, writes CSVs to `out/gridsearch/`, and runs on CUDA. Rebuild the image whenever `pyproject.toml` or `uv.lock` changes.

A single job runs the three optimizers one after another and hits the walltime, so the script is a job array with **one task per optimizer**, matching the split the Sobol design already uses. Submit it with no arguments:

```bash
sbatch cluster/run_pcn_gridsearch.slurm
```

Task 0 runs Adam, 1 SGD, 2 Muon, taken from `space.OPTIMIZERS`: `--optimizer` defaults to `OPTIMIZERS[SLURM_ARRAY_TASK_ID]`. The script holds the rest, in the `#SBATCH` lines and a `SEARCH_ARGS` line for `--n-samples` and `--num-epochs`.

Add a fourth optimizer to `SEARCH_SPACE` and you also need to bump `--array` to `0-3`. Forget, and the task warns you at startup, because a short array drops that optimizer and you would find the gap hours later in the merged table. Arguments on the `sbatch` line override `SEARCH_ARGS`, which covers a one-off `sbatch cluster/run_pcn_gridsearch.slurm --num-epochs 5`.

Each task evaluates `--n-samples` configurations, so 128 samples gives each task 128 configs, a third of the 384-config design.

Each task writes its own `folds_<optimizer>.csv` and `results_<optimizer>.csv`, so tasks do not clobber one another. Fold them into the two tables the notebooks read once the array finishes:

```bash
uv run tools/merge_gridsearch.py
```

The runner numbers `config_id` across the whole design before applying the optimizer filter, so the merged tables carry the numbering an unsplit run would produce. The merge de-duplicates on `config_id` and keeps the newest file, so re-running one optimizer (`sbatch --array=2 cluster/run_pcn_gridsearch.slurm`) supersedes its rows instead of double-counting them.

## Background

- Stenlund, "Introduction to Predictive Coding Networks for Machine Learning" (arXiv:2506.06332)
- Anonymous, "Towards Stable Learning in Predictive Coding Networks" (under review, ICLR 2025)
