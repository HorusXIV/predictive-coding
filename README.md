# Predictive Coding Networks

A PyTorch implementation of a predictive-coding network (PCN) with a supervised readout, plus tooling to compare it against an architecturally matched MLP and to search its hyperparameters.

## Setup

```bash
uv sync
```

## Layout

```
src/pcn/          the network and a 3D inference visualizer
src/comparison/   PCN vs. equivalent MLP, run repeatedly on MNIST
src/gridsearch/   PCN hyperparameter search
test/             regression tests for src/pcn
tools/            one-off scripts (merging the grid-search tables)
experiments/      notebooks: PCN on HerzCheck, MNIST, CIFAR-10
notebooks/        grid-search result analysis
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

The MLP in `models.py` uses the same layer widths as the PCN, with the same `bias=False` layers, so both networks carry the same parameter count (468,224 for the default `dims=(784, 512, 128)`). Only the training mechanism differs: predictive-coding inference against plain backprop.

```bash
uv run pcn-comparison              # 10 runs each, results go to out/
uv run pcn-comparison --n-runs 3 --device cpu
```

Each run index gets one seed, and that seed drives both the PCN trial and the MLP trial for that index: same train/val split, same image order, same batches. Init and the training mechanism are the only things left to vary, which supports a paired comparison across the ten runs rather than twenty independent draws.

Every trial's summary lands in `out/results.csv`. Its full per-epoch history goes to `out/runs/<model>_run<idx>_seed<seed>.json`.

## src/gridsearch

The notebooks above use hand-picked values for `eta_infer`, `T_infer`, `lr`, and `weight_decay`. This searches them instead.

```bash
uv run pcn-gridsearch                          # 128 samples per optimizer, 5-fold CV
uv run pcn-gridsearch --n-samples 20 --num-epochs 5
```

`space.py` draws a separate scrambled Sobol design for each optimizer over the eight non-categorical dimensions: `eta_infer`, `T_infer`, `lr`, `weight_decay`, `negative_slope`, `batch_size`, `hidden_width`, and `n_hidden_layers`. `--n-samples` counts configurations per optimizer, so with Adam, SGD, and Muon, the default `--n-samples 128` evaluates 384 in total. Keep it a power of two: Sobol points lose their balance properties at other sizes, and scipy warns. `search.py` scores each candidate with 5-fold cross-validation, all folds shared across every candidate. `data.py` loads only the MNIST train split.

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
