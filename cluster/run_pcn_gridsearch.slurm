#!/bin/bash
#SBATCH --job-name=pcn-gridsearch
#SBATCH --partition=h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3

SEARCH_ARGS=(--n-samples 128 --num-epochs 15)

set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
CONTAINER_IMAGE="$PROJECT_ROOT/containers/pcn_uv.sif"

mkdir -p "$PROJECT_ROOT/out/gridsearch"

singularity run --nv \
    --bind "$PROJECT_ROOT:/workspace" \
    "$CONTAINER_IMAGE" \
    "${SEARCH_ARGS[@]}" "$@"
