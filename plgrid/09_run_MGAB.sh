#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --account=plgaiforsoc2025-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --array=0-2                           # <- (0-2) means 3 datasets
#SBATCH --output=logs/%x_%A_%a.out            # <- %A = array job ID, %a = task ID
#SBATCH --error=logs/%x_%A_%a.err

# Set env variables
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SCRATCH/.venv_ts2img"

# Load modules
ml ML-bundle/24.06a
source $SCRATCH/.venv_ts2img/bin/activate

DATASETS=(
    "09_MGAB-1"
    "09_MGAB-2"
    "09_MGAB-3"
)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Processing dataset: $DATASET (task $SLURM_ARRAY_TASK_ID)"

# Run experiment
uv run --no-sync -m cli.entrypoint ts2i \
    --csv-file cli/datasets/$DATASET.csv \
    --output-dir $SCRATCH/ts2i_images \
    --config-file configs/default.yaml \
    --device cuda

uv run --no-sync -m cli.entrypoint train \
    --csv-file cli/datasets/$DATASET.csv \
    --images-dir $SCRATCH/ts2i_images \
    --output-dir $SCRATCH/ts2i_results \
    --config-file configs/default.yaml \
    --device cuda
