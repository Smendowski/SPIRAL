#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --account=plgaiforsoc2025-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Set env variables
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SCRATCH/.venv_ts2img"

# IMPORTANT: load the modules for machine learning tasks and libraries
ml ML-bundle/24.06a

# Load venv
source $SCRATCH/.venv_ts2img/bin/activate

# Run experiment
DATASET="18_UCR-9"

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
