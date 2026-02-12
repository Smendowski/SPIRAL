#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --account=plgaiforsoc2025-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=logs/prepare_env_%j.out
#SBATCH --error=logs/prepare_env_%j.err

export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SCRATCH/.venv_novel_ts2i"

# IMPORTANT: load the modules for machine learning tasks and libraries
ml ML-bundle/24.06a

# Download and install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up venv
rm -rf $SCRATCH/.venv_novel_ts2i
python3 -m venv $SCRATCH/.venv_novel_ts2i
source $SCRATCH/.venv_novel_ts2i/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install uv dependencies
uv sync
uv tool install ruff

# Override torch version to match the one provided by the ML-bundle
pip install /net/software/aarch64/el8/wheels/ML-bundle/24.06a/torch-2.7.0rc9+cu124-cp311-cp311-linux_aarch64.whl /net/software/aarch64/el8/wheels/ML-bundle/24.06a/torchvision-0.21.0+cu124torch260-cp311-cp311-linux_aarch64.whl --force-reinstall
pip install numpy==2.3.0
