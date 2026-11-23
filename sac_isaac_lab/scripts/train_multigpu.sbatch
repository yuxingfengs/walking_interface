#!/usr/bin/env bash
#SBATCH --job-name=nao-omni
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=12

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${ISAACLAB_PATH:?Set ISAACLAB_PATH to the Isaac Lab checkout on the node}"
exec "${ISAAC_PYTHON:-python}" -u sac_isaac_lab/runner/multigpu.py "$@"
