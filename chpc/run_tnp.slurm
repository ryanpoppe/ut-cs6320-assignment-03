#!/bin/bash
# =====================================================================
# CS 6320 Assignment 3 - Part A
# CHPC SLURM batch script: Chicago TNP trip-duration training loop
#
# Submit from the assignment-03 repository root:
#     cd /path/to/ut-cs6320-assignment-03
#     sbatch chpc/run_tnp.slurm
#
# Allocation headers below reuse the Week 3 dc_energy example template
# (account=larsenc, partition/qos=utucset-gpu-grn). The model is tiny and
# trains fine on CPU; the GPU request just matches the provided template and
# is harmless. Captured stdout/stderr (slurm-tnp-%j.out/.err) is the CHPC
# evidence that the computation ran.
# =====================================================================
#SBATCH --partition=utucset-gpu-grn
#SBATCH --qos=utucset-gpu-grn
#SBATCH --account=larsenc
#SBATCH --job-name=tnp-mlp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-tnp-%j.out
#SBATCH --error=slurm-tnp-%j.err

set -euo pipefail

# Slurm runs a spooled copy of this script, so use the submit directory.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRATCH="/scratch/ucgd/serial/u0090307/slurm-tnp"

echo "############################################################"
echo "# Job:        ${SLURM_JOB_ID:-local}"
echo "# Host:       $(hostname)"
echo "# Submit dir: ${SCRIPT_DIR}"
echo "# Date (UTC): $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "# CUDA dev:   ${CUDA_VISIBLE_DEVICES:-unset}"
echo "############################################################"

cd "${SCRIPT_DIR}"

if [[ ! -f "train_tnp_mlp.py" || ! -f "prepare_tnp_data.py" ]]; then
  echo "ERROR: submit from the assignment-03 directory (could not find the scripts)."
  echo "Run: cd /path/to/ut-cs6320-assignment-03 && sbatch chpc/run_tnp.slurm"
  exit 1
fi

# ---------------------------------------------------------------------
# Environment. A plain venv + pip with requirements.txt (portable across
# CHPC partitions). If you prefer uv, swap in:
#     module load uv && uv venv && uv pip install -r requirements.txt
# ---------------------------------------------------------------------
module load python/3.12.4 2>/dev/null || module load python/3.12 2>/dev/null || true
mkdir -p "${SCRATCH}" 2>/dev/null || true

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo "# python: $(python --version 2>&1)"
nvidia-smi -L 2>/dev/null || echo "# no GPU visible (CPU run is fine for this model)"

# ---------------------------------------------------------------------
# Pipeline: prepare splits -> context baselines -> training-loop MLP
# ---------------------------------------------------------------------
echo ">>> STEP 1: prepare_tnp_data.py"
python prepare_tnp_data.py

echo ">>> STEP 2: train_baseline.py  (mean predictor + ridge)"
python train_baseline.py

echo ">>> STEP 3: train_tnp_mlp.py   (explicit training loop)"
python train_tnp_mlp.py

echo ">>> DONE. Metrics, history, model, and plot under artifacts/tnp_prep/runs/"
