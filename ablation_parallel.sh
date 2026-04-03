#!/bin/bash
#SBATCH --job-name=ablation_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=48:00:00

#SBATCH --array=0-6

#SBATCH --output=/cluster/tufts/c26sp1cs0137/ashen05/logs/ablation_%A_%a.out
#SBATCH --error=/cluster/tufts/c26sp1cs0137/ashen05/logs/ablation_%A_%a.err

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ashen05@tufts.edu

set -uo pipefail

# ── Config ─────────────────────────────────────────────────────────────
ABLATION_SCRIPT=/cluster/tufts/c26sp1cs0137/ashen05/ablation.py
LOG_DIR=/cluster/tufts/c26sp1cs0137/ashen05/logs

mkdir -p "$LOG_DIR"
mkdir -p /cluster/tufts/c26sp1cs0137/ashen05/ablation

# ── Label list (ORDER MATTERS) ─────────────────────────────────────────
LABELS=(
    "only_wind_u"
    "only_wind_v"
    "only_wind_gust"
    "only_precipitation"
    "only_pressure"
    "only_clouds_rad"
    "only_other"
)

LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}

echo "=============================================="
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Task ID     : $SLURM_ARRAY_TASK_ID"
echo "  Label       : $LABEL"
echo "  Node        : $SLURMD_NODENAME"
echo "  Start time  : $(date)"
echo "=============================================="

# ── Load modules ───────────────────────────────────────────────────────
module load class/default
module load cs137/2026spring

# ── GPU info ───────────────────────────────────────────────────────────
echo "--- GPU ---"
nvidia-smi

# ── Run ONE experiment ─────────────────────────────────────────────────
echo ""
echo "Running experiment: $LABEL"
echo ""

python "$ABLATION_SCRIPT" "$LABEL"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "DONE: $LABEL"
else
    echo "FAILED: $LABEL (exit $EXIT_CODE)"
fi

echo "End time: $(date)"
exit $EXIT_CODE