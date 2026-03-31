#!/bin/bash
#SBATCH --job-name=ablation_study
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/cluster/tufts/c26sp1cs0137/ashen05/logs/ablation_%j.out
#SBATCH --error=/cluster/tufts/c26sp1cs0137/ashen05/logs/ablation_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ashen05@tufts.edu

# ── Setup ─────────────────────────────────────────────────────────────────────
# Do NOT use set -e here — we want the loop to continue even if one
# experiment fails (e.g. a rare CUDA OOM on a single config).
set -uo pipefail

ABLATION_SCRIPT=/cluster/tufts/c26sp1cs0137/ashen05/ablation.py
RESULTS_JSON=/cluster/tufts/c26sp1cs0137/ashen05/ablation/ablation_results.json
LOG_DIR=/cluster/tufts/c26sp1cs0137/ashen05/logs

echo "=============================================="
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Node        : $SLURMD_NODENAME"
echo "  Start time  : $(date)"
echo "  Working dir : $(pwd)"
echo "=============================================="

mkdir -p "$LOG_DIR"
mkdir -p /cluster/tufts/c26sp1cs0137/ashen05/ablation

# ── Load modules ──────────────────────────────────────────────────────────────
module load class/default
module load cs137/2026spring

# ── System info ───────────────────────────────────────────────────────────────
echo ""
echo "--- GPU info ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo ""
echo "--- CPU / memory ---"
echo "CPUs allocated : $SLURM_CPUS_PER_TASK"
echo "Memory limit   : $(cat /sys/fs/cgroup/memory/slurm/uid_${UID}/job_${SLURM_JOB_ID}/memory.limit_in_bytes 2>/dev/null | awk '{printf "%.0f GB\n", $1/1073741824}' || echo 'n/a')"

echo ""
echo "--- Python / PyTorch ---"
python -c "
import torch, sys
print(f'Python  {sys.version.split()[0]}')
print(f'PyTorch {torch.__version__}')
print(f'CUDA    {torch.cuda.is_available()}  ({torch.version.cuda})')
if torch.cuda.is_available():
    print(f'Device  {torch.cuda.get_device_name(0)}')
"

# ── Resume detection ──────────────────────────────────────────────────────────
echo ""
if [ -f "$RESULTS_JSON" ]; then
    N_DONE=$(python -c "import json; d=json.load(open('$RESULTS_JSON')); print(len(d))" 2>/dev/null || echo 0)
    echo "--- Resuming: $N_DONE experiments already complete ---"
else
    echo "--- Fresh run: no prior results found ---"
fi

# ── Run ablation (with automatic resume on prior partial results) ─────────────
echo ""
echo "--- Starting ablation study : $(date) ---"
echo ""

python "$ABLATION_SCRIPT"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "--- Ablation study COMPLETE : $(date) ---"
else
    echo "--- Ablation study FAILED (exit code $EXIT_CODE) : $(date) ---"
    echo "    Partial results (if any) are in $RESULTS_JSON"
    echo "    Re-submit this script to resume from where it stopped."
fi

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "--- Resource usage ---"
sacct -j "$SLURM_JOB_ID" \
      --format=JobID,Elapsed,CPUTime,MaxRSS,MaxVMSize \
      --noheader 2>/dev/null || true

echo ""
echo "=============================================="
echo "  End time : $(date)"
echo "=============================================="

exit $EXIT_CODE