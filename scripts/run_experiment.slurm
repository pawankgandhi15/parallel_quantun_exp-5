#!/bin/bash
#====================================================================
# SLURM Job Script — QC-CNN-Parallel Automated 48-Hour Batch Runner
#====================================================================
# Designed for H100 GPU clusters running SLURM with a 48-hour queue limit.
#
# Submit with:   sbatch scripts/run_experiment.slurm
# Check status:  squeue -u $USER
# Cancel job:    scancel <JOB_ID>
#====================================================================

#--- SLURM Directives -------------------------------------------------
#SBATCH --job-name=qccnn_h100
#SBATCH --time=48:00:00                    # Strict 48-hour walltime limit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                       # Request 1 H100 GPU
#SBATCH --mem=32G
#SBATCH --output=logs/slurm_%j_out.log
#SBATCH --error=logs/slurm_%j_err.log

#--- Environment & Working Directory ----------------------------------
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

if [ -d "implementation" ]; then
    PROJECT_ROOT="$(pwd)"
elif [ -f "run_all.py" ]; then
    PROJECT_ROOT="$(cd .. && pwd)"
    cd "$PROJECT_ROOT"
elif [ -d "../implementation" ]; then
    PROJECT_ROOT="$(cd .. && pwd)"
    cd "$PROJECT_ROOT"
else
    PROJECT_ROOT="$(pwd)"
fi

mkdir -p logs
mkdir -p implementation/results

SCRIPT_PATH="${PROJECT_ROOT}/scripts/run_experiment.slurm"
if [ ! -f "$SCRIPT_PATH" ]; then
    SCRIPT_PATH="$(pwd)/run_experiment.slurm"
fi

# Load cluster modules (uncomment/adjust for your cluster)
# module load python/3.10
# module load cuda/12.2

# Activate virtual environment or conda environment (uncomment/adjust)
# source activate quantum_env
# conda activate quantum_env
# source ~/venvs/quantum/bin/activate

echo "======================================================================"
echo "  SLURM Job ID    : ${SLURM_JOB_ID:-Local/Interactive}"
echo "  Running Node    : $(hostname)"
echo "  Started at      : $(date)"
echo "  Project Root    : ${PROJECT_ROOT}"
echo "  Submission Script: ${SCRIPT_PATH}"
echo "  GPU Information :"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
echo "======================================================================"

#--- Completion & Cycle Safety Checks ---------------------------------
SENTINEL="${PROJECT_ROOT}/implementation/results/ALL_COMPLETED"
COUNTER_FILE="${PROJECT_ROOT}/logs/slurm_resubmit_count.txt"
MAX_CYCLES=20  # Safety cap: up to 20 cycles * 48h = 960 hours of compute

if [ -f "$SENTINEL" ]; then
    echo ""
    echo "======================================================================"
    echo "  🎉 All experiments are ALREADY COMPLETED! Nothing left to run."
    echo "  Found completion marker: ${SENTINEL}"
    echo "  Finished at: $(date)"
    echo "======================================================================"
    exit 0
fi

CYCLE_COUNT=0
if [ -f "$COUNTER_FILE" ]; then
    CYCLE_COUNT=$(cat "$COUNTER_FILE")
fi
CYCLE_COUNT=$((CYCLE_COUNT + 1))
echo "$CYCLE_COUNT" > "$COUNTER_FILE"
echo "  Current 48-Hour Batch Cycle: ${CYCLE_COUNT} of ${MAX_CYCLES} allowed"

if [ "$CYCLE_COUNT" -gt "$MAX_CYCLES" ]; then
    echo "  [ERROR] Maximum allowed auto-resubmission cycles (${MAX_CYCLES}) reached."
    echo "  Halting auto-resubmit to prevent runaway jobs. Inspect logs/ for details."
    exit 1
fi

#--- Signal Handler for Early SLURM Preemption / Walltime Warning ------
on_slurm_signal() {
    echo ""
    echo "  [ALERT] Received termination signal from SLURM scheduler!"
    if [ ! -f "$SENTINEL" ] && [ "$CYCLE_COUNT" -le "$MAX_CYCLES" ]; then
        echo "  Automatically submitting continuation job before exit..."
        sbatch "$SCRIPT_PATH"
    fi
    exit 143
}
trap 'on_slurm_signal' SIGTERM SIGUSR1 SIGUSR2

#--- Run Experiments with 47.0-Hour Time Budget -----------------------
cd "${PROJECT_ROOT}/implementation"

echo ""
echo "  Starting experiment execution..."
echo "  Walltime budget: 47.0 hours (Safe cutoff before 48:00:00 queue limit)"
echo ""

python run_all.py \
    --exp 1 2 3 4 5 \
    --max_hours 47.0 \
    --checkpoint_interval_batches 25

EXIT_CODE=$?
cd "${PROJECT_ROOT}"

echo ""
echo "======================================================================"
echo "  Python process finished at: $(date)"
echo "  Exit code: ${EXIT_CODE}"
echo "======================================================================"

#--- Handle Continuation / Auto-Resubmission --------------------------
if [ -f "$SENTINEL" ]; then
    echo ""
    echo "======================================================================"
    echo "  🎉 SUCCESS: All experiments in the suite completed!"
    echo "  No further job submissions needed."
    echo "======================================================================"
    rm -f "$COUNTER_FILE"
    exit 0
fi

if [ "$EXIT_CODE" -eq 42 ] || [ "$EXIT_CODE" -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "  ⏳ 48-HOUR QUEUE CYCLE COMPLETE (or walltime budget reached)."
    echo "  All weights, optimizer states, batches, and metrics safely saved."
    echo "  Automatically re-submitting continuation job to SLURM queue..."
    echo "======================================================================"

    NEXT_JOB_ID=$(sbatch "$SCRIPT_PATH")
    echo "  → Successfully submitted next batch job: ${NEXT_JOB_ID}"
    echo "  → The new job will automatically resume from the exact checkpoint."
    echo "======================================================================"
    exit 0
else
    echo ""
    echo "  [ERROR] Python process exited with unexpected code: ${EXIT_CODE}."
    echo "  Auto-resubmission paused. Please review logs/slurm_*_err.log."
    exit ${EXIT_CODE}
fi
