#!/bin/bash
#====================================================================
# submit_all.sh — Submit experiments as separate 48-Hour Batch Jobs
#====================================================================
# Submits each experiment as an independent batch job configured for
# a 48-hour cluster queue limit, with automatic re-submission and
# checkpoint-based resume.
#
# Works with both PBS (qsub) and SLURM (sbatch).
#
# Usage:
#   chmod +x scripts/submit_all.sh scripts/run_single_exp.sh
#   ./scripts/submit_all.sh              # Submit all 5 experiments
#   ./scripts/submit_all.sh 2            # Submit only experiment 2
#   ./scripts/submit_all.sh 2 3          # Submit experiments 2 and 3
#====================================================================

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
mkdir -p logs

EXPERIMENTS="${@:-1 2 3 4 5}"

echo "======================================================================"
echo "  Submitting QC-CNN-Parallel Experiments to 48-Hour Cluster Queue"
echo "  Root Directory: ${PROJECT_ROOT}"
echo "  Experiments   : ${EXPERIMENTS}"
echo "  Queue Limit   : 48:00:00 per job (auto-resubmits until finished)"
echo "======================================================================"

# Detect cluster scheduler
USE_PBS=false
USE_SLURM=false

if command -v qsub >/dev/null 2>&1; then
    USE_PBS=true
    echo "  [SCHEDULER] Detected PBS (qsub)."
elif command -v sbatch >/dev/null 2>&1; then
    USE_SLURM=true
    echo "  [SCHEDULER] Detected SLURM (sbatch)."
else
    echo "  [WARNING] Neither qsub nor sbatch found in PATH."
    echo "  Defaulting to PBS job submission syntax."
    USE_PBS=true
fi

JOB2=""

for EXP in $EXPERIMENTS; do
    echo ""
    echo "  --- Configuring Experiment ${EXP} ---"

    if [ "$USE_PBS" = true ]; then
        DEPEND=""
        # Experiment 3 needs Experiment 2's trained weights
        if [ "$EXP" -eq 3 ] && [ -n "$JOB2" ]; then
            DEPEND="-W depend=afterok:$JOB2"
            echo "  (depends on Exp 2: $JOB2)"
        fi

        JOB_ID=$(qsub -N "qccnn_exp${EXP}" \
            -l select=1:ncpus=8:ngpus=1:mem=32gb \
            -l walltime=48:00:00 \
            -q workq \
            -V \
            $DEPEND \
            -o "logs/exp${EXP}_output.log" \
            -e "logs/exp${EXP}_error.log" \
            -- "${PROJECT_ROOT}/scripts/run_single_exp.sh" "${EXP}")

        echo "  → PBS Job submitted: ${JOB_ID}"
        if [ "$EXP" -eq 2 ]; then
            JOB2="$JOB_ID"
        fi

    elif [ "$USE_SLURM" = true ]; then
        DEPEND=""
        if [ "$EXP" -eq 3 ] && [ -n "$JOB2" ]; then
            DEPEND="--dependency=afterok:$JOB2"
            echo "  (depends on Exp 2: $JOB2)"
        fi

        JOB_ID=$(sbatch --job-name="qccnn_exp${EXP}" \
            --time=48:00:00 \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=8 \
            --gres=gpu:1 \
            --mem=32G \
            $DEPEND \
            --output="logs/exp${EXP}_output.log" \
            --error="logs/exp${EXP}_error.log" \
            --parsable \
            "${PROJECT_ROOT}/scripts/run_single_exp.sh" "${EXP}")

        echo "  → SLURM Job submitted: ${JOB_ID}"
        if [ "$EXP" -eq 2 ]; then
            JOB2="$JOB_ID"
        fi
    fi
done

echo ""
echo "======================================================================"
echo "  All requested experiment batch jobs submitted!"
if [ "$USE_PBS" = true ]; then
    echo "  Check status:  qstat -u \$USER"
    echo "  Cancel job:    qdel <JOB_ID>"
elif [ "$USE_SLURM" = true ]; then
    echo "  Check status:  squeue -u \$USER"
    echo "  Cancel job:    scancel <JOB_ID>"
fi
echo "  View live log: tail -f logs/exp*_output.log"
echo "======================================================================"
