#!/bin/bash
#====================================================================
# run_single_exp.sh - Self-Resubmitting Single Experiment Batch Runner
#====================================================================
# Runs a single experiment (1, 2, 3, 4, or 5) under a 48-hour cluster queue.
# Automatically resubmits itself if the 48-hour limit is reached.
#
# Usage:
#   bash scripts/run_single_exp.sh <EXP_NUM>
# Or via PBS:
#   qsub -N qccnn_exp1 -l select=1:ncpus=8:ngpus=1:mem=32gb -l walltime=48:00:00 -q workq -V -- scripts/run_single_exp.sh 1
# Or via SLURM:
#   sbatch --job-name=qccnn_exp1 --time=48:00:00 --gres=gpu:1 -- scripts/run_single_exp.sh 1
#====================================================================

EXP_NUM="${1:-2}"

if [ -n "$PBS_O_WORKDIR" ]; then
    cd "$PBS_O_WORKDIR"
elif [ -n "$SLURM_SUBMIT_DIR" ]; then
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

#--- Python Environment ---------------------------------------------
if [ -f "/apps/compilers/anaconda3/bin/activate" ]; then
    source "/apps/compilers/anaconda3/bin/activate" qcm
fi

PYTHON="/Data4/it_25201815/.conda/envs/qcm/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3 || command -v python)"
fi

COUNTER_FILE="logs/resubmit_exp${EXP_NUM}.txt"
MAX_CYCLES=20
COUNT=0
if [ -f "$COUNTER_FILE" ]; then
    COUNT=$(cat "$COUNTER_FILE")
fi
COUNT=$((COUNT + 1))
echo "$COUNT" > "$COUNTER_FILE"

echo "======================================================================"
echo "  QC-CNN-Parallel Single Experiment Runner"
echo "  Target Experiment : ${EXP_NUM}"
echo "  Cycle Iteration   : ${COUNT} of ${MAX_CYCLES}"
echo "  Walltime Budget   : 47.0 hours (Safe cutoff for 48:00:00 queue)"
echo "  Started at        : $(date)"
echo "  Python            : ${PYTHON}"
echo "======================================================================"

export PYTHONUNBUFFERED=1
if [ -n "$PBS_GPUFILE" ] && [ -s "$PBS_GPUFILE" ]; then
    export CUDA_VISIBLE_DEVICES=$(cat "$PBS_GPUFILE" | tr '\n' ',' | sed 's/,$//')
elif [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

cd "${PROJECT_ROOT}/implementation"

"$PYTHON" -u run_all.py \
    --exp "${EXP_NUM}" \
    --max_hours 47.0 \
    --checkpoint_interval_batches 25

EXIT_CODE=$?
cd "${PROJECT_ROOT}"

echo ""
echo "======================================================================"
echo "  Experiment ${EXP_NUM} finished with exit code: ${EXIT_CODE}"
echo "======================================================================"

# If exited with 42 (planned 47h pause), auto-resubmit
if [ "$EXIT_CODE" -eq 42 ]; then
    if [ "$COUNT" -lt "$MAX_CYCLES" ]; then
        echo "  [RESUBMIT] 48h walltime budget reached. Progress safely checkpointed."
        echo "  Auto-resubmitting continuation job for Experiment ${EXP_NUM}..."

        # Detect scheduler
        if command -v qsub >/dev/null 2>&1; then
            NEXT_JOB=$(qsub -N "qccnn_exp${EXP_NUM}" \
                -l select=1:ncpus=8:ngpus=1:mem=32gb \
                -l walltime=48:00:00 \
                -q workq \
                -V \
                -o "logs/exp${EXP_NUM}_out.log" \
                -e "logs/exp${EXP_NUM}_err.log" \
                -- "${PROJECT_ROOT}/scripts/run_single_exp.sh" "${EXP_NUM}")
            echo "  -> Queued next PBS job: ${NEXT_JOB}"
        elif command -v sbatch >/dev/null 2>&1; then
            NEXT_JOB=$(sbatch --job-name="qccnn_exp${EXP_NUM}" \
                --time=48:00:00 \
                --gres=gpu:1 \
                --cpus-per-task=8 \
                --mem=32G \
                --output="logs/exp${EXP_NUM}_out.log" \
                --error="logs/exp${EXP_NUM}_err.log" \
                "${PROJECT_ROOT}/scripts/run_single_exp.sh" "${EXP_NUM}")
            echo "  -> Queued next SLURM job: ${NEXT_JOB}"
        fi
    else
        echo "  [STOP] Reached max resubmit cycles ($MAX_CYCLES)."
    fi
elif [ "$EXIT_CODE" -eq 0 ]; then
    echo "  Experiment ${EXP_NUM} completed successfully!"
    rm -f "$COUNTER_FILE"
fi

exit $EXIT_CODE
