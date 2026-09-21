#!/bin/bash
#====================================================================
# PBS Job Scripts — Submit individual experiments separately
#====================================================================
# This script submits each experiment as a separate PBS job.
# Jobs that depend on prior results (e.g., Exp 3 needs Exp 2 weights)
# are submitted with dependencies using -W depend=afterok:<jobid>
#
# Usage:
#   chmod +x submit_all.sh
#   ./submit_all.sh              # Submit all experiments
#   ./submit_all.sh 2            # Submit only experiment 2
#   ./submit_all.sh 2 3          # Submit experiments 2 and 3
#====================================================================

set -e
cd "$(dirname "$0")"
mkdir -p logs

# Parse which experiments to run (default: all)
EXPERIMENTS="${@:-1 2 3 4 5}"

echo "========================================"
echo "  Submitting QC-CNN-Parallel Experiments"
echo "  Experiments: $EXPERIMENTS"
echo "========================================"

for EXP in $EXPERIMENTS; do
    case $EXP in
        1)
            echo ""
            echo "  Submitting Experiment 1: Circuit Selection..."
            JOB1=$(qsub -N qccnn_exp1 \
                -l select=1:ncpus=4:ngpus=1:mem=16gb \
                -l walltime=72:00:00 \
                -q gpu \
                -o logs/exp1_output.log \
                -e logs/exp1_error.log \
                -- bash -c "cd $PBS_O_WORKDIR && python run_all.py --exp 1")
            echo "  → Job ID: $JOB1"
            ;;
        2)
            echo ""
            echo "  Submitting Experiment 2: Classification..."
            JOB2=$(qsub -N qccnn_exp2 \
                -l select=1:ncpus=4:ngpus=1:mem=16gb \
                -l walltime=48:00:00 \
                -q gpu \
                -o logs/exp2_output.log \
                -e logs/exp2_error.log \
                -- bash -c "cd $PBS_O_WORKDIR && python run_all.py --exp 2")
            echo "  → Job ID: $JOB2"
            ;;
        3)
            echo ""
            echo "  Submitting Experiment 3: Noise Robustness..."
            # Exp 3 needs Exp 2's trained weights — add dependency if Exp 2 was submitted
            DEPEND=""
            if [ ! -z "$JOB2" ]; then
                DEPEND="-W depend=afterok:$JOB2"
                echo "  (depends on Exp 2: $JOB2)"
            fi
            JOB3=$(qsub -N qccnn_exp3 \
                -l select=1:ncpus=4:ngpus=1:mem=16gb \
                -l walltime=24:00:00 \
                -q gpu \
                $DEPEND \
                -o logs/exp3_output.log \
                -e logs/exp3_error.log \
                -- bash -c "cd $PBS_O_WORKDIR && python run_all.py --exp 3")
            echo "  → Job ID: $JOB3"
            ;;
        4)
            echo ""
            echo "  Submitting Experiment 4: Ablation Study..."
            JOB4=$(qsub -N qccnn_exp4 \
                -l select=1:ncpus=4:ngpus=1:mem=16gb \
                -l walltime=72:00:00 \
                -q gpu \
                -o logs/exp4_output.log \
                -e logs/exp4_error.log \
                -- bash -c "cd $PBS_O_WORKDIR && python run_all.py --exp 4")
            echo "  → Job ID: $JOB4"
            ;;
        5)
            echo ""
            echo "  Submitting Experiment 5: Scalability Study..."
            JOB5=$(qsub -N qccnn_exp5 \
                -l select=1:ncpus=4:ngpus=1:mem=16gb \
                -l walltime=72:00:00 \
                -q gpu \
                -o logs/exp5_output.log \
                -e logs/exp5_error.log \
                -- bash -c "cd $PBS_O_WORKDIR && python run_all.py --exp 5")
            echo "  → Job ID: $JOB5"
            ;;
        *)
            echo "  ⚠ Unknown experiment: $EXP (valid: 1-5)"
            ;;
    esac
done

echo ""
echo "========================================"
echo "  All jobs submitted!"
echo "  Check status:  qstat -u \$USER"
echo "  View logs:     tail -f logs/exp*_output.log"
echo "========================================"
