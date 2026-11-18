#!/bin/bash
#SBATCH --qos=preemptable
#SBATCH --job-name=nn_02clark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --chdir /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Scripts/run_rc_paper_clark
#SBATCH --exclude bmem-rico1
#SBATCH --output=slurm_logs/nn_02clark_%A_%a.out
#SBATCH --error=slurm_logs/nn_02clark_%A_%a.err

# 500 tasks x 40 conditions = 20,000 conditions; up to 50 tasks at a time
#SBATCH --array=1-500%50

# Condition 02, Clark special case: sorting factor = true EA (h2 = vg2, r = am22), observed EA =
# true EA + measurement error (theta; rg = 1). 15 generations. Writes to
# Data/DataGeneratingNN_Paper_clark/02AElatentAM_clark. Combine AFTER the whole array
# (submit_all_clark.sh does both):
#   jid=$(sbatch --parsable submit_datagenerating_nn_combined_02AElatentAM_clark.sh)
#   sbatch --dependency=afterany:$jid submit_combine_nn_clark.sh 02AElatentAM_clark

# --- Job Configuration ---
mkdir -p slurm_logs

echo "================================================"
echo "Data Generation NN - Combined Simulation & Analysis"
echo "Condition: 02AElatentAM_clark"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Slurm Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Slurm Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Running on host: $(hostname)"
echo "================================================"

# Load environment
module purge
source /curc/sw/anaconda3/latest
conda activate /projects/xuly4739/general_env

# Export task ID
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}

# Let task 1 run immediately so it can write the config file;
# all other tasks wait 30 seconds before starting.
if [ "$SLURM_ARRAY_TASK_ID" != "1" ]; then
    sleep 30
fi

echo "Starting combined simulation and analysis for task $SLURM_ARRAY_TASK_ID..."
python -u DataGeneratingNN_Combined_02AElatentAM_clark.py

echo "Task $SLURM_ARRAY_TASK_ID completed."
