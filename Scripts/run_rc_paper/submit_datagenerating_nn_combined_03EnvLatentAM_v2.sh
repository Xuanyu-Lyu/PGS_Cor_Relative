#!/bin/bash
#SBATCH --qos=preemptable
#SBATCH --job-name=nn_03v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --chdir /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Scripts/run_rc_paper
#SBATCH --exclude bmem-rico1
#SBATCH --output=slurm_logs/nn_03v2_%A_%a.out
#SBATCH --error=slurm_logs/nn_03v2_%A_%a.err

# 500 tasks x 40 conditions = 20,000 conditions; up to 50 tasks at a time
#SBATCH --array=1-500%50

# Condition 03, version 2 (prop_h2_latent1, vg1, re, am22, rg sampled; f11 = f22 = 0.10).
# Writes to Data/DataGeneratingNN_Paper/03EnvLatentAM_v2. Combine AFTER the whole array:
#   jid=$(sbatch --parsable submit_datagenerating_nn_combined_03EnvLatentAM_v2.sh)
#   sbatch --dependency=afterany:$jid submit_combine_nn_03EnvLatentAM_v2.sh

# --- Job Configuration ---
mkdir -p slurm_logs

echo "================================================"
echo "Data Generation NN - Combined Simulation & Analysis"
echo "Condition: 03EnvLatentAM_v2 (VT fixed at 0.10; latent AM on trait 2)"
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
python -u DataGeneratingNN_Combined_03EnvLatentAM_v2.py

echo "Task $SLURM_ARRAY_TASK_ID completed."
