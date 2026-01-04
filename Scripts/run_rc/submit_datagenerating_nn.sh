#!/bin/bash
#SBATCH --qos=preemptable
#SBATCH --job-name=datagen_nn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --chdir /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Scripts/run_rc
#SBATCH --exclude bmem-rico1
#SBATCH --output=slurm_logs/datagen_nn_%A_%a.out
#SBATCH --error=slurm_logs/datagen_nn_%A_%a.err

# Array for 200 different parameter conditions
# Run up to 50 jobs in parallel to manage cluster resources
#SBATCH --array=1-200%50

# --- Job Configuration ---
mkdir -p slurm_logs

echo "================================================"
echo "Data Generation for Neural Network Training"
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

# Run data generation script
echo "Starting data generation for condition $SLURM_ARRAY_TASK_ID..."
python -u DataGeneratingNN.py

echo "Task $SLURM_ARRAY_TASK_ID completed."
