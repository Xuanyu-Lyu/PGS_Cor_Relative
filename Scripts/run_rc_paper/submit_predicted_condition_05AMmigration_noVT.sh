#!/bin/bash
#SBATCH --qos=preemptable
#SBATCH --job-name=pred_cond_05AMmigration_noVT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --chdir /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Scripts/run_rc_paper
#SBATCH --exclude bmem-rico1
#SBATCH --output=slurm_logs/pred_cond_05AMmigration_noVT_%A_%a.out
#SBATCH --error=slurm_logs/pred_cond_05AMmigration_noVT_%A_%a.err

# Array for 50 iterations / 5 per task = 10 tasks total
#SBATCH --array=1-10%10

# --- Job Configuration ---
mkdir -p slurm_logs

echo "================================================"
echo "Predicted Condition: 05AMmigration_noVT (AE + latent AM on trait 1, island migration on trait 2, no VT)"
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

# Export task parameters
export ITERATIONS_PER_TASK=5
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}

# Run simulation script
python -u run_predicted_condition_rc_05AMmigration_noVT.py

echo "Task $SLURM_ARRAY_TASK_ID completed."
