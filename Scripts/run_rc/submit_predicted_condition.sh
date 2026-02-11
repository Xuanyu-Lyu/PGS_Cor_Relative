#!/bin/bash
#SBATCH --qos=preemptable
#SBATCH --job-name=predicted_cond
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --chdir /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Scripts/run_rc
#SBATCH --exclude bmem-rico1
#SBATCH --output=slurm_logs/predicted_cond_%A_%a.out
#SBATCH --error=slurm_logs/predicted_cond_%A_%a.err

# Array for 100 iterations / 5 per task = 20 tasks total
#SBATCH --array=1-20%10

# --- Job Configuration ---
mkdir -p slurm_logs

echo "================================================"
echo "Predicted Condition Simulation"
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
echo "Starting predicted condition simulation..."
echo "Parameters from neural network prediction:"
echo "  f11: -0.0005 prop_h2_latent1: 0.7819 vg1: 0.5522 vg2: 0.5850 f22: 0.0422 am22: 0.6689 rg: 0.9020"
echo ""
python -u run_predicted_condition_rc.py

echo "Task $SLURM_ARRAY_TASK_ID completed."
