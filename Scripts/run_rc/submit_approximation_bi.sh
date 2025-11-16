#!/bin/bash
#SBATCH --qos=preemptable
#SBATCH --job-name=approx_bi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --chdir /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Scripts/run_rc
#SBATCH --exclude bmem-rico1
#SBATCH --output=slurm_logs/approx_bi_%A_%a.out
#SBATCH --error=slurm_logs/approx_bi_%A_%a.err

# Array for 4 conditions * 200 iterations / 20 per task = 40 tasks total
#SBATCH --array=1-40%20

# --- Job Configuration ---
mkdir -p slurm_logs

echo "================================================"
echo "Bivariate Approximation Simulation"
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
export ITERATIONS_PER_TASK=20
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}

# Run simulation script
echo "Starting bivariate approximation simulation..."
python -u run_approximation_bi_rc.py

echo "Task $SLURM_ARRAY_TASK_ID completed."
