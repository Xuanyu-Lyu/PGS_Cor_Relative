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
#SBATCH --output=slurm_logs/predicted_cond_%j.out
#SBATCH --error=slurm_logs/predicted_cond_%j.err

# --- Job Configuration ---
mkdir -p slurm_logs

echo "================================================"
echo "Predicted Condition Simulation"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Running on host: $(hostname)"
echo "================================================"

# Load environment
module purge
source /curc/sw/anaconda3/latest
conda activate /projects/xuly4739/general_env

# Run simulation script
echo "Starting predicted condition simulation..."
echo "Parameters from neural network prediction:"
echo "  f11: 0.1331 prop_h2_latent1: 0.7278 vg1: 0.5948 vg2: 0.6569 f22: 0.0794 am22: 0.6573 rg: 0.6683"
echo ""
python -u run_predicted_condition_rc.py

echo "Job completed."
