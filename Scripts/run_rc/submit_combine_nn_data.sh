#!/bin/bash
#SBATCH --qos=preemptable
#SBATCH --job-name=combine_nn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --chdir /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Scripts/run_rc
#SBATCH --exclude bmem-rico1
#SBATCH --output=slurm_logs/combine_nn_%j.out
#SBATCH --error=slurm_logs/combine_nn_%j.err

# ============================================================================
# SLURM Job Script for Combining NN Training Data
# This script combines all condition data into a single dataset
# Use --dependency=afterok:<JOB_ID> to run after analysis completes
# ============================================================================

# --- Job Configuration ---
mkdir -p slurm_logs

echo "================================================"
echo "Combining Neural Network Training Data"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Running on host: $(hostname)"
echo "================================================"

# Load environment
module purge
source /curc/sw/anaconda3/latest
conda activate /projects/xuly4739/general_env

# Run combine script with train/test split
echo "Combining all condition data..."
python -u CombineNN_Data_Large.py --split --test_size 0.2

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "Data combination completed successfully"
    echo "================================================"
else
    echo ""
    echo "================================================"
    echo "Data combination failed"
    echo "================================================"
    exit 1
fi
