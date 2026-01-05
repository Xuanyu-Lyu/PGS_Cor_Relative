#!/bin/bash
#SBATCH --qos=preemptable
#SBATCH --job-name=nn_analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Scripts/run_rc
#SBATCH --exclude bmem-rico1
#SBATCH --output=slurm_logs/nn_analyze_%A_%a.out
#SBATCH --error=slurm_logs/nn_analyze_%A_%a.err

# Array for 200 different parameter conditions
# Run up to 40 jobs in parallel to manage cluster resources
#SBATCH --array=1-200%40

# --- Job Configuration ---
mkdir -p slurm_logs

echo "================================================"
echo "Data Generation NN Analysis"
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

# Run analysis script
echo "Starting analysis for condition $SLURM_ARRAY_TASK_ID..."
python -u DataGeneratingNN_Analyze.py

echo "Task $SLURM_ARRAY_TASK_ID completed."

# If this is the last task, combine all data
if [ "$SLURM_ARRAY_TASK_ID" == "200" ]; then
    echo ""
    echo "================================================"
    echo "This is the final task - combining all data..."
    echo "================================================"
    python -u CombineNN_Data.py --split --test_size 0.2
    echo "Data combination completed."
fi
