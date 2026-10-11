#!/bin/bash
#SBATCH --qos=preemptable
#SBATCH --job-name=nn_04eamig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --chdir /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Scripts/run_rc_paper
#SBATCH --exclude bmem-rico1
#SBATCH --output=slurm_logs/nn_04eamig_%A_%a.out
#SBATCH --error=slurm_logs/nn_04eamig_%A_%a.err

# Array for 250 tasks: 250 tasks × 10 conditions = 2,500 simulations;
# each simulation is read out at n_gen = 3..10 -> 20,000 training rows
# Run up to 50 jobs in parallel to manage cluster resources
#SBATCH --array=1-250%50

# --- Job Configuration ---
mkdir -p slurm_logs

echo "================================================"
echo "Data Generation NN - Combined Simulation & Analysis"
echo "Condition: 04EAmigration"
echo "  EA = additive genetics + unique environment; no AM, no VT, no shared env"
echo "  5 islands; migration sorted directly on EA (move_p varies)"
echo "  One 10-generation simulation per condition, read out at n_gen = 3..10"
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

# Let task 1 run immediately so it can write the conditions config file;
# all other tasks wait 30 seconds before starting.
if [ "$SLURM_ARRAY_TASK_ID" != "1" ]; then
    sleep 30
fi

# Run combined script
echo "Starting combined simulation and analysis for task $SLURM_ARRAY_TASK_ID..."
python -u DataGeneratingNN_Combined_04EAmigration.py

echo "Task $SLURM_ARRAY_TASK_ID completed."

# If this is the last task, combine all data
if [ "$SLURM_ARRAY_TASK_ID" == "250" ]; then
    echo ""
    echo "================================================"
    echo "This is the final task - combining all data..."
    echo "================================================"
    python -u CombineNN_Data_Large_04EAmigration.py --split --test_size 0.2
    echo "Data combination completed."
fi
