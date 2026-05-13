#!/bin/bash
#SBATCH --qos=preemptable
#SBATCH --job-name=nn_04migration_test1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --chdir /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Scripts/run_rc_paper
#SBATCH --exclude bmem-rico1
#SBATCH --output=slurm_logs/nn_04migration_test1_%A_%a.out
#SBATCH --error=slurm_logs/nn_04migration_test1_%A_%a.err

# Array for 50 tasks: 50 tasks × 20 conditions = 1,000 total conditions
# All 50 jobs run in parallel
#SBATCH --array=1-50%50

# --- Job Configuration ---
mkdir -p slurm_logs

echo "================================================"
echo "Data Generation NN - Combined Simulation & Analysis"
echo "Condition: 04AMmigration_test1"
echo "  Trait 1: EA (Educational Attainment) — mating trait (am11 varies)"
echo "  Trait 2: Migration — genetically/environmentally correlated with EA;"
echo "           all genetic effects are latent (prop_h2_latent2 = 1, no PGS)"
echo "  within-trait VT only (f11, f22); no shared env (s=0); 40 generations"
echo "  Total conditions: 1,000 (50 jobs × 20 conditions)"
echo "  Equilibrium check: last 5 gen-pairs, threshold=5% relative change in V_y"
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
python -u DataGeneratingNN_Combined_04AMmigration_test1.py

echo "Task $SLURM_ARRAY_TASK_ID completed."

# If this is the last task, combine all data
if [ "$SLURM_ARRAY_TASK_ID" == "50" ]; then
    echo ""
    echo "================================================"
    echo "This is the final task - combining all data..."
    echo "================================================"
    python -u CombineNN_Data_Large_04AMmigration_test1.py --split --test_size 0.2
    echo "Data combination completed."
fi
