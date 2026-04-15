#!/bin/bash
#SBATCH --qos=preemptable
#SBATCH --job-name=pred_cond_uni_DirAM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --chdir /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Scripts/run_rc_paper
#SBATCH --exclude bmem-rico1
#SBATCH --output=slurm_logs/pred_cond_uni_DirAM_%A_%a.out
#SBATCH --error=slurm_logs/pred_cond_uni_DirAM_%A_%a.err

# Array for 100 iterations / 5 per task = 20 tasks total
#SBATCH --array=1-20%10

# --- Job Configuration ---
mkdir -p slurm_logs

echo "================================================"
echo "Predicted Condition: Two independent univariate Direct AM models"
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
echo "Starting simulation: two independent univariate Direct AM models"
echo "Trait 1 (AE model, 01DirAM_AE posterior means):"
echo "  prop_h2_latent1=0.7414  vg1=0.2325  am11=0.3343  f11=0.0"
echo "Trait 2 (AFE model, 01DirAM_AFE posterior means):"
echo "  prop_h2_latent2=0.7082  vg2=0.2708  am22=0.3288  f22=0.0111  s22=0.2600"
echo "Cross-trait: rg=0.0  re=0.0 (independent)"
echo ""
python -u run_predicted_condition_rc.py

echo "Task $SLURM_ARRAY_TASK_ID completed."
