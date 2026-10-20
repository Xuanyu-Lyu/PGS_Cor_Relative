#!/bin/bash
#SBATCH --qos=preemptable
#SBATCH --job-name=combine_03v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Scripts/run_rc_paper
#SBATCH --output=slurm_logs/combine_03v2_%j.out
#SBATCH --error=slurm_logs/combine_03v2_%j.err

# Combine condition-03 v2 training data once the whole simulation array has finished:
#   jid=$(sbatch --parsable submit_datagenerating_nn_combined_03EnvLatentAM_v2.sh)
#   sbatch --dependency=afterany:$jid submit_combine_nn_03EnvLatentAM_v2.sh
# Writes nothing if any configured condition is missing (rerun the failed tasks, then
# submit this again). Output: Data/DataGeneratingNN_Paper/nn_training_combined_03EnvLatentAM_v2.csv

mkdir -p slurm_logs
module purge
source /curc/sw/anaconda3/latest
conda activate /projects/xuly4739/general_env

python -u CombineNN_Data_Large_03EnvLatentAM_v2.py --split --test_size 0.2
