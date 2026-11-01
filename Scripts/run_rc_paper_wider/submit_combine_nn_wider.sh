#!/bin/bash
#SBATCH --qos=preemptable
#SBATCH --job-name=combine_wider
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Scripts/run_rc_paper_wider
#SBATCH --output=slurm_logs/combine_wider_%j.out
#SBATCH --error=slurm_logs/combine_wider_%j.err

# Combine one wider-prior condition's training data once its simulation array has finished:
#   jid=$(sbatch --parsable submit_datagenerating_nn_combined_02AElatentAM_wider.sh)
#   sbatch --dependency=afterany:$jid submit_combine_nn_wider.sh 02AElatentAM_wider
# Writes nothing if any configured condition is missing (rerun the failed tasks, then
# submit this again). Output: Data/DataGeneratingNN_Paper_wider/nn_training_combined_<condition>.csv

set -euo pipefail
if [ $# -ne 1 ]; then
    echo "usage: sbatch submit_combine_nn_wider.sh <condition>   (e.g. 02AElatentAM_wider)" >&2
    exit 1
fi

mkdir -p slurm_logs
module purge
source /curc/sw/anaconda3/latest
conda activate /projects/xuly4739/general_env

python -u CombineNN_Data_Large_wider.py "$1" --split --test_size 0.2
