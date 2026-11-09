#!/bin/bash
#SBATCH --qos=preemptable
#SBATCH --job-name=combine_rescaled
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Scripts/run_rc_paper_rescaled
#SBATCH --output=slurm_logs/combine_rescaled_%j.out
#SBATCH --error=slurm_logs/combine_rescaled_%j.err

# Combine one rescaled condition's training data once its simulation array has finished:
#   jid=$(sbatch --parsable submit_datagenerating_nn_combined_02AElatentAM_rescaled.sh)
#   sbatch --dependency=afterany:$jid submit_combine_nn_rescaled.sh 02AElatentAM_rescaled
# Writes nothing if any configured condition is missing (rerun the failed tasks, then
# submit this again). Output: Data/to_nn_NPE/nn_training_<condition>.csv, named as in
# Scripts/nn_NPE (rsync Data/to_nn_NPE/ into Scripts/nn_NPE/).

set -euo pipefail
if [ $# -ne 1 ]; then
    echo "usage: sbatch submit_combine_nn_rescaled.sh <condition>   (e.g. 02AElatentAM_rescaled)" >&2
    exit 1
fi

condition=$1

mkdir -p slurm_logs

# Clear the arguments first: a sourced file sees this script's $1, and the CURC conda setup
# would try to activate it as an environment. Conda's shell hooks are not written for
# strict mode, so it stays off until the environment is active.
set --
set +eu
module purge
source /curc/sw/anaconda3/latest
conda activate /projects/xuly4739/general_env || exit 1
set -eu

python -u CombineNN_Data_Large_rescaled.py "$condition" --split --test_size 0.2
