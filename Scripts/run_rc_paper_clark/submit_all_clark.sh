#!/bin/bash
# Submit the training-data simulations of the Clark design (02AElatentAM_clark), followed by
# a combine job that starts once the whole array has finished. Run on the CURC login node,
# after sync_cluster_submodule.sh (the design needs the corrected engine):
#     bash submit_all_clark.sh
#
# The generator writes to Data/DataGeneratingNN_Paper_clark/02AElatentAM_clark/ and refuses to
# start if an existing conditions_config.csv there was written for another design.
# Resubmitting skips the conditions that are already simulated.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p slurm_logs

if [ $# -gt 0 ]; then
    CONDS=("$@")
else
    CONDS=(02AElatentAM_clark)
fi

for cond in "${CONDS[@]}"; do
    jid=$(sbatch --parsable "submit_datagenerating_nn_combined_${cond}.sh")
    cid=$(sbatch --parsable --dependency=afterany:"${jid}" submit_combine_nn_clark.sh "${cond}")
    echo "${cond}: simulation array ${jid}, combine job ${cid} (starts after the array)"
done
