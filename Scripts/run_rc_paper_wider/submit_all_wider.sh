#!/bin/bash
# Submit the wider-prior training-data simulations of conditions 01-04, each followed by
# a combine job that starts once its whole array has finished. Run on the CURC login
# node, after sync_cluster_submodule.sh:
#     bash submit_all_wider.sh
# or name the conditions to submit:
#     bash submit_all_wider.sh 02AElatentAM_wider 03EnvLatentAM_wider
#
# Each generator writes to Data/DataGeneratingNN_Paper_wider/<condition>/ and refuses to
# start if an existing conditions_config.csv there was written for another design.
# Resubmitting skips the conditions that are already simulated.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p slurm_logs

if [ $# -gt 0 ]; then
    CONDS=("$@")
else
    CONDS=(01DirAM_wider 02AElatentAM_wider 03EnvLatentAM_wider 04EAmigration_wider)
fi

for cond in "${CONDS[@]}"; do
    jid=$(sbatch --parsable "submit_datagenerating_nn_combined_${cond}.sh")
    cid=$(sbatch --parsable --dependency=afterany:"${jid}" submit_combine_nn_wider.sh "${cond}")
    echo "${cond}: simulation array ${jid}, combine job ${cid} (starts after the array)"
done
