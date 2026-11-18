#!/bin/bash
# Submit the training-data simulations of condition 04 with the founder heritability of EA fixed
# at 0.4 (04EAmigration_h2fixed), followed by a combine job that starts once the whole array has
# finished. Run on the CURC login node, after sync_cluster_submodule.sh (the design needs the
# corrected engine):
#     bash submit_all_h2fixed.sh
#
# The generator writes to Data/DataGeneratingNN_Paper_h2fixed/04EAmigration_h2fixed/ and refuses
# to start if an existing conditions_config.csv there was written for another design.
# Resubmitting skips the conditions that are already simulated.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p slurm_logs

if [ $# -gt 0 ]; then
    CONDS=("$@")
else
    CONDS=(04EAmigration_h2fixed)
fi

for cond in "${CONDS[@]}"; do
    jid=$(sbatch --parsable "submit_datagenerating_nn_combined_${cond}.sh")
    cid=$(sbatch --parsable --dependency=afterany:"${jid}" submit_combine_nn_h2fixed.sh "${cond}")
    echo "${cond}: simulation array ${jid}, combine job ${cid} (starts after the array)"
done
