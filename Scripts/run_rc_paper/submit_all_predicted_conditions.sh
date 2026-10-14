#!/bin/bash
# Submit every predicted-condition job (submit_predicted_condition_*.sh).
# Run on the CURC login node, after sync_cluster_submodule.sh:
#     bash submit_all_predicted_conditions.sh
#
# Output folders of earlier runs of the same condition are renamed to
# <folder>_before_<timestamp> first, so compile_predicted_condition.py never mixes
# task files simulated at old parameter values with the new ones.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p slurm_logs
STAMP=$(date +%Y%m%d-%H%M)

for submit in submit_predicted_condition_*.sh; do
    py=$(grep -o 'run_predicted_condition_rc_[A-Za-z0-9_]*\.py' "$submit")
    for base_var in SCRATCH_BASE PROJECT_BASE; do
        base=$(sed -n "s/^${base_var} = Path(\"\(.*\)\")/\1/p" "$py")
        if [ -n "$base" ] && [ -d "$base" ]; then
            mv "$base" "${base}_before_${STAMP}"
            echo "moved earlier output: $base -> ${base}_before_${STAMP}"
        fi
    done
    echo "submitting $submit ($py)"
    sbatch "$submit"
done
