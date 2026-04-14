#!/bin/bash
# submit_all_predicted_condition.sh
#
# Wrapper script: submits the simulation array job, then submits a compile job
# that runs automatically once ALL array tasks succeed.
#
# Usage (run from the run_rc directory on the cluster):
#   bash /path/to/run_rc_paper/submit_all_predicted_condition.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_RC_DIR="/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Scripts/run_rc"

mkdir -p "${RUN_RC_DIR}/slurm_logs"

# ── 1. Submit the simulation array job ───────────────────────────────────────
ARRAY_JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/submit_predicted_condition.sh")
if [[ -z "${ARRAY_JOB_ID}" ]]; then
    echo "ERROR: failed to submit array job."
    exit 1
fi
echo "Submitted simulation array job: ${ARRAY_JOB_ID}"

# ── 2. Submit the compile job (runs after all array tasks finish OK) ──────────
COMPILE_JOB_ID=$(sbatch \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --parsable \
    --job-name=compile_pred_cond_uni \
    --qos=preemptable \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=16G \
    --time=01:00:00 \
    --chdir "${RUN_RC_DIR}" \
    --output="${RUN_RC_DIR}/slurm_logs/compile_pred_cond_uni_%j.out" \
    --error="${RUN_RC_DIR}/slurm_logs/compile_pred_cond_uni_%j.err" \
    --wrap "source /curc/sw/anaconda3/latest && conda activate /projects/xuly4739/general_env && python -u ${SCRIPT_DIR}/compile_predicted_condition.py")

if [[ -z "${COMPILE_JOB_ID}" ]]; then
    echo "ERROR: failed to submit compile job."
    exit 1
fi
echo "Submitted compile job:          ${COMPILE_JOB_ID}  (depends on ${ARRAY_JOB_ID})"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${RUN_RC_DIR}/slurm_logs/compile_pred_cond_uni_${COMPILE_JOB_ID}.out"
