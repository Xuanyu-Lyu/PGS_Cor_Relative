#!/bin/bash
# Pulls the latest PGS_Cor_Relative commits and syncs the SimulationFunctions
# submodule (GeneEvolve-Python) to whatever commit is pinned in that pull.
#
# Run this on the CURC cluster (login node, which has internet access --
# compute nodes generally don't) before submitting SLURM jobs under
# Scripts/run_rc/ or Scripts/run_rc_paper/, whenever the submodule pin has
# changed upstream. Jobs import from Scripts/SimulationFunctions/ at runtime
# and will fail with ModuleNotFoundError if it isn't synced.
#
# Usage: ./sync_cluster_submodule.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "Pulling latest PGS_Cor_Relative..."
git pull

echo ""
echo "Syncing submodules (SimulationFunctions / GeneEvolve-Python)..."
git submodule sync --recursive
git submodule update --init --recursive

echo ""
echo "Submodule status:"
git submodule status

echo ""
echo "Done. SLURM jobs submitted from this checkout will now use this version."
