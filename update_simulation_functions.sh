#!/bin/bash
# Bumps the SimulationFunctions submodule (GeneEvolve-Python) to the latest
# commit on its main branch, and commits that change locally.
#
# Run this locally (wherever you have push access to origin) whenever you
# want to pick up new commits from https://github.com/Xuanyu-Lyu/GeneEvolve-Python.
#
# Usage: ./update_simulation_functions.sh
#
# After running: review with `git log -1 -p`, then `git push`, then run
# sync_cluster_submodule.sh on the CURC login node (or the equivalent manual
# `git pull && git submodule update --init --recursive`).

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

SUBMODULE_PATH="Scripts/SimulationFunctions"

if [ ! -f .gitmodules ] || ! grep -q "$SUBMODULE_PATH" .gitmodules; then
    echo "Error: $SUBMODULE_PATH is not configured as a git submodule in this repo." >&2
    exit 1
fi

echo "Current pinned commit:"
git submodule status "$SUBMODULE_PATH"

echo ""
echo "Fetching latest commit on GeneEvolve-Python's main branch..."
git submodule update --remote "$SUBMODULE_PATH"

if git diff --quiet -- "$SUBMODULE_PATH"; then
    echo ""
    echo "Already up to date -- nothing to commit."
    exit 0
fi

NEW_COMMIT=$(git -C "$SUBMODULE_PATH" rev-parse --short HEAD)
git add "$SUBMODULE_PATH"
git commit -m "Bump SimulationFunctions (GeneEvolve-Python) submodule to ${NEW_COMMIT}"

echo ""
echo "Updated pinned commit:"
git submodule status "$SUBMODULE_PATH"

echo ""
echo "Committed locally. Next steps:"
echo "  1. git push"
echo "  2. On the CURC login node: ./sync_cluster_submodule.sh"
echo "     (or manually: git pull && git submodule update --init --recursive)"
