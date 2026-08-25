# PGS_Cor_Relative
Simulation code for the consequence of various genetic and environmental effects on PGS correlations on distant relatives.

## SimulationFunctions dependency (git submodule)

`Scripts/SimulationFunctions/` is a **git submodule** pointing at
[GeneEvolve-Python](https://github.com/Xuanyu-Lyu/GeneEvolve-Python), pinned to a specific
commit. That's a separate repo so the simulation engine (assortative mating, island migration,
relationship finding, post-processing) can be reused across multiple projects. It is not a plain
directory of committed files — the actual `.py` files live in the submodule's own git history, and
this repo only records which commit to check out.

### First-time setup

If you're cloning `PGS_Cor_Relative` fresh:
```bash
git clone --recurse-submodules https://github.com/Xuanyu-Lyu/PGS_Cor_Relative.git
```

If you already have a clone without the submodule initialized (e.g. `Scripts/SimulationFunctions/`
looks empty or missing files):
```bash
git submodule update --init --recursive
```

### The CURC / run_rc part — read this before submitting jobs

The SLURM jobs under `Scripts/run_rc/` and `Scripts/run_rc_paper/` (e.g.
`submit_predicted_condition_04AMmigration.sh`) import from `Scripts/SimulationFunctions/` at
runtime via `sys.path.insert(...)`. If the submodule hasn't been initialized or synced on the
cluster checkout, those jobs will fail immediately with `ModuleNotFoundError: No module named
'core_simulation'` (or similar).

CURC compute nodes generally don't have reliable outbound internet, so **never rely on fetching
from GitHub at job runtime** — the submodule must be pulled down once from a node that does have
internet (the login node) before you submit jobs. Run `sync_cluster_submodule.sh` (see below) on
the login node:
- after a fresh clone of this repo on the cluster,
- and again any time the submodule's pinned commit changes upstream (i.e. after someone runs
  `update_simulation_functions.sh` and pushes).

### Updating to a newer GeneEvolve-Python commit

Two scripts at the repo root handle this — see [Maintenance scripts](#maintenance-scripts) below.
Short version: run `update_simulation_functions.sh` locally, review, push, then run
`sync_cluster_submodule.sh` on the CURC login node.

## Maintenance scripts

- **`update_simulation_functions.sh`** — run locally to bump the `Scripts/SimulationFunctions`
  submodule to the latest commit on GeneEvolve-Python's `main` branch, and commit that pointer
  change. Does not push automatically.
- **`sync_cluster_submodule.sh`** — run on the CURC login node to pull the latest
  `PGS_Cor_Relative` commits and sync the submodule checkout to match. Run this before submitting
  SLURM jobs whenever the submodule pin has changed.
