# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase that does simulation-based inference on correlations of EA polygenic scores and phenotypes between relatives, in-laws included. Several forward-time models of assortative mating, vertical transmission and island migration are simulated. An NPE (`sbi`, SNPE-C) is trained on the simulated correlations and conditioned on the observed EA correlations. Each model is then re-simulated at its estimates (the "predicted condition") and compared with the data.

**[README.md](README.md) is the main documentation.** It covers the model equations, the relationship paths, the table of conditions with ranges and fixed values, the NPE details, the history of runs (v1, v2, ea4new, ea4pheno, wider, rescaled) and known issues. Nearly every commit updates it. When you change a design, fit, run or output name, update the matching README section in the same change, in its plain, precise style.

## Environments

- **Local (macOS):** `python3` is the conda env `miracle` (Python 3.12, `sbi` 0.26, `torch` 2.11). NPE training and prediction, the notebooks and small diagnostic simulations run here.
- **Cluster (CURC Alpine, SLURM):** training-data generation and predicted-condition simulations run here. The submit scripts activate `/projects/xuly4739/general_env`. The generators and `run_predicted_condition_rc_*.py` hard-code absolute cluster paths (`/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/...`, `/scratch/alpine/xuly4739/...`), so they do not run locally as they are. For local simulations, use the small-population wrapper `Scripts/run_rc_paper_wider/diagnose_02_03/sim_driver.py`.
- The repo has no build step, linter or package config. The scripts are standalone and run from their own directory.

## Commands

```bash
# Simulation-engine tests (inside the submodule)
cd Scripts/SimulationFunctions && python -m pytest tests/test_simulation.py -v
python -m pytest tests/test_simulation.py -v -k <test_name>     # a single test

# NPE: train + predict (local, from Scripts/nn_NPE)
python run_npe_pipeline.py --runs 02AElatentAM_wider --jobs 1
python run_npe_pipeline.py --skip_train --runs 04EAmigration     # re-predict only
#   other flags: --observed_pgs, --observed_pheno, --pred_dir, --archive_tag, --extra_train_args

# Cluster (login node, after ./sync_cluster_submodule.sh)
cd Scripts/run_rc_paper_rescaled && bash submit_all_rescaled.sh [<cond>_rescaled ...]
cd Scripts/run_rc_paper && bash submit_all_predicted_conditions.sh
python compile_predicted_condition.py          # after predicted-condition jobs finish
python check_predicted_conditions.py           # run after any refit
```

The training CSVs and fit outputs reach the local machine with `rsync` from `<cluster>/Data/to_nn_NPE/` into `Scripts/nn_NPE/` (README §4.1).

## Architecture

The pipeline alternates between the cluster and the local machine:

1. **Generate** (cluster): `DataGeneratingNN_Combined_<cond>.py` is a SLURM array. Each task simulates a block of parameter draws. The draws are seeded and written once to `conditions_config.csv`. Each task writes one row of correlations per condition.
2. **Combine** (cluster): `CombineNN_Data_Large_*.py` produces `nn_training_<cond>.csv`.
3. **Fit** (local): `Scripts/nn_NPE/run_npe_pipeline.py` drives `train_npe_pgs_and_pheno.py` and `predict_npe_pgs_and_pheno.py`, plus `npe_count.py` for the integer `n_gen` of condition 04. It writes `results_npe_unweighted_<fit>/`.
4. **Predicted condition** (cluster): `run_rc_paper/run_predicted_condition_rc_<cond>.py` holds the posterior estimates in a hand-edited `CONDITION` dict and runs 100 replicates. `compile_predicted_condition.py` produces `summary_statistics_<...>.csv`.
5. **Compare** (local): the `Scripts/nn_NPE/Analysis_NPE*.ipynb` notebooks.

### How names connect the files

A condition ID (`01DirAM`, `02AElatentAM`, `03EnvLatentAM`, `04EAmigration`, `05AMmigration_noVT`) plus a design suffix (`_wider`, `_rescaled`, `_v2`) carries through every stage:
- the generator and submit-script file names;
- `Data/DataGeneratingNN_Paper*/<cond>/`;
- `nn_training_<data>.csv`;
- the `RUNS` key in `run_npe_pipeline.py`;
- `run_specs/<fit>/{features,params}.txt`;
- `results_npe_unweighted_<fit>/`;
- the predicted-condition script and its output folder;
- the summary file names the notebooks read.

A fit name can differ from its data name. Condition 01 is fitted twice from the same data, as `01DirAM_AE` (trait 1) and `01DirAM_AFE` (trait 2). Any fit can also have a `_PGSonly` variant. When you rename or add something, follow it through every stage.

- **Adding a fit:** add a `RUNS` entry (data name, trait, extra `train_args` such as `--group_col Condition`, `--count_param n_gen`, `--row_filter ...`). Create `run_specs/<fit>/features.txt`, with feature names `<path>_PGS<t>` / `<path>_Y<t>` such as `MSC_PGS1`, and `params.txt`. The training CSV must be in `Scripts/nn_NPE/`.
- **Adding or changing a design:** copy the closest generator into a new file with a new output folder, and `diff` it against its parent. Generators are near-duplicates by design (`run_rc_paper` → `_wider` → `_rescaled`). Each generator refuses an existing `conditions_config.csv` from a different design. The `_wider` and `_rescaled` generators also skip conditions already simulated with the same parameters, so re-running in an old folder keeps the old rows.
- **After a refit:** update the `CONDITION` dicts in the predicted-condition scripts to the new posterior means (mode for `n_gen`). Then run `check_predicted_conditions.py`, which checks the dicts against the fits and the generators' fixed values.
- **Pipeline archiving:** `run_npe_pipeline.py` moves an existing `results_npe_unweighted_<fit>/` to `results_archive/<tag>/`. It never deletes, and it raises an error if that archive folder already exists.

### Simulation engine (git submodule)

`Scripts/SimulationFunctions/` is the submodule [GeneEvolve-Python](https://github.com/Xuanyu-Lyu/GeneEvolve-Python). It contains `core_simulation.AssortativeMatingSimulation`, `island_migration.IslandMigrationSimulation`, `relationship_finder` and `postprocessing`. Its own `README.md` documents its parameters and gotchas. Scripts import it via `sys.path.insert(...)`. Engine changes belong in that repo. Bump the pin locally with `./update_simulation_functions.sh`, push, then run `./sync_cluster_submodule.sh` on the CURC login node. Compute nodes have no reliable internet access, and jobs fail with `ModuleNotFoundError: core_simulation` if the submodule is not synced.

## Pitfalls

- **Engine fix of 2026-11-08.** Before this fix, founder genetic variance was ≈ 1 whatever `vg` was (founder h² ≈ 1/(2 − vg)), and `s_mat` was ignored. Every result up to and including the `_wider` designs uses that old scale. Re-running an older script now uses the fixed engine and does not reproduce those results. The `_rescaled` designs are the re-run. Some docstrings, such as the one in `diagnose_02_03/sim_driver.py`, still describe the old behaviour.
- **Data and results are not in git.** `*.csv`, `*.png`, `results_*/` and `predictions/` are gitignored. Training data, fits and summaries exist only on disk locally and on the cluster.
- **Combine timing.** The older `run_rc_paper` submit scripts combine from the last array task, which can start before other tasks finish. Re-run the combine step by hand. The `_wider` and `_rescaled` designs use a separate dependent combine job, which writes nothing while any condition is missing.
- **Relationship paths** use the steps M/P/C/S. `PSC` is first cousin and `PPSCC` is second cousin, although some code comments say otherwise. `HSFS` is listed but never computed. `PSCC` and `PPSCC` do not exist at `n_gen = 3`, so `04EAmigration_wider` filters to `n_gen >= 4`.
- **Condition 04** gives 8 rows per simulation (`n_gen` 3–10). Its splits must group on `Condition`.

## Folders outside the main pipeline

- `Scripts/Archive/`: retired conditions and generator designs that were never run.
- `Scripts/fit_nn/`, `Scripts/run_rc/`: the earlier PyTorch-regression approach and its cluster scripts. `nn_NPE` and `run_rc_paper*` replace them.
- `Scripts/nn_validate/`: a separate validation study of NPE against OpenMx (ACE, SEM-PGS), with its own README.
- `Scripts/test_04EAmigration/`: identifiability and architecture benchmarks for condition 04.
- `Scripts/run_rc_paper_wider/diagnose_02_03/`: the local diagnosis of conditions 02 and 03 that led to the engine fix and to the fixed values of condition 03 in the `_rescaled` design.
- `Scripts/tmp_quick_sim/`: one-off quick simulations.

## Git

This is a solo research repo with a direct-to-main history. Commit and push to `main`, not to a feature branch, because the cluster checkout runs the scripts from `main`.
