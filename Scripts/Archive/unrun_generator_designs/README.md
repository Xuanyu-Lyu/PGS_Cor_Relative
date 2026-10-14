# Generator designs without training data

These versions of the `run_rc_paper/DataGeneratingNN_Combined_*.py` generators were
edited to a new parameter design, but the NN training data in `Scripts/nn_NPE/` (and so
the NPE fits and predicted conditions) come from the earlier design. The generators in
`run_rc_paper/` were set back to the design of the training data on 2026-10-13; these
copies keep the newer designs in case they are simulated later.

| file | design | training data actually used |
|---|---|---|
| `DataGeneratingNN_Combined_01DirAM_design20260908.py` | vg2 = 0.5 and s22 = 0.1 fixed, 20 generations | vg2 and s22 sampled, 15 generations (341 rows of this design were simulated before the run skipped existing conditions; the NPE fits filter them out) |
| `DataGeneratingNN_Combined_03EnvLatentAM_design20261007.py` | prop_h2_latent1, vg1, re, am22, rg sampled; f11 = f22 = 0.10 | vg1, f11, f22, am22, rg sampled; prop_h2_latent1 = 0.7, re = 0.1 |
| `DataGeneratingNN_Combined_05AMmigration_noVT_design20260917.py` | vg1 sampled (0.3-0.8), vg2 = 0.5 | vg1 = 0.6, vg2 = 0.8 fixed |

To use one of these designs: simulate it, copy the combined training CSV to `nn_NPE/`,
update `nn_NPE/run_specs/` and `run_npe_pipeline.py`, retrain, and update the
predicted-condition script.
