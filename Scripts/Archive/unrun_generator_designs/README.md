# Generator designs without training data

These versions of the `run_rc_paper/DataGeneratingNN_Combined_*.py` generators were
edited to a new parameter design, but the NN training data in `Scripts/nn_NPE/` (and so
the NPE fits and predicted conditions) come from the earlier design. The generators in
`run_rc_paper/` were set back to the design of the training data on 2026-10-13; these
copies keep the newer designs in case they are simulated later.

| file | design | training data actually used |
|---|---|---|
| `DataGeneratingNN_Combined_01DirAM_design20260908.py` | vg2 = 0.5 and s22 = 0.1 fixed, 20 generations | vg2 and s22 sampled, 15 generations (341 rows of this design were simulated before the run skipped existing conditions; the NPE fits filter them out) |
| `DataGeneratingNN_Combined_05AMmigration_noVT_design20260917.py` | vg1 sampled (0.3-0.8), vg2 = 0.5 | vg1 = 0.6, vg2 = 0.8 fixed |

To use one of these designs: simulate it, copy the combined training CSV to `nn_NPE/`,
update `nn_NPE/run_specs/` and `run_npe_pipeline.py`, retrain, and update the
predicted-condition script.

The condition-03 design of 2026-10-07 (prop_h2_latent1, vg1, re, am22, rg sampled; f11 = f22 =
0.10) moved back to `run_rc_paper/DataGeneratingNN_Combined_03EnvLatentAM_v2.py` on 2026-10-19,
with its own output folder (`03EnvLatentAM_v2`). Its first cluster run reused the first design's
`conditions_config.csv` and condition folders, so none of that output was simulated under the
new design.
