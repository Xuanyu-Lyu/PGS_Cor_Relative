"""
Small-population wrapper around the unmodified simulation engine, used only for the
diagnostics in this folder (see diagnose_02_03.ipynb).

Two ways of setting the genetic variance:

  scale="engine"  the n_CV route the production generators use. Before the engine fix of
                  2026-11-08 this left each trait's genetic variance at about 1, whatever vg1 and
                  vg2 were (the "production scale" results saved in results/ and shown in the
                  notebook). Since that fix the engine scales the effects from d_mat and a_mat,
                  so this route now gives the same scale as "true".
  scale="true"    the effect sizes are multiplied by sqrt(vg), so the base genetic variance
                  really is vg and the founders' phenotypic variance is 1. Done by handing the
                  engine a ready-made cv_info table; no engine code changes.

Optional c1 / c2 add a shared family environment (one draw per sibship, not inherited except
through the phenotype), through the engine's _post_offspring_hook. The engine itself has no such
component: s_mat is stored but never applied.
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "SimulationFunctions"))

from core_simulation import AssortativeMatingSimulation  # noqa: E402
from postprocessing import extract_individual_measures  # noqa: E402
from relationship_finder import find_relationship_pairs  # noqa: E402

# The 26 relationship types of the observed EA4 data
RELS = ["S", "PSC", "PPSCC", "M", "MS", "SMS", "MSC", "MSM", "SMSC", "SMSM", "SMSMS", "MSMSM",
        "MSMSC", "PSMSC", "SMSMSC", "MSMSMS", "SC", "PSCC", "MPS", "MSCM", "SMPS", "MPSC", "SCMP",
        "MPSCM", "MSMPS", "SMSCM"]

DEFAULTS = dict(vg1=0.35, vg2=0.5, prop_h2_latent1=0.7, prop_h2_latent2=1.0, rg=0.0, re=0.0,
                am22=0.0, f11=0.0, f22=0.0, c1=0.0, c2=0.0, rc=1.0, s=0.0)


class FamilyEnvSimulation(AssortativeMatingSimulation):
    """Adds one shared-environment draw per sibship to Y (and the E column)."""

    def __init__(self, *args, c_mat=None, **kwargs):
        self.c_mat = None if c_mat is None else np.asarray(c_mat, dtype=float)
        super().__init__(*args, **kwargs)

    def _post_offspring_hook(self, offspring_data, gen_abs_idx):
        phen = offspring_data["PHEN"]
        if self.c_mat is None or phen.empty:
            return offspring_data
        _, family = np.unique(phen["Father.ID"].to_numpy(float), return_inverse=True)
        draws = np.random.multivariate_normal(np.zeros(2), self.c_mat, size=family.max() + 1)
        for j, trait in enumerate(("1", "2")):
            phen["Y" + trait] = phen["Y" + trait].to_numpy(float) + draws[family, j]
            phen["E" + trait] = phen["E" + trait].to_numpy(float) + draws[family, j]
        return offspring_data


def build_matrices(p, n_gen):
    """Engine inputs for one parameter set, following the production setup_matrices."""
    vg1, vg2 = p["vg1"], p["vg2"]
    ve1, ve2 = 1 - vg1 - p["c1"], 1 - vg2 - p["c2"]
    if ve1 <= 0 or ve2 <= 0:
        raise ValueError(f"non-positive environmental variance: ve1={ve1}, ve2={ve2}")
    d = np.diag([np.sqrt(vg1 * (1 - p["prop_h2_latent1"])), np.sqrt(vg2 * (1 - p["prop_h2_latent2"]))])
    a = np.diag([np.sqrt(vg1 * p["prop_h2_latent1"]), np.sqrt(vg2 * p["prop_h2_latent2"])])
    k2 = np.array([[1, p["rg"]], [p["rg"], 1]])
    cove = p["re"] * np.sqrt(ve1 * ve2)
    cove_mat = np.array([[ve1, cove], [cove, ve2]])
    return dict(cove_mat=cove_mat, f_mat=np.diag([p["f11"], p["f22"]]), s_mat=np.full((2, 2), float(p["s"])),
                a_mat=a, d_mat=d, am_list=[p["am22"]] * n_gen, mate_on_trait=2,
                covy_mat=d @ k2 @ d.T + a @ k2 @ a.T + cove_mat, k2_matrix=k2)


def relationship_correlations(trimmed, rels):
    """Pearson correlation of PGS1 and Y1 over each relationship's pairs (as postprocessing does)."""
    ind = extract_individual_measures(trimmed, ["Y1", "Y2", "TPO1", "TPO2", "TMO1", "TMO2"])
    ind["PGS1"] = ind["TPO1"] + ind["TMO1"]
    ind = ind.drop_duplicates("ID").set_index("ID")
    out = {}
    for rel in rels:
        try:
            with contextlib.redirect_stdout(io.StringIO()):  # the finder prints per-path progress
                pairs = find_relationship_pairs(trimmed, rel, output_format="long", generations=[0, 1, 2])
        except Exception:
            continue
        if len(pairs) < 10:
            continue
        for var in ("PGS1", "Y1", "Y2"):
            x = ind[var].reindex(pairs["Person_ID"]).to_numpy(float)
            y = ind[var].reindex(pairs["Relative_ID"]).to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            out[f"{rel}_{var}"] = float(np.corrcoef(x[ok], y[ok])[0, 1])
    return out


def variance_components(phen, tag):
    """Realised variance components of one generation's data frame."""
    col = {c: phen[c].to_numpy(float) for c in ["AO1", "AL1", "AO2", "AL2", "F1", "F2", "E1", "E2", "Y1", "Y2"]}
    g1, g2 = col["AO1"] + col["AL1"], col["AO2"] + col["AL2"]
    out = {"VG1": g1.var(), "VAO1": col["AO1"].var(), "VAL1": col["AL1"].var(), "VF1": col["F1"].var(),
           "VE1": col["E1"].var(), "VP1": col["Y1"].var(), "VG2": g2.var(), "VP2": col["Y2"].var(),
           "h2_1": g1.var() / col["Y1"].var(), "h2_2": g2.var() / col["Y2"].var(),
           "h2obs_1": col["AO1"].var() / col["Y1"].var(),
           "r_Y1Y2": np.corrcoef(col["Y1"], col["Y2"])[0, 1], "rg_real": np.corrcoef(g1, g2)[0, 1]}
    return {f"{k}_{tag}": float(v) for k, v in out.items()}


def simulate(params, scale="engine", N=8000, n_cv=500, n_gen=15, seed=1, rels=RELS):
    """Run one simulation and return the correlations and realised variance components."""
    p = {**DEFAULTS, **params}
    final_gens = [n_gen - 3, n_gen - 2, n_gen - 1]
    matrices = build_matrices(p, n_gen)
    np.random.seed(seed)  # the engine draws its variants before it applies `seed`
    c_mat = None
    if p["c1"] > 0 or p["c2"] > 0:
        cc = p["rc"] * np.sqrt(p["c1"] * p["c2"])
        c_mat = [[p["c1"], cc], [cc, p["c2"]]]

    common = dict(num_generations=n_gen, pop_size=N, mating_type="phenotypic", avoid_inbreeding=True,
                  save_each_gen=True, save_covs=False, seed=seed, output_summary_filename=None, c_mat=c_mat)
    with contextlib.redirect_stdout(io.StringIO()):
        if scale == "engine":
            sim = FamilyEnvSimulation(n_CV=n_cv, rg_effects=p["rg"], maf_min=0.01, maf_max=0.5, **common, **matrices)
        elif scale == "true":
            cv = AssortativeMatingSimulation.prepare_CV_random_selection(
                n_cv, p["rg"], 0.01, 0.5, 1 - p["prop_h2_latent1"], 1 - p["prop_h2_latent2"])
            cv["alpha1"] *= np.sqrt(p["vg1"])
            cv["alpha2"] *= np.sqrt(p["vg2"])
            sim = FamilyEnvSimulation(cv_info=cv, **common, **matrices)
        else:
            raise ValueError(scale)
        results = sim.run_simulation()

    history = results["HISTORY"]
    n_mates = len(history["MATES"])
    mates_idx = final_gens + [final_gens[-1] + 1] if final_gens[-1] + 1 < n_mates else final_gens
    trimmed = {"HISTORY": {"PHEN": [history["PHEN"][i] for i in final_gens],
                           "XO": [history["XO"][i] for i in final_gens],
                           "XL": [history["XL"][i] for i in final_gens],
                           "MATES": [history["MATES"][i] if i < n_mates else None for i in mates_idx]}}
    out = {"scale": scale, "N": N, "n_cv": n_cv, "n_gen": n_gen, "seed": seed, **{f"param_{k}": v for k, v in p.items()}}
    out.update(variance_components(history["PHEN"][0], "gen0"))
    out.update(variance_components(history["PHEN"][final_gens[-1]], "final"))
    out.update(relationship_correlations(trimmed, rels))
    return out


if __name__ == "__main__":
    import time
    t = time.time()
    r = simulate(dict(vg1=0.1, vg2=0.5, prop_h2_latent1=0.7, rg=0.7, re=0.5, am22=0.7), scale="engine", seed=3)
    print(f"{time.time() - t:.1f} s")
    print({k: round(v, 3) for k, v in r.items() if k.endswith("_gen0") or k in ("S_Y1", "M_Y1", "S_PGS1", "M_PGS1")})
    print("correlations returned:", sum(k.endswith("_Y1") for k in r))
