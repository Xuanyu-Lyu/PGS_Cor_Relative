"""
Check every run_predicted_condition_rc_*.py against the NPE fits and the training data.

For each predicted-condition script:
  1. every estimated parameter equals the posterior mean of its NPE fit (4 decimals;
     a count parameter such as n_gen equals the posterior mode);
  2. every parameter sampled in the training data (generator PARAM_BOUNDS, and the
     param_ columns actually present in the training CSV after the run's row filter)
     is estimated by one of the fits, and every estimate lies inside that range;
  3. every other parameter in CONDITION equals the generator's FIXED_PARAMS value;
  4. N_GENERATIONS / FINAL_GENS equal the generator's (04: n_gen and the last 3 before it).

Usage (from anywhere):
    python check_predicted_conditions.py
Exit status 1 if anything fails.
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RC = Path(__file__).resolve().parent
NPE = RC.parent / "nn_NPE"

# script tag -> (generator tag, training CSV tag, NPE runs supplying its estimates)
SCRIPTS = {
    "01DirAM":                     ("01DirAM", "01DirAM", ["01DirAM_AE", "01DirAM_AFE"]),
    "01DirAM_PGSonly":             ("01DirAM", "01DirAM", ["01DirAM_AE_PGSonly", "01DirAM_AFE_PGSonly"]),
    "02AElatentAM":                ("02AElatentAM", "02AElatentAM", ["02AElatentAM"]),
    "02AElatentAM_PGSonly":        ("02AElatentAM", "02AElatentAM", ["02AElatentAM_PGSonly"]),
    "03EnvLatentAM":               ("03EnvLatentAM", "03EnvLatentAM", ["03EnvLatentAM"]),
    "03EnvLatentAM_PGSonly":       ("03EnvLatentAM", "03EnvLatentAM", ["03EnvLatentAM_PGSonly"]),
    "04EAmigration":               ("04EAmigration", "04EAmigration", ["04EAmigration"]),
    "05AMmigration_noVT":          ("05AMmigration_noVT", "05AMmigration_noVT", ["05AMmigration_noVT2"]),
    "05AMmigration_noVT2_PGSonly": ("05AMmigration_noVT", "05AMmigration_noVT", ["05AMmigration_noVT2_PGSonly"]),
}


def assignments(path):
    """Top-level NAME = <value> assignments of a Python file, without running it.

    Literals are read directly; simple expressions of earlier assignments
    (e.g. list(range(3, N_GEN_MAX + 1)), int(CONDITION['n_gen'])) are evaluated.
    """
    out = {}
    safe = {"__builtins__": {}, "list": list, "range": range, "int": int}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                try:
                    out[node.targets[0].id] = eval(compile(ast.Expression(node.value), str(path), "eval"),
                                                   safe, dict(out))
                except Exception:
                    pass
    return out


def main():
    failures = []
    data_cache = {}

    def check(ok, msg):
        print(f"  {'ok  ' if ok else 'FAIL'} {msg}")
        if not ok:
            failures.append(msg)

    for tag, (gen_tag, data_tag, runs) in SCRIPTS.items():
        print(f"\n=== run_predicted_condition_rc_{tag}.py")
        pred = assignments(RC / f"run_predicted_condition_rc_{tag}.py")
        gen = assignments(RC / f"DataGeneratingNN_Combined_{gen_tag}.py")
        cond, bounds, fixed = pred["CONDITION"], gen["PARAM_BOUNDS"], gen["FIXED_PARAMS"]

        # posterior means / modes of the fits
        estimates, row_filter = {}, None
        for run in runs:
            run_dir = NPE / f"results_npe_unweighted_{run}"
            config = json.loads((run_dir / "config.json").read_text())
            row_filter = config.get("row_filter")
            summary = pd.read_csv(run_dir / "predictions" / "posterior_summary.csv")
            means = dict(zip(summary["parameter"], summary["mean"]))
            if config.get("count_param"):
                pmf = pd.read_csv(run_dir / "predictions" / "posterior_count_pmf.csv")
                means[config["count_param"]] = int(pmf.iloc[pmf["probability"].idxmax(), 0])
            sampling = run_dir / "predictions" / "posterior_sampling.json"
            share = json.loads(sampling.read_text())["share_of_unconstrained_flow_inside_prior"] \
                if sampling.exists() else float("nan")
            print(f"  {run}: {share:.1%} of the unconstrained posterior inside the simulated range")
            estimates.update(means)

        # 1. estimated values
        for p, m in estimates.items():
            expected = m if isinstance(m, int) else round(float(m), 4)
            check(p in cond and np.isclose(cond[p], expected, atol=5e-5),
                  f"{p} = {cond.get(p)} (posterior {expected})")

        # 2. everything sampled in the training data is estimated, inside its range
        if data_tag not in data_cache:
            data_cache[data_tag] = pd.read_csv(NPE / f"nn_training_{data_tag}.csv")
        df = data_cache[data_tag]
        if row_filter:
            df = df.query(row_filter, engine="python")
        sampled = {c[6:] for c in df.columns if c.startswith("param_") and df[c].notna().any()}
        check(sampled == set(bounds) | ({"n_gen"} if "GEN_VALUES" in gen else set()),
              f"training data sample {sorted(sampled)} = generator PARAM_BOUNDS")
        for p in sorted(sampled):
            lo, hi = df[f"param_{p}"].min(), df[f"param_{p}"].max()
            check(p in estimates and lo - 1e-4 <= cond.get(p, np.nan) <= hi + 1e-4,
                  f"{p} estimated and inside the simulated range [{lo:.3f}, {hi:.3f}]")

        # 3. fixed values
        for p, v in fixed.items():
            check(p in cond and np.isclose(cond[p], v), f"{p} = {cond.get(p)} (fixed {v} in the generator)")
        others = sorted(set(cond) - set(fixed) - set(estimates) - {"name"})
        for p in others:
            check(cond[p] == 0, f"{p} = {cond[p]} (not in the generator; engine default 0)")

        # 4. generations
        if "GEN_VALUES" in gen:
            n_gen = int(cond["n_gen"])
            check(n_gen in gen["GEN_VALUES"], f"n_gen = {n_gen} is one of the simulated {gen['GEN_VALUES']}")
            check(pred.get("N_GENERATIONS") == n_gen and pred.get("FINAL_GENS") == list(range(n_gen - 3, n_gen)),
                  f"N_GENERATIONS {pred.get('N_GENERATIONS')} / FINAL_GENS {pred.get('FINAL_GENS')} "
                  f"(n_gen {n_gen}: generations {list(range(n_gen - 3, n_gen))}, as in the training rows)")
        else:
            check(pred.get("N_GENERATIONS") == gen["N_GENERATIONS"] and pred.get("FINAL_GENS") == gen["FINAL_GENS"],
                  f"N_GENERATIONS {pred.get('N_GENERATIONS')} / FINAL_GENS {pred.get('FINAL_GENS')} "
                  f"(generator {gen['N_GENERATIONS']} / {gen['FINAL_GENS']})")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
