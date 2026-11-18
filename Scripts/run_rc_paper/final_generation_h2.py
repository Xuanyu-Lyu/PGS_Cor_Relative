"""
Heritability and PGS R² in the final generations of the predicted conditions.

For every replicate of a predicted condition (run_predicted_condition_rc_<cond>.py, folder
<SCRATCH_BASE>/<CONDITION name>/Iteration_XXX/ on scratch), two generations are read:

  - the youngest generation of the analysed window (FINAL_GENS[-1]: 14 for 01-03, n_gen - 1 for
    04), whose relatives the correlations are computed on, from its individual-level file
    iteration_XXX_phen_gen<g>.tsv (exact);
  - the last simulated generation (15 for 01-03, n_gen for 04), the children of the window's
    last couples, which is not saved per person, from the last row of
    iteration_XXX_covariances_log.tsv (covariances rounded to 3 decimals by the engine).

Per trait, it prints the mean, SD, min and max over replicates of:

  h2          Var(AO + AL) / Var(Y)    heritability: the full additive genetic value, including
                                       2 Cov(AO, AL), which assortative mating and sorted
                                       migration make positive
  h2_engine   (Var(AO) + Var(AL)) / Var(Y)    the engine's 'h2' (summary files), which leaves
                                       out 2 Cov(AO, AL), shown for comparison
  r2_pgs      corr(PGS, Y)²            R² of the phenotype on the PGS (PGS = TPO + TMO = AO,
                                       as in the relative correlations)
  pgs_share   Var(PGS) / Var(Y)        phenotypic variance of the PGS itself
  r_AO_AL     corr(AO, AL)             0 in the founders
  VF/VY, VS/VY, VE/VY                  shares of vertical transmission, shared family
                                       environment and unique environment
  VY          Var(Y)                   1 in the founders

Y = AO + AL + F + S + MT + E (MT = 0 here). Variances and correlations are over everyone in the
generation (all islands together for 04).

Run on the cluster, where the scratch outputs are (it only reads files):
    python final_generation_h2.py                      # the four predicted conditions (01-04)
    python final_generation_h2.py 02AElatentAM         # some conditions
    python final_generation_h2.py --scratch_root DIR   # look for the run folders under DIR
"""

import argparse
import ast
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

RC = Path(__file__).resolve().parent

# condition -> traits to report (trait 2 of 02-04 is fully latent or inert and has no PGS)
TRAITS = {"01DirAM": [1, 2], "02AElatentAM": [1], "03EnvLatentAM": [1], "04EAmigration": [1]}
QUANTITIES = ["h2", "h2_engine", "r2_pgs", "pgs_share", "r_AO_AL", "VF/VY", "VS/VY", "VE/VY", "VY"]
WINDOW, LAST = "youngest analysed generation", "last simulated generation"


def run_folder(cond, scratch_root=None):
    """Replicate folder of a condition, as set in its predicted-condition script."""
    src = (RC / f"run_predicted_condition_rc_{cond}.py").read_text()
    base = Path(re.search(r'^SCRATCH_BASE = Path\("(.*)"\)', src, re.M).group(1))
    name = next(ast.literal_eval(n.value)["name"] for n in ast.parse(src).body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "CONDITION")
    return (Path(scratch_root) / base.name if scratch_root else base) / name


def measures(var, cov, t):
    """The QUANTITIES of trait t from var(name) and cov(name1, name2) of the generation's columns."""
    ao, al, y = f"AO{t}", f"AL{t}", f"Y{t}"
    vy, vao, val, c = var(y), var(ao), var(al), cov(ao, al)
    return {"h2": (vao + val + 2 * c) / vy,
            "h2_engine": (vao + val) / vy,
            "r2_pgs": cov(ao, y) ** 2 / (vao * vy) if vao > 0 else np.nan,
            "pgs_share": vao / vy,
            "r_AO_AL": c / np.sqrt(vao * val) if vao > 0 and val > 0 else np.nan,
            "VF/VY": var(f"F{t}") / vy, "VS/VY": var(f"S{t}") / vy, "VE/VY": var(f"E{t}") / vy,
            "VY": vy}


def replicate_measures(args):
    """Both generations of one replicate folder."""
    iteration_dir, traits = args
    rows = []

    phen = list(iteration_dir.glob("iteration_*_phen_gen*.tsv"))
    if phen:
        gen_of = lambda f: int(re.search(r"_phen_gen(\d+)\.tsv$", f.name).group(1))
        path = max(phen, key=gen_of)
        cols = [f"{c}{t}" for t in traits for c in ("AO", "AL", "F", "S", "E", "Y", "TPO", "TMO")]
        d = pd.read_csv(path, sep="\t", usecols=cols).astype(float)
        for t in traits:
            # the PGS of the correlations is TPO + TMO; it equals AO, checked here
            if not np.allclose(d[f"TPO{t}"] + d[f"TMO{t}"], d[f"AO{t}"], atol=1e-8):
                raise ValueError(f"{path}: TPO{t} + TMO{t} differs from AO{t}")
        C = d.cov()
        for t in traits:
            rows.append({"generation": gen_of(path), "source": WINDOW, "trait": t,
                         **measures(lambda a: C.loc[a, a], lambda a, b: C.loc[a, b], t)})

    log = list(iteration_dir.glob("iteration_*_covariances_log.tsv"))
    if log:
        last = pd.read_csv(log[0], sep="\t")
        # covariances of column a: a dict literal that can hold nan, so not ast.literal_eval
        cell = lambda a: eval(last[a].iloc[-1], {"__builtins__": {}}, {"nan": np.nan, "inf": np.inf})
        cache = {}
        get = lambda a: cache.setdefault(a, cell(a))
        for t in traits:
            # the log has one row per simulated generation 1..N (no founders), so its length is N
            rows.append({"generation": len(last), "source": LAST, "trait": t,
                         **measures(lambda a: get(a)[a], lambda a, b: get(a)[b], t)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("conditions", nargs="*", default=list(TRAITS), choices=list(TRAITS))
    ap.add_argument("--scratch_root", default=None,
                    help="folder holding the predicted_condition_<cond>_<run> folders "
                         "(default: SCRATCH_BASE of each predicted-condition script)")
    ap.add_argument("--jobs", type=int, default=4, help="replicates read in parallel")
    args = ap.parse_args()

    summaries = []
    for cond in args.conditions:
        folder = run_folder(cond, args.scratch_root)
        dirs = sorted(folder.glob("Iteration_*"))
        print(f"\n=== {cond}: {folder}")
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = [r for rs in pool.map(replicate_measures, [(d, TRAITS[cond]) for d in dirs]) for r in rs]
        if not rows:
            print("  no replicate outputs found")
            continue
        df = pd.DataFrame(rows)
        for t, by_trait in df.groupby("trait"):
            print(f"\n  trait {t}: mean (SD) [min, max] over replicates; window = {WINDOW}, last = {LAST}")
            table = {}
            for source in (WINDOW, LAST):
                g = by_trait[by_trait["source"] == source]
                if g.empty:
                    continue
                gens = sorted(g["generation"].unique())
                label = f"gen {'/'.join(map(str, gens))} ({'window' if source == WINDOW else 'last'}, n = {len(g)})"
                table[label] = {q: f"{g[q].mean():.4f} ({g[q].std():.4f}) [{g[q].min():.3f}, {g[q].max():.3f}]"
                                for q in QUANTITIES}
                summaries.append({"condition": cond, "trait": t, "generation": gens[-1], "source": source,
                                  "replicates": len(g), **{q: round(g[q].mean(), 4) for q in QUANTITIES},
                                  **{f"{q}_sd": round(g[q].std(), 4) for q in ("h2", "r2_pgs")}})
            print("  " + pd.DataFrame(table).to_string().replace("\n", "\n  "))

    if summaries:
        print("\n=== Summary (means over replicates; _sd: SD over replicates)")
        print(pd.DataFrame(summaries).to_csv(index=False).rstrip())


if __name__ == "__main__":
    sys.exit(main())
