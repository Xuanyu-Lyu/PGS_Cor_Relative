"""
test_equilibrium.py
===================
Equilibrium probe for the condition-04 (AM + island migration) model.

Purpose
-------
Under assortative mating + vertical transmission the phenotypic variance can
keep inflating and never settle by the final generation. This script sweeps
each parameter across its range (holding the others at the baseline midpoint),
runs a small fast simulation, and classifies the variance trajectory as:

    equilibrium : variance plateaus (small drift over the final generations)
    exploding   : variance is still climbing / has run away at the end
    collapsing  : variance shrinks toward zero

It then reports, per parameter, the sub-range that stays at equilibrium so you
can decide which parameters to fix or narrow. ``move_p`` is always reported as
free (it is a target you want to estimate, never fixed here).

It is independent of test_identifiability.py.

Usage
-----
    python test_equilibrium.py
    python test_equilibrium.py --grid 8 --n_generations 25 --pop_size 1500
    python test_equilibrium.py --params am11 f11 f22 move_p

Output
------
    output/equilibrium_sweep.csv        (one row per swept parameter value)
    output/equilibrium_variance.csv     (last-3-generation variances + verdict)
    output/equilibrium_pgs_corr.csv     (PGS correlations, 1st & 2nd degree)
    output/equilibrium_pheno_corr.csv   (phenotypic correlations, 1st & 2nd degree)
    plus printed tables and a summary of recommended stable ranges.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import condition4_common as c4


DEFAULT_SWEEP = ["am11", "f11", "vg2", "re", "rg", "move_p"]
DEFAULT_FIXED = {"f22": 0.0}   # f22 held at 0 for the equilibrium check


def classify_trajectory(traj, cap_ratio, drift_tol, big_var, collapse_frac):
    """Classify a variance trajectory over its final three generations.

    Returns a metrics dict that includes, for both traits, the variance at each
    of the last three recorded generations, a plateau/drift status, and a plain
    'equilibrium' vs 'still changing' verdict.
    """
    v1 = traj["var_Y1"].to_numpy(dtype=float)
    v2 = traj["var_Y2"].to_numpy(dtype=float)

    def _metrics(v):
        # Drop non-finite generations (e.g. the founder GEN 0 has no variance yet)
        vv = v[np.isfinite(v)]
        if vv.size < 5:
            return dict(last3=(np.nan, np.nan, np.nan), late=np.nan,
                        ratio=np.nan, drift=np.nan, finite=False)
        last3 = tuple(float(x) for x in vv[-3:])
        early = float(np.mean(vv[:3]))
        late = float(np.mean(vv[-3:]))
        mid = float(np.mean(vv[-6:-3])) if vv.size >= 6 else float(np.mean(vv[:-3]))
        ratio = late / max(early, 1e-9)
        drift = (late - mid) / max(abs(mid), 1e-9)
        return dict(last3=last3, late=late, ratio=ratio, drift=drift, finite=True)

    m1, m2 = _metrics(v1), _metrics(v2)

    def _explodes(m):
        return (not m["finite"]) or (np.isfinite(m["ratio"]) and m["ratio"] > cap_ratio) \
            or (np.isfinite(m["late"]) and m["late"] > big_var)

    def _collapses(m):
        return np.isfinite(m["ratio"]) and m["ratio"] < collapse_frac

    def _plateau(m):
        return m["finite"] and np.isfinite(m["drift"]) and abs(m["drift"]) < drift_tol \
            and not _explodes(m) and not _collapses(m)

    exploding = _explodes(m1) or _explodes(m2)
    collapsing = _collapses(m1) or _collapses(m2)
    equilibrium = _plateau(m1) and _plateau(m2)

    status = "equilibrium" if equilibrium else ("exploding" if exploding
             else ("collapsing" if collapsing else "drifting"))
    verdict = "equilibrium" if equilibrium else "still changing"

    return dict(
        var1_g1=m1["last3"][0], var1_g2=m1["last3"][1], var1_g3=m1["last3"][2],
        var2_g1=m2["last3"][0], var2_g2=m2["last3"][1], var2_g3=m2["last3"][2],
        var1_late=m1["late"], var2_late=m2["late"],
        var1_drift=m1["drift"], var2_drift=m2["drift"],
        status=status, equilibrium=bool(equilibrium), verdict=verdict,
    )


CORR_LABELS = list(c4.DEGREE_RELS)


def _nanmean(vals):
    arr = np.asarray(vals, dtype=float)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else np.nan


def _average_trajectories(trajs):
    """Average per-generation variance trajectories across iterations."""
    df = pd.concat(trajs, ignore_index=True)
    return df.groupby("gen", as_index=False)[["var_Y1", "var_Y2", "h2_1", "h2_2"]].mean()


def _empty_metrics():
    metrics = dict(status="error", equilibrium=False, verdict="error",
                   error="all iterations failed", n_iter=0)
    for k in ("var1_g1", "var1_g2", "var1_g3", "var2_g1", "var2_g2", "var2_g3",
              "var1_late", "var2_late", "var1_drift", "var2_drift"):
        metrics[k] = np.nan
    for lab in CORR_LABELS:
        metrics[f"pgs_{lab}"] = np.nan
        metrics[f"phe_{lab}"] = np.nan
    return metrics


def run_one(params, args, seed):
    """Run several iterations of a condition and report their average.

    Each of ``args.n_iter`` iterations is an independent simulation (different
    seed). We average the per-generation variance trajectory and the relative
    correlations, so the reported statistics are not single-run cases. Uses
    save_history=True so PGS/phenotypic correlations can be computed.
    """
    n_iter = max(1, int(getattr(args, "n_iter", 1)))
    rel_map = dict(c4.DEGREE_RELS)
    min_pairs = getattr(args, "min_pairs", 30)

    trajs, pgs_list, phe_list = [], [], []
    for k in range(n_iter):
        try:
            results = c4.run_island_simulation(
                params, pop_size=args.pop_size, n_generations=args.n_generations,
                n_islands=args.n_islands, n_cv=args.n_cv, seed=seed + k * 7919,
                save_history=True)
            trajs.append(c4.variance_trajectory(results))
            pgs_list.append(c4.relative_correlations(results, args.n_generations, rel_map,
                                                     variable="PGS1", min_pairs=min_pairs))
            phe_list.append(c4.relative_correlations(results, args.n_generations, rel_map,
                                                     variable="Y1", min_pairs=min_pairs))
        except Exception:  # keep going if a single iteration fails
            continue

    if not trajs:
        return _empty_metrics()

    metrics = classify_trajectory(_average_trajectories(trajs),
                                  args.cap, args.drift_tol, args.big, args.collapse)
    for lab in CORR_LABELS:
        metrics[f"pgs_{lab}"] = _nanmean([d.get(lab, np.nan) for d in pgs_list])
        metrics[f"phe_{lab}"] = _nanmean([d.get(lab, np.nan) for d in phe_list])
    metrics["error"] = ""
    metrics["n_iter"] = len(trajs)
    return metrics


def sweep(args):
    fixed = getattr(args, "fixed", {}) or {}
    rows = []
    for param in args.params:
        lo, hi = c4.bounds_for(param)
        grid = np.linspace(lo, hi, args.grid)
        print(f"\n--- sweeping {param} over [{lo:.3f}, {hi:.3f}] ({args.grid} points) ---")
        for val in grid:
            params = c4.make_full_params({**fixed, param: float(val)})
            m = run_one(params, args, seed=args.seed)
            m.update(swept_param=param, value=float(val))
            rows.append(m)
            print(f"  {param}={val:0.3f}  ->  {m['verdict']:14s} "
                  f"(Y1 last3={m['var1_g1']:.2f}/{m['var1_g2']:.2f}/{m['var1_g3']:.2f} | "
                  f"Y2 last3={m['var2_g1']:.2f}/{m['var2_g2']:.2f}/{m['var2_g3']:.2f})")
    return pd.DataFrame(rows)


def variance_table(df):
    """Variance of both traits over the last three generations + verdict."""
    cols = ["swept_param", "value", "var1_g1", "var1_g2", "var1_g3",
            "var2_g1", "var2_g2", "var2_g3", "status", "verdict"]
    t = df[[c for c in cols if c in df.columns]].copy()
    return t.rename(columns={
        "swept_param": "param",
        "var1_g1": "Y1[-3]", "var1_g2": "Y1[-2]", "var1_g3": "Y1[-1]",
        "var2_g1": "Y2[-3]", "var2_g2": "Y2[-2]", "var2_g3": "Y2[-1]"})


def _corr_table(df, prefix):
    cols = ["swept_param", "value"] + [f"{prefix}_{lab}" for lab in CORR_LABELS]
    t = df[[c for c in cols if c in df.columns]].copy()
    return t.rename(columns={"swept_param": "param",
                             **{f"{prefix}_{lab}": lab for lab in CORR_LABELS}})


def pgs_corr_table(df):
    """PGS1 correlations for 1st- and 2nd-degree relatives (S, PO, GP, AV)."""
    return _corr_table(df, "pgs")


def pheno_corr_table(df):
    """Phenotypic (Y1) correlations for 1st- and 2nd-degree relatives."""
    return _corr_table(df, "phe")


def joint_random_check(args):
    """Draw random parameter sets from the full sampled ranges and report the
    fraction that reach equilibrium (how often the current ranges explode)."""
    if args.n_random <= 0:
        return None
    fixed = getattr(args, "fixed", {}) or {}
    rng = np.random.default_rng(args.seed + 1)
    rows = []
    print(f"\n--- joint random check ({args.n_random} draws from full sampled ranges) ---")
    for i in range(args.n_random):
        overrides = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in c4.PARAM_BOUNDS.items()}
        overrides.update(fixed)
        params = c4.make_full_params(overrides)
        m = run_one(params, args, seed=args.seed + 100 + i)
        m.update(overrides)
        rows.append(m)
        print(f"  draw {i+1:2d}: {m['verdict']:14s} "
              f"am11={overrides['am11']:.2f} f11={overrides['f11']:.2f} "
              f"f22={overrides['f22']:.2f} vg2={overrides['vg2']:.2f} move_p={overrides['move_p']:.2f}")
    return pd.DataFrame(rows)


def summarize(df, args):
    print("\n" + "=" * 72)
    print("EQUILIBRIUM SUMMARY  (recommended stable ranges)")
    print("=" * 72)
    for param in args.params:
        sub = df[df["swept_param"] == param]
        eq = sub[sub["equilibrium"]]
        full_lo, full_hi = c4.bounds_for(param)
        tag = " (free target)" if param == "move_p" else ""
        if len(eq) == 0:
            print(f"  {param:16s}{tag}: NO grid point reached equilibrium over "
                  f"[{full_lo:.3f}, {full_hi:.3f}] -- narrow it or fix another driver.")
            continue
        s_lo, s_hi = eq["value"].min(), eq["value"].max()
        frac = len(eq) / len(sub)
        covers_full = np.isclose(s_lo, full_lo) and np.isclose(s_hi, full_hi)
        note = "stable across full range" if covers_full and frac == 1.0 else \
               f"stable in [{s_lo:.3f}, {s_hi:.3f}] ({len(eq)}/{len(sub)} points)"
        print(f"  {param:16s}{tag}: {note}")

    # Overall recommendation
    print("\nRecommendations:")
    problem = []
    for param in args.params:
        if param == "move_p":
            continue
        sub = df[df["swept_param"] == param]
        eq = sub[sub["equilibrium"]]
        if len(eq) < len(sub):
            problem.append(param)
    if problem:
        print("  - Parameters whose high end drives non-equilibrium (consider fixing or"
              " narrowing): " + ", ".join(problem))
    else:
        print("  - All swept parameters reached equilibrium across their ranges at these"
              " settings.")
    print("  - move_p is kept free/estimable; see its stable range above.")


def _parse_fixed(items):
    fixed = dict(DEFAULT_FIXED)
    for kv in (items or []):
        key, _, val = kv.partition("=")
        fixed[key.strip()] = float(val)
    return fixed


def main():
    p = argparse.ArgumentParser(description="Condition-04 equilibrium probe.")
    p.add_argument("--pop_size", type=int, default=3000)
    p.add_argument("--n_generations", type=int, default=30)
    p.add_argument("--n_islands", type=int, default=c4.DEFAULTS["n_islands"])
    p.add_argument("--n_cv", type=int, default=200)
    p.add_argument("--min_pairs", type=int, default=30, help="min relative pairs to trust a correlation")
    p.add_argument("--n_iter", type=int, default=10, help="independent iterations averaged per condition")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--grid", type=int, default=6, help="points per parameter sweep")
    p.add_argument("--cap", type=float, default=3.0, help="late/early variance ratio flagged as exploding")
    p.add_argument("--drift_tol", type=float, default=0.05, help="max |relative drift| over final gens for equilibrium")
    p.add_argument("--big", type=float, default=15.0, help="absolute variance flagged as exploding")
    p.add_argument("--collapse", type=float, default=0.2, help="late/early ratio below which variance is collapsing")
    p.add_argument("--n_random", type=int, default=8, help="joint random draws from full ranges (0 to skip)")
    p.add_argument("--params", nargs="+", default=DEFAULT_SWEEP, help="parameters to sweep")
    p.add_argument("--fix", nargs="*", default=None,
                   help="parameters to hold fixed as name=value (default: f22=0)")
    p.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "output" / "equilibrium_sweep.csv"))
    args = p.parse_args()
    args.fixed = _parse_fixed(args.fix)

    print("Condition-04 equilibrium probe")
    print(f"  pop_size={c4.valid_pop_size(args.pop_size, args.n_islands)}  "
          f"n_generations={args.n_generations}  n_islands={args.n_islands}  n_cv={args.n_cv}")
    print(f"  fixed: {args.fixed}")

    df = sweep(args)
    joint = joint_random_check(args)

    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    variance_table(df).to_csv(outdir / "equilibrium_variance.csv", index=False)
    pgs_corr_table(df).to_csv(outdir / "equilibrium_pgs_corr.csv", index=False)
    pheno_corr_table(df).to_csv(outdir / "equilibrium_pheno_corr.csv", index=False)
    print(f"\nSaved sweep results to {args.out} (+ variance / pgs_corr / pheno_corr tables)")

    print("\n=== Variance over last three generations ===")
    print(variance_table(df).round(3).to_string(index=False))
    print("\n=== PGS correlations by relationship type ===")
    print(pgs_corr_table(df).round(3).to_string(index=False))
    print("\n=== Phenotypic (Y1) correlations by relationship type ===")
    print(pheno_corr_table(df).round(3).to_string(index=False))

    if joint is not None:
        joint_out = str(Path(args.out).with_name("equilibrium_joint_random.csv"))
        joint.to_csv(joint_out, index=False)
        n_eq = int(joint["equilibrium"].sum())
        print(f"\nSaved joint random check to {joint_out}")
        print(f"Joint random check: {n_eq}/{len(joint)} draws reached equilibrium "
              f"({100.0 * n_eq / len(joint):.0f}%).")

    summarize(df, args)


if __name__ == "__main__":
    main()
