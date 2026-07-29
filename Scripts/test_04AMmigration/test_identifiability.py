"""
test_identifiability.py
=======================
Identifiability probe for the condition-04 (AM + island migration) model.

Purpose
-------
Predicting the model parameters from PGS correlations is often under-identified:
some parameters barely move the correlations, and some pairs move them in the
same direction (confounded). This script quantifies that so you can decide which
parameters to FIX to make the remaining ones identifiable.

Two complementary analyses (both PGS-correlation based):

1. Local sensitivity Jacobian (cheap, default).
   Finite-difference d(PGS correlations)/d(parameter) at the baseline, using
   common random numbers so the differences reflect the parameter, not noise.
   From the Jacobian S we report:
     - per-parameter sensitivity (column norm)      -> tiny => weakly identified
     - pairwise collinearity of effects (cosine)     -> ~1  => confounded pair
     - SVD condition number + null-space loadings    -> unidentified combinations

2. Cross-validated recovery R^2 (optional, --n_samples > 0).
   Sample parameter sets, simulate, and try to predict each parameter back from
   the PGS correlations with ridge regression. Low R^2 => hard to identify.

It is independent of test_equilibrium.py.

Usage
-----
    python test_identifiability.py
    python test_identifiability.py --n_samples 0            # Jacobian only (fast)
    python test_identifiability.py --n_samples 80 --replicates 2

Output
------
    output/identifiability_jacobian.csv
    output/identifiability_collinearity.csv
    output/identifiability_recovery.csv   (if --n_samples > 0)
    plus a printed report with recommendations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import condition4_common as c4


# f22 is fixed at 0 (matching the equilibrium setup); it is not estimated here.
FIXED = {"f22": 0.0}
TARGETS = ["vg2", "f11", "re", "am11", "rg", "move_p"]


def feature_vector(params, args, seed, n_iter=1):
    """PGS1 correlations for one condition, averaged over ``n_iter`` iterations.

    Each iteration is an independent simulation (different seed); the resulting
    correlation vectors are averaged so the features are not single-run cases.
    """
    n_iter = max(1, int(n_iter))
    series = []
    for k in range(n_iter):
        try:
            results = c4.run_island_simulation(
                params, pop_size=args.pop_size, n_generations=args.n_generations,
                n_islands=args.n_islands, n_cv=args.n_cv, seed=seed + k * 7919,
                save_history=True)
            corrs = c4.pgs1_correlations(results, args.n_generations,
                                         rel_types=c4.DEFAULT_REL_TYPES, min_pairs=args.min_pairs)
        except Exception as exc:
            print(f"    ! sim failed ({str(exc)[:80]}) -> NaN features")
            corrs = {rel: np.nan for rel in c4.DEFAULT_REL_TYPES}
        series.append(pd.Series(corrs, dtype=float))
    return pd.concat(series, axis=1).mean(axis=1)


def build_jacobian(args):
    """Central-difference sensitivity matrix S (features x targets).

    Averaged over ``args.n_iter`` iterations: each iteration is an independent
    common-random-number finite-difference estimate, and the columns are the
    per-iteration average.
    """
    baseline = c4.make_full_params(FIXED)
    columns = {t: [] for t in TARGETS}

    reps = int(getattr(args, "n_iter", getattr(args, "replicates", 1)))
    for rep in range(reps):
        seed = args.seed + rep * 13
        print(f"\n[Jacobian] iteration {rep + 1}/{reps} (seed={seed})")
        for t in TARGETS:
            lo, hi = c4.bounds_for(t)
            rng = hi - lo
            half = 0.5 * args.perturb * rng
            v_plus = float(np.clip(baseline[t] + half, lo, hi))
            v_minus = float(np.clip(baseline[t] - half, lo, hi))
            step = v_plus - v_minus
            if step <= 0:
                columns[t].append(pd.Series(dtype=float))
                continue
            f_plus = feature_vector(c4.make_full_params({**FIXED, t: v_plus}), args, seed)
            f_minus = feature_vector(c4.make_full_params({**FIXED, t: v_minus}), args, seed)
            # change in correlation across the full parameter range
            col = (f_plus - f_minus) / step * rng
            columns[t].append(col)
            print(f"    d/d {t:7s}: ||effect|| = {np.linalg.norm(col.dropna().to_numpy()):.4f}")

    # average replicates, then assemble aligned matrix
    avg_cols = {t: pd.concat(columns[t], axis=1).mean(axis=1) for t in TARGETS}
    S = pd.DataFrame(avg_cols).dropna(axis=0, how="any")
    return S


def analyze_jacobian(S, args):
    print("\n" + "=" * 72)
    print("JACOBIAN ANALYSIS (PGS-correlation sensitivity)")
    print("=" * 72)
    if S.shape[0] < 2:
        print("  Too few usable correlation features; increase --pop_size or --n_generations.")
        return None, None

    print(f"  Features used: {S.shape[0]}  ({', '.join(S.index)})")

    # per-parameter sensitivity
    col_norm = np.linalg.norm(S.to_numpy(), axis=0)
    rel_sens = col_norm / col_norm.max() if col_norm.max() > 0 else col_norm
    sens = pd.DataFrame({"sensitivity": col_norm, "relative": rel_sens}, index=S.columns)
    sens = sens.sort_values("relative", ascending=False)
    print("\n  Per-parameter sensitivity (higher = better identified from PGS corr):")
    for name, row in sens.iterrows():
        flag = "  <-- weak" if row["relative"] < args.weak_frac else ""
        print(f"    {name:8s}: {row['relative']:.3f}{flag}")

    # collinearity (cosine similarity between effect columns)
    M = S.to_numpy()
    norms = np.linalg.norm(M, axis=0)
    Mn = M / np.where(norms > 0, norms, 1.0)
    C = Mn.T @ Mn
    collin = pd.DataFrame(C, index=S.columns, columns=S.columns)
    print("\n  Confounded parameter pairs (|cosine of effect directions| > "
          f"{args.confound_thresh}):")
    pairs = []
    cols = list(S.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(C[i, j]) > args.confound_thresh:
                pairs.append((cols[i], cols[j], C[i, j]))
                print(f"    {cols[i]:8s} ~ {cols[j]:8s}  cos={C[i, j]:+.3f}")
    if not pairs:
        print("    (none)")

    # SVD / condition number
    sv = np.linalg.svd(M, compute_uv=False)
    cond = sv.max() / sv.min() if sv.min() > 1e-12 else np.inf
    Vt = np.linalg.svd(M, full_matrices=False)[2]
    null_dir = pd.Series(Vt[-1], index=S.columns)
    print(f"\n  Condition number of S: {cond:.1f}  "
          f"({'well-conditioned' if cond < 30 else 'ill-conditioned / under-identified'})")
    print("  Least-identified parameter combination (smallest singular vector):")
    for name, w in null_dir.abs().sort_values(ascending=False).items():
        print(f"    {name:8s}: {null_dir[name]:+.3f}")

    return sens, collin


def recovery(args):
    """Cross-validated ridge recovery R^2 of each target from PGS correlations."""
    if args.n_samples <= 0:
        return None
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score, KFold
        from sklearn.preprocessing import StandardScaler
    except Exception:
        print("\n[Recovery] scikit-learn not available; skipping recovery R^2.")
        return None

    print("\n" + "=" * 72)
    print(f"RECOVERY R^2  ({args.n_samples} sampled conditions)")
    print("=" * 72)
    rng = np.random.default_rng(args.seed + 999)

    feats, targ = [], []
    for i in range(args.n_samples):
        overrides = {t: float(rng.uniform(*c4.bounds_for(t))) for t in TARGETS}
        params = c4.make_full_params({**FIXED, **overrides})
        fv = feature_vector(params, args, seed=args.seed + 1000 + i,
                            n_iter=getattr(args, "n_iter", 1))
        feats.append(fv)
        targ.append(overrides)
        if (i + 1) % 10 == 0:
            print(f"    simulated {i + 1}/{args.n_samples}")

    X = pd.DataFrame(feats)
    Y = pd.DataFrame(targ)
    # drop features missing in > 30% of rows, impute the rest with the mean
    keep = X.columns[X.isna().mean() < 0.30]
    X = X[keep]
    X = X.fillna(X.mean())
    if X.shape[1] < 2 or len(X) < 5:
        print("  Not enough usable data for recovery; increase --n_samples/--pop_size.")
        return None

    n_splits = min(5, len(X))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    Xs = StandardScaler().fit_transform(X.to_numpy())

    rows = []
    print(f"\n  Using {X.shape[1]} features, {len(X)} samples, {n_splits}-fold CV:")
    for t in TARGETS:
        y = Y[t].to_numpy()
        scores = cross_val_score(Ridge(alpha=args.ridge_alpha), Xs, y, cv=kf, scoring="r2")
        r2 = float(np.mean(scores))
        rows.append({"parameter": t, "recovery_r2": r2})
        flag = "  <-- poorly identified" if r2 < args.r2_floor else ""
        print(f"    {t:8s}: R^2 = {r2:+.3f}{flag}")
    return pd.DataFrame(rows).set_index("parameter")


def recommend(sens, collin, rec, args):
    print("\n" + "=" * 72)
    print("RECOMMENDATIONS")
    print("=" * 72)

    weak = []
    if sens is not None:
        weak = [n for n, r in sens["relative"].items() if r < args.weak_frac]

    confounded = []
    if collin is not None:
        cols = list(collin.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if abs(collin.iloc[i, j]) > args.confound_thresh:
                    confounded.append((cols[i], cols[j]))

    poor = []
    if rec is not None:
        poor = [n for n, r in rec["recovery_r2"].items() if r < args.r2_floor]

    fix_candidates = sorted(set(weak) | set(poor) - {"move_p"})
    if "move_p" in fix_candidates:
        fix_candidates.remove("move_p")

    if fix_candidates:
        print("  Consider FIXING (weak sensitivity and/or low recovery R^2):")
        print("    " + ", ".join(fix_candidates))
    else:
        print("  No individually weak parameters detected at these settings.")

    if confounded:
        print("  Confounded pairs -- fix ONE member of each to break the tie:")
        for a, b in confounded:
            print(f"    {a} <-> {b}")

    # move_p focus
    print("\n  move_p (target to keep estimable):")
    mp_sens = sens.loc["move_p", "relative"] if sens is not None and "move_p" in sens.index else np.nan
    mp_r2 = rec.loc["move_p", "recovery_r2"] if rec is not None and "move_p" in rec.index else np.nan
    mp_conf = [p for p in confounded if "move_p" in p]
    print(f"    relative sensitivity = {mp_sens:.3f}"
          + (f", recovery R^2 = {mp_r2:+.3f}" if mp_r2 == mp_r2 else ""))
    if mp_conf:
        others = sorted({(a if b == 'move_p' else b) for a, b in mp_conf})
        print(f"    move_p is confounded with: {', '.join(others)} -> fixing those should"
              " make move_p identifiable.")
    else:
        print("    move_p is not strongly confounded with another parameter here.")


def main():
    p = argparse.ArgumentParser(description="Condition-04 identifiability probe.")
    p.add_argument("--pop_size", type=int, default=c4.DEFAULTS["pop_size"])
    p.add_argument("--n_generations", type=int, default=15)
    p.add_argument("--n_islands", type=int, default=c4.DEFAULTS["n_islands"])
    p.add_argument("--n_cv", type=int, default=300)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--min_pairs", type=int, default=30, help="min relative pairs to trust a correlation")
    p.add_argument("--perturb", type=float, default=0.30, help="finite-difference step as fraction of each range")
    p.add_argument("--n_iter", type=int, default=10, help="iterations averaged (Jacobian replicates and recovery features)")
    p.add_argument("--weak_frac", type=float, default=0.15, help="relative sensitivity below this is 'weak'")
    p.add_argument("--confound_thresh", type=float, default=0.90, help="|cosine| above this flags a confounded pair")
    p.add_argument("--n_samples", type=int, default=40, help="samples for recovery R^2 (0 to skip)")
    p.add_argument("--ridge_alpha", type=float, default=1.0)
    p.add_argument("--r2_floor", type=float, default=0.30, help="recovery R^2 below this is 'poorly identified'")
    args = p.parse_args()

    print("Condition-04 identifiability probe")
    print(f"  pop_size={c4.valid_pop_size(args.pop_size, args.n_islands)}  "
          f"n_generations={args.n_generations}  n_islands={args.n_islands}  n_cv={args.n_cv}")
    print(f"  targets: {', '.join(TARGETS)}")

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    S = build_jacobian(args)
    S.to_csv(out_dir / "identifiability_jacobian.csv")
    sens, collin = analyze_jacobian(S, args)
    if collin is not None:
        collin.to_csv(out_dir / "identifiability_collinearity.csv")

    rec = recovery(args)
    if rec is not None:
        rec.to_csv(out_dir / "identifiability_recovery.csv")

    recommend(sens, collin, rec, args)
    print(f"\nSaved outputs under {out_dir}")


if __name__ == "__main__":
    main()
