"""
test_identifiability.py
=======================
Which subsets of {move_p, n_gen, vg1, prop_h2_latent1} can be identified *jointly*
from PGS1 + Y1 relative correlations in the EA-migration model (condition 04)?

Every non-empty subset S (15 in total) is treated as "free" while the remaining
parameters are fixed/known, and checked with three methods:

1. Local sensitivity Jacobian (analytic, notebook method).
   Finite-difference d(correlations)/d(parameter) at the baseline using common
   random numbers, averaged over ``n_iter`` replicates. Rows are scaled by the
   single-run noise SD of each feature (from baseline replicates). For S we use
   the columns of J that belong to S and report relative sensitivity, the largest
   |cosine| between effect directions, the condition number and the Cramér-Rao
   SE  sqrt(diag((J_S' J_S)^-1))  in units of each parameter's full range.

2. Cross-validated ridge recovery R^2 (notebook method).
   One shared simulation design with all four parameters free. For subset S
   the features are the correlations PLUS the true values of the parameters not
   in S (they are "known"), which is equivalent to fixing them, averaged over
   their range. Each parameter in S is predicted with 5-fold CV ridge.

3. Neural posterior estimation (simulation method).
   Same design and same "known" convention. Mixed estimator (npe_mixed.py): a
   categorical classifier for n_gen and an NSF flow for the continuous
   parameters given n_gen. Reports test R^2 of the posterior mean, posterior
   SD / prior SD, 90% interval coverage, and for n_gen the mode accuracy and
   log score.

Usage
-----
    python test_identifiability.py --stage pilot
    python test_identifiability.py --stage trajectory
    python test_identifiability.py --stage jacobian
    python test_identifiability.py --stage design --n_design 2000
    python test_identifiability.py --stage ridge
    python test_identifiability.py --stage npe
    python test_identifiability.py --stage fixcheck
    python test_identifiability.py --stage compare
    python test_identifiability.py --stage all

All outputs go to ``output/``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import condition4_common as c4

TARGETS = c4.PARAMS
OUT = c4.OUTPUT_DIR

# Pass thresholds
WEAK_FRAC = 0.15        # relative sensitivity below this = weak
CONFOUND_THRESH = 0.90  # |cosine| above this = confounded pair
COND_MAX = 30.0         # condition number above this = ill-conditioned
CRB_MAX = 0.15          # Cramér-Rao SE / range above this = not identified
R2_FLOOR = 0.50         # ridge / NPE R^2 below this = not identified
SD_RATIO_MAX = 0.60     # NPE posterior SD / prior SD above this = not identified
MAX_MISSING = 0.05      # drop features missing in more than this share of design rows
NPE_EMBEDDING = "identity"  # features straight to the flow; "mlp" = PgsPhenoEmbeddingNet (dropout 0.3). See benchmark_npe_architecture.py


def all_subsets(params=TARGETS):
    """All non-empty subsets, largest first."""
    out = []
    for k in range(len(params), 0, -1):
        out.extend(list(c) for c in itertools.combinations(params, k))
    return out


def subset_label(subset):
    return "+".join(subset)


def sim_kwargs_from(args):
    return dict(pop_size=args.pop_size, n_islands=args.n_islands, n_cv=args.n_cv,
                min_pairs=args.min_pairs)


# ============================================================================
# 0. PILOT AND TRAJECTORIES
# ============================================================================

def pilot(args):
    """Time a few simulations and list features that come out NaN."""
    rows = []
    for G, mp in [(3, 0.4), (6, 0.4), (10, 0.4), (10, 0.05), (10, 0.7), (3, 0.01)]:
        params = c4.make_full_params({"n_gen": G, "move_p": mp})
        t0 = time.time()
        feats = c4.simulate_features(params, seed=args.seed, sim_kwargs=sim_kwargs_from(args))
        rows.append({"n_gen": G, "move_p": mp, "seconds": round(time.time() - t0, 1),
                     "n_nan": int(feats.isna().sum()),
                     "nan_features": ", ".join(feats[feats.isna()].index)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT / "pilot.csv", index=False)
    return df


def _trajectory_worker(job):
    key, params, seed, sim_kwargs = job
    sim_kwargs = {k: v for k, v in sim_kwargs.items() if k != "min_pairs"}
    res = c4.run_island_simulation(params, seed=seed, save_history=True, **sim_kwargs)
    tr = c4.strat_trajectory(res)
    for k, v in key.items():
        tr[k] = v
    return tr


def trajectories(args, move_ps=(0.05, 0.2, 0.4, 0.7), n_rep=3):
    """Stratification over 10 generations for several move_p (others at baseline)."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    jobs = []
    for m in move_ps:
        for r in range(n_rep):
            params = c4.make_full_params({"move_p": m, "n_gen": 10})
            jobs.append(({"move_p": m, "rep": r}, params, args.seed + 31 * r, sim_kwargs_from(args)))
    c4.single_thread_env()
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=mp.get_context("spawn")) as ex:
        trs = list(ex.map(_trajectory_worker, jobs))
    df = pd.concat(trs, ignore_index=True)
    avg = df.groupby(["move_p", "gen"], as_index=False)[
        ["var_Y1", "between_share_Y1", "between_share_PGS1"]].mean()
    avg.to_csv(OUT / "trajectory.csv", index=False)
    print(avg.pivot(index="gen", columns="move_p", values="between_share_Y1").round(3))
    return avg


# ============================================================================
# 1. JACOBIAN + NOISE-SCALED INFORMATION
# ============================================================================

def _step_values(t):
    """Plus/minus values for the central difference of parameter t."""
    lo, hi = c4.PARAM_BOUNDS[t]
    base = c4.BASELINE_PARAMS[t]
    if t in c4.INT_PARAMS:
        return base + 1, base - 1
    half = 0.5 * 0.30 * (hi - lo)
    return float(np.clip(base + half, lo, hi)), float(np.clip(base - half, lo, hi))


def build_jacobian(args):
    """Central-difference sensitivity S (features x targets), full-range scaled.

    Each replicate uses one seed for both the + and - runs (common random numbers);
    the columns are averaged over ``args.n_iter`` replicates. Also simulates
    ``args.n_noise`` baseline replicates to estimate the single-run SD of each feature.
    """
    jobs = []
    for rep in range(args.n_iter):
        seed = args.seed + 13 * rep
        for t in TARGETS:
            v_plus, v_minus = _step_values(t)
            jobs.append((("J", t, "+", rep), c4.make_full_params({t: v_plus}), seed, sim_kwargs_from(args)))
            jobs.append((("J", t, "-", rep), c4.make_full_params({t: v_minus}), seed, sim_kwargs_from(args)))
    for r in range(args.n_noise):
        jobs.append((("N", r), c4.make_full_params(), args.seed + 5000 + r, sim_kwargs_from(args)))

    print(f"[jacobian] {len(jobs)} simulations on {args.n_workers} workers", flush=True)
    results = {key: feats for key, _, feats in c4.run_jobs(jobs, n_workers=args.n_workers)}

    cols = {}
    for t in TARGETS:
        v_plus, v_minus = _step_values(t)
        lo, hi = c4.PARAM_BOUNDS[t]
        reps = [(results[("J", t, "+", r)] - results[("J", t, "-", r)]) / (v_plus - v_minus) * (hi - lo)
                for r in range(args.n_iter)]
        cols[t] = pd.concat(reps, axis=1).mean(axis=1)
    S = pd.DataFrame(cols)

    noise = pd.concat([results[("N", r)] for r in range(args.n_noise)], axis=1)
    sd = noise.std(axis=1, ddof=1).rename("noise_sd")
    S = S.dropna(axis=0, how="any")
    sd = sd.reindex(S.index)
    keep = sd > 1e-6
    S, sd = S[keep], sd[keep]
    S.to_csv(OUT / "jacobian.csv")
    sd.to_csv(OUT / "jacobian_noise_sd.csv")
    return S, sd


def jacobian_subsets(S, sd):
    """Per-subset, per-parameter Jacobian diagnostics on the noise-scaled Jacobian."""
    Jw = S.div(sd, axis=0)                      # each feature in units of its single-run SD
    norms = np.linalg.norm(Jw.to_numpy(), axis=0)
    rel_sens = pd.Series(norms / norms.max(), index=Jw.columns)

    rows = []
    for subset in all_subsets():
        M = Jw[subset].to_numpy()
        n = np.linalg.norm(M, axis=0)
        Mn = M / np.where(n > 0, n, 1.0)
        C = Mn.T @ Mn
        sv = np.linalg.svd(M, compute_uv=False)
        cond = sv.max() / sv.min() if sv.min() > 1e-12 else np.inf
        try:
            crb = np.sqrt(np.diag(np.linalg.inv(M.T @ M)))
        except np.linalg.LinAlgError:
            crb = np.full(len(subset), np.inf)
        for j, p in enumerate(subset):
            others = [abs(C[j, k]) for k in range(len(subset)) if k != j]
            max_cos = max(others) if others else 0.0
            ok = (rel_sens[p] >= WEAK_FRAC and max_cos <= CONFOUND_THRESH
                  and cond <= COND_MAX and crb[j] <= CRB_MAX)
            rows.append({"subset": subset_label(subset), "size": len(subset), "parameter": p,
                         "jac_rel_sens": rel_sens[p], "jac_max_cos": max_cos,
                         "jac_cond": cond, "jac_crb_se": crb[j], "jac_pass": ok})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "subsets_jacobian.csv", index=False)
    return df


# ============================================================================
# 2. SIMULATION DESIGN
# ============================================================================

def design_path(name="design"):
    return OUT / f"{name}.csv"


def load_design(path=None):
    """Load a design CSV, drop sparse features, mean-impute the rest.

    Returns (design_df, feature_columns).
    """
    df = pd.read_csv(path or design_path())
    feats = [c for c in c4.FEATURES if c in df.columns]
    miss = df[feats].isna().mean()
    feats = [c for c in feats if miss[c] <= MAX_MISSING]
    df = df.dropna(subset=feats, how="all").reset_index(drop=True)
    df[feats] = df[feats].fillna(df[feats].mean())
    return df, feats


def _xy(design, feats, subset):
    """Inputs = correlations + known (non-free) parameters; targets = subset."""
    known = [f"param_{p}" for p in TARGETS if p not in subset]
    X = design[feats + known].to_numpy(dtype=float)
    Y = design[[f"param_{p}" for p in subset]].to_numpy(dtype=float)
    return X, Y, known


# ============================================================================
# 3. RIDGE RECOVERY
# ============================================================================

def ridge_subsets(design, feats, alpha=1.0, seed=0):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for subset in all_subsets():
        X, Y, _ = _xy(design, feats, subset)
        for j, p in enumerate(subset):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            r2 = float(np.mean(cross_val_score(model, X, Y[:, j], cv=kf, scoring="r2")))
            rows.append({"subset": subset_label(subset), "size": len(subset), "parameter": p,
                         "ridge_r2": r2, "ridge_pass": r2 >= R2_FLOOR})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "subsets_ridge.csv", index=False)
    return df


# ============================================================================
# 4. NPE
# ============================================================================

def _import_npe():
    nn_dir = Path(__file__).resolve().parent.parent / "nn_NPE"
    if str(nn_dir) not in sys.path:
        sys.path.insert(0, str(nn_dir))
    import train_npe_pgs_and_pheno as tnpe   # noqa: E402
    return tnpe


def npe_one(design, feats, subset, out_dir, *, epochs=300, test_frac=0.15,
            n_post=500, seed=0, n_threads=3, make_plots=True, embedding=None):
    """Train the mixed NPE for one subset and evaluate on a held-out test set.

    n_gen (when free) gets a categorical posterior p(n_gen | x); the continuous
    parameters get a flow p(theta_c | x, n_gen) (see npe_mixed.py). Point
    estimate = posterior mean; for n_gen also the mode accuracy and the log
    probability given to the true count are reported.
    """
    import torch
    from sklearn.preprocessing import StandardScaler
    from npe_mixed import MixedNPE

    tnpe = _import_npe()
    torch.set_num_threads(n_threads)
    rng = np.random.default_rng(seed)
    embedding = NPE_EMBEDDING if embedding is None else embedding

    X, Y, known = _xy(design, feats, subset)
    idx = rng.permutation(len(X))
    n_test = int(round(test_frac * len(X)))
    te, tr = idx[:n_test], idx[n_test:]
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])

    count = next((p for p in subset if p in c4.INT_PARAMS), None)
    model = MixedNPE(subset, c4.PARAM_BOUNDS, count_param=count, embedding=embedding,
                     seed=seed, max_epochs=epochs).fit(Xtr, Y[tr])
    samp = np.stack([model.sample(x, n_post, rng) for x in Xte])     # (n_test, n_post, d)

    res, rows = {}, []
    for j, p in enumerate(subset):
        true, s = Y[te, j], samp[:, :, j]
        pm, psd = s.mean(1), s.std(1)
        r2 = float(1 - np.sum((true - pm) ** 2) / np.sum((true - true.mean()) ** 2))
        lo, hi = c4.PARAM_BOUNDS[p]
        if p == count:
            k = hi - lo + 1
            prior_sd = np.sqrt((k ** 2 - 1) / 12.0)                  # discrete uniform
            pmf = model.count_pmf(Xte)
            t_idx = np.rint(true).astype(int) - lo
            extra = {"npe_mode_acc": float(np.mean(pmf.argmax(1) == t_idx)),
                     "npe_logscore": float(np.mean(np.log(np.clip(pmf[np.arange(len(t_idx)), t_idx], 1e-4, None))))}
        else:
            prior_sd = (hi - lo) / np.sqrt(12.0)
            extra = {}
        q05, q95 = np.quantile(s, [0.05, 0.95], axis=1)
        sd_ratio = float(np.mean(psd) / prior_sd)
        res[p] = dict(r2=r2, true=true, pred_mean=pm, pred_std=psd, mean_sigma=float(psd.mean()))
        rows.append({"subset": subset_label(subset), "size": len(subset), "parameter": p,
                     "npe_r2": r2, "npe_sd_ratio": sd_ratio,
                     "npe_cover90": float(np.mean((true >= q05) & (true <= q95))),
                     "npe_pass": (r2 >= R2_FLOOR) and (sd_ratio <= SD_RATIO_MAX), **extra})
    if make_plots:
        out_dir.mkdir(parents=True, exist_ok=True)
        tnpe.plot_npe_results(res, out_dir, list(subset))
        if count:
            lo, hi = c4.PARAM_BOUNDS[count]
            np.savez(out_dir / "count_pmf.npz", pmf=model.count_pmf(Xte),
                     true=Y[te, subset.index(count)], values=np.arange(lo, hi + 1))
    return pd.DataFrame(rows)


def npe_subsets(design, feats, args):
    frames = []
    for subset in all_subsets():
        t0 = time.time()
        df = npe_one(design, feats, subset, OUT / "npe" / subset_label(subset),
                     epochs=args.epochs, seed=args.seed, n_threads=args.n_workers)
        frames.append(df)
        print(f"[npe] {subset_label(subset):35s} "
              + "  ".join(f"{r.parameter}: R2={r.npe_r2:+.2f} sd={r.npe_sd_ratio:.2f}"
                          for r in df.itertuples())
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        pd.concat(frames).to_csv(OUT / "subsets_npe.csv", index=False)
    return pd.concat(frames, ignore_index=True)


# ============================================================================
# 5. FIXED-VS-KNOWN CHECK
# ============================================================================

def fixcheck(args, subset=None):
    """Truly fix the complement of ``subset`` at baseline and compare NPE with the
    known-input result on an equally sized subsample of the main design."""
    if subset is None:
        comp = pd.read_csv(OUT / "subsets_compare.csv")
        three = comp[(comp["size"] == 3)].sort_values("npe_min_r2", ascending=False)
        subset = three.iloc[0]["subset"].split("+")
    fixed = {p: c4.BASELINE_PARAMS[p] for p in TARGETS if p not in subset}
    name = "design_fixed_" + "_".join(f"{k}{v}" for k, v in fixed.items())
    print(f"[fixcheck] free={subset} fixed={fixed}", flush=True)
    c4.simulate_design(args.n_fixcheck, design_path(name), free=subset, fixed=fixed,
                       seed=args.seed + 77, n_workers=args.n_workers,
                       sim_kwargs=sim_kwargs_from(args))
    dfix, feats_fix = load_design(design_path(name))
    dmain, feats_main = load_design()
    feats = [f for f in feats_main if f in feats_fix]
    dsub = dmain.sample(n=len(dfix), random_state=args.seed).reset_index(drop=True)

    label = subset_label(subset)
    a = npe_one(dfix, feats, subset, OUT / "npe_fixcheck" / f"{label}_fixed",
                epochs=args.epochs, seed=args.seed, n_threads=args.n_workers)
    b = npe_one(dsub, feats, subset, OUT / "npe_fixcheck" / f"{label}_known",
                epochs=args.epochs, seed=args.seed, n_threads=args.n_workers)
    a["mode"], b["mode"] = "truly fixed", "known input (same n)"
    out = pd.concat([a, b], ignore_index=True)
    out.to_csv(OUT / "fixcheck.csv", index=False)
    print(out.to_string(index=False))
    return out


# ============================================================================
# 6. COMPARISON
# ============================================================================

def compare():
    # Recompute the Jacobian diagnostics on the same features the design keeps,
    # so all three methods see identical inputs.
    S = pd.read_csv(OUT / "jacobian.csv", index_col=0)
    sd = pd.read_csv(OUT / "jacobian_noise_sd.csv", index_col=0)["noise_sd"]
    _, feats = load_design()
    keep = [f for f in S.index if f in feats]
    jac = jacobian_subsets(S.loc[keep], sd.loc[keep])
    rid = pd.read_csv(OUT / "subsets_ridge.csv")
    npe = pd.read_csv(OUT / "subsets_npe.csv")
    keys = ["subset", "size", "parameter"]
    long = jac.merge(rid, on=keys).merge(npe, on=keys)
    long.to_csv(OUT / "subsets_long.csv", index=False)

    agg = long.groupby(["subset", "size"]).agg(
        jac_all=("jac_pass", "all"), ridge_all=("ridge_pass", "all"), npe_all=("npe_pass", "all"),
        jac_max_crb=("jac_crb_se", "max"), ridge_min_r2=("ridge_r2", "min"),
        npe_min_r2=("npe_r2", "min"), npe_max_sd_ratio=("npe_sd_ratio", "max"),
    ).reset_index().sort_values(["size", "npe_min_r2"], ascending=[False, False])

    def maximal(col):
        ok = agg[agg[col]]["subset"].tolist()
        sets = [set(s.split("+")) for s in ok]
        return [s for s, st in zip(ok, sets) if not any(st < o for o in sets)]

    agg["all_methods"] = agg[["jac_all", "ridge_all", "npe_all"]].all(axis=1)
    agg.to_csv(OUT / "subsets_compare.csv", index=False)
    summary = {col: maximal(col) for col in ["jac_all", "ridge_all", "npe_all", "all_methods"]}
    with open(OUT / "maximal_subsets.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(agg.to_string(index=False))
    print("\nLargest jointly identifiable subsets:")
    for k, v in summary.items():
        print(f"  {k:12s}: {v}")
    return agg, summary


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Condition-04 (EA migration) subset identifiability.")
    p.add_argument("--stage", default="all",
                   choices=["pilot", "trajectory", "jacobian", "design", "ridge", "npe",
                            "fixcheck", "compare", "all"])
    p.add_argument("--pop_size", type=int, default=c4.DEFAULTS["pop_size"])
    p.add_argument("--n_islands", type=int, default=c4.DEFAULTS["n_islands"])
    p.add_argument("--n_cv", type=int, default=c4.DEFAULTS["n_cv"])
    p.add_argument("--min_pairs", type=int, default=c4.DEFAULTS["min_pairs"])
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--n_workers", type=int, default=3, help="max CPU processes / torch threads")
    p.add_argument("--n_iter", type=int, default=5, help="Jacobian replicates")
    p.add_argument("--n_noise", type=int, default=20, help="baseline replicates for feature noise SD")
    p.add_argument("--n_design", type=int, default=2000, help="simulations in the shared design")
    p.add_argument("--n_fixcheck", type=int, default=600)
    p.add_argument("--epochs", type=int, default=300)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    st = args.stage
    if st in ("pilot",):
        pilot(args)
    if st in ("trajectory", "all"):
        trajectories(args)
    if st in ("jacobian", "all"):
        S, sd = build_jacobian(args)
        print(jacobian_subsets(S, sd).to_string(index=False))
    if st in ("design", "all"):
        c4.simulate_design(args.n_design, design_path(), seed=args.seed,
                           n_workers=args.n_workers, sim_kwargs=sim_kwargs_from(args))
    if st in ("ridge", "all"):
        design, feats = load_design()
        print(ridge_subsets(design, feats).to_string(index=False))
    if st in ("npe", "all"):
        design, feats = load_design()
        npe_subsets(design, feats, args)
    if st in ("compare", "all"):
        compare()
    if st in ("fixcheck", "all"):
        fixcheck(args)


if __name__ == "__main__":
    main()
