"""
condition4_common.py
====================
Shared helpers for the condition-04 (EA-driven island migration) test scripts.

Model (overhauled condition 04)
-------------------------------
    EA (trait 1)  = additive genetics (PGS part + latent part) + unique environment
    No assortative mating (random mating within islands), no vertical transmission,
    no shared environment.
    Individuals migrate between ``n_islands`` islands sorted directly on their EA
    phenotype (Y1): each generation a fraction ``move_p`` of each sex on each island
    is pooled, sorted by Y1 and redistributed so that high-EA movers end up on the
    high-numbered islands.
    The simulation starts from a random founder population and runs ``n_gen``
    generations (3..10) in total; ``n_gen`` is itself a parameter to estimate.

The engine is bivariate, so trait 2 is carried along as an inert dummy
(vg2 = 0.5, rg = 0, re = 0, fully latent); it drives nothing.

Estimable parameters: move_p, n_gen, vg1, prop_h2_latent1.

It reuses the production simulation engine (IslandMigrationSimulation) and the
relationship/measure extraction utilities under ``Scripts/SimulationFunctions``.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- make the shared SimulationFunctions package importable ------------------
_SIMFUNC_DIR = Path(__file__).resolve().parent.parent / "SimulationFunctions"
if str(_SIMFUNC_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMFUNC_DIR))

from island_migration import IslandMigrationSimulation          # noqa: E402
from relationship_finder import find_relationship_pairs         # noqa: E402
from postprocessing import extract_individual_measures          # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# ============================================================================
# CONDITION-04 PARAMETER SPACE
# ============================================================================

# Estimable / sampled parameters
PARAM_BOUNDS = {
    "move_p":          (0.01, 0.70),  # fraction of each sex per island that migrates per generation
    "n_gen":           (3, 10),       # total generations since the random founders (integer)
    "vg1":             (0.20, 0.80),  # EA total genetic variance
    "prop_h2_latent1": (0.30, 0.90),  # share of EA genetic variance NOT captured by the PGS
}
PARAMS = list(PARAM_BOUNDS)
INT_PARAMS = {"n_gen"}

# Values used when a parameter is held fixed
BASELINE_PARAMS = {
    "move_p": 0.40,
    "n_gen": 6,
    "vg1": 0.50,
    "prop_h2_latent1": 0.60,
}

# Structural constants (never estimated)
FIXED_STRUCTURE = {
    "vg2": 0.5,              # dummy trait 2
    "prop_h2_latent2": 1.0,
    "rg": 0.0,
    "re": 0.0,
}

# Relationship types used as features (the 17 types of the observed data in
# nn_NPE/observed_correlations_*.csv), for both PGS1 and Y1.
REL_TYPES = [
    "S", "HSFS", "PSC", "PPSCC", "M", "MS", "SMS", "MSM", "MSC", "SMSC", "SMSM",
    "SMSMS", "MSMSM", "MSMSC", "PSMSC", "SMSMSC", "MSMSMS",
]
VARIABLES = ["PGS1", "Y1"]
FEATURES = [f"{rel}_{var}" for var in VARIABLES for rel in REL_TYPES]

# Laptop-sized simulation defaults
DEFAULTS = dict(
    pop_size=4000,     # must be divisible by n_islands * 2
    n_islands=5,
    n_cv=300,
    maf_min=0.01,
    maf_max=0.50,
    min_pairs=30,
)


# ============================================================================
# UTILITIES
# ============================================================================

@contextlib.contextmanager
def suppress_output():
    """Silence the very chatty per-generation prints from the engine."""
    with open(os.devnull, "w") as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def make_full_params(overrides=None):
    """Build a complete parameter dict from the baseline plus overrides."""
    params = dict(BASELINE_PARAMS)
    if overrides:
        params.update(overrides)
    params["n_gen"] = int(round(params["n_gen"]))
    return params


def sample_params(rng, free=None, fixed=None):
    """Draw the free parameters uniformly from PARAM_BOUNDS (n_gen as an integer).

    Parameters not in ``free`` take their value from ``fixed`` or the baseline.
    """
    free = PARAMS if free is None else list(free)
    out = make_full_params(fixed)
    for p in free:
        lo, hi = PARAM_BOUNDS[p]
        if p in INT_PARAMS:
            out[p] = int(rng.integers(lo, hi + 1))
        else:
            out[p] = float(rng.uniform(lo, hi))
    return out


def valid_pop_size(pop_size, n_islands):
    """Round pop_size down to a multiple of n_islands*2 (engine requirement)."""
    block = n_islands * 2
    return max(block, (int(pop_size) // block) * block)


# ============================================================================
# MATRIX SETUP
# ============================================================================

def setup_matrices(params, n_generations):
    """Model matrices for the EA-migration condition (AE on EA; no AM, VT or S)."""
    vg1 = params["vg1"]
    vg2 = FIXED_STRUCTURE["vg2"]
    rg = FIXED_STRUCTURE["rg"]
    re = FIXED_STRUCTURE["re"]
    prop_h2_latent1 = params["prop_h2_latent1"]
    prop_h2_latent2 = FIXED_STRUCTURE["prop_h2_latent2"]

    k2_matrix = np.array([[1.0, rg], [rg, 1.0]])

    # Observable (PGS) genetic path matrix
    d11 = np.sqrt(max(vg1 * (1.0 - prop_h2_latent1), 0.0))
    d22 = np.sqrt(max(vg2 * (1.0 - prop_h2_latent2), 0.0))
    delta_mat = np.array([[d11, 0.0], [0.0, d22]])

    # Latent genetic path matrix
    a11 = np.sqrt(max(vg1 * prop_h2_latent1, 0.0))
    a22 = np.sqrt(max(vg2 * prop_h2_latent2, 0.0))
    a_mat = np.array([[a11, 0.0], [0.0, a22]])

    covg_mat = (delta_mat @ k2_matrix @ delta_mat.T) + (a_mat @ k2_matrix @ a_mat.T)

    # Unique environment (total phenotypic variance = 1 at generation 0)
    ve1 = 1.0 - vg1
    ve2 = 1.0 - vg2
    cove = re * np.sqrt(ve1 * ve2)
    cove_mat = np.array([[ve1, cove], [cove, ve2]])

    return {
        "cove_mat": cove_mat,
        "f_mat": np.zeros((2, 2)),          # no vertical transmission
        "s_mat": np.zeros((2, 2)),          # no shared environment
        "a_mat": a_mat,
        "d_mat": delta_mat,
        "am_list": [0.0] * n_generations,   # no assortative mating
        "covy_mat": covg_mat + cove_mat,
        "k2_matrix": k2_matrix,
    }


# ============================================================================
# RUN A SIMULATION
# ============================================================================

def run_island_simulation(params, *, pop_size=DEFAULTS["pop_size"],
                          n_islands=DEFAULTS["n_islands"], n_cv=DEFAULTS["n_cv"],
                          seed=1, save_history=True,
                          maf_min=DEFAULTS["maf_min"], maf_max=DEFAULTS["maf_max"]):
    """Construct and run one EA-migration simulation for ``params['n_gen']`` generations."""
    n_gen = int(params["n_gen"])
    pop_size = valid_pop_size(pop_size, n_islands)
    matrices = setup_matrices(params, n_gen)

    with suppress_output():
        sim = IslandMigrationSimulation(
            n_islands=n_islands,
            move_p=params["move_p"],
            within_island_am=0.0,  # random mating within islands
            migration_trait=1,     # islands sorted directly on EA (Y1)
            mating_trait=1,
            n_jobs=1,
            n_CV=n_cv,
            rg_effects=FIXED_STRUCTURE["rg"],
            maf_min=maf_min,
            maf_max=maf_max,
            num_generations=n_gen,
            pop_size=pop_size,
            mating_type="phenotypic",
            avoid_inbreeding=True,
            save_each_gen=save_history,
            save_covs=False,
            seed=seed,
            output_summary_filename=None,
            **matrices,
        )
        results = sim.run_simulation()
    return results


# ============================================================================
# READ-OUTS
# ============================================================================

def strat_trajectory(results):
    """Per-generation Var(Y1) and the between-island share of Y1 and PGS1 variance.

    Requires a run with ``save_history=True``. Generation 0 is the founders; its
    snapshot is saved before island IDs are attached, so its shares are NaN.
    Later generations carry the island they were born on (mother's island).
    """
    rows = []
    for g, phen in enumerate(results["HISTORY"]["PHEN"]):
        if "island_id" not in phen.columns:
            rows.append({"gen": g, "var_Y1": pd.to_numeric(phen["Y1"]).var(),
                         "between_share_Y1": np.nan, "between_share_PGS1": np.nan})
            continue
        df = phen[["Y1", "TPO1", "TMO1", "island_id"]].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["island_id"])
        df["PGS1"] = df["TPO1"] + df["TMO1"]
        row = {"gen": g, "var_Y1": df["Y1"].var()}
        for v in ("Y1", "PGS1"):
            total = df[v].var()
            between = df.groupby("island_id")[v].mean().var() if df["island_id"].nunique() > 1 else 0.0
            row[f"between_share_{v}"] = between / total if total > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _trim_to_final_generations(results, n_generations, n_final=3):
    """Slice HISTORY to the last ``n_final`` analysed generations.

    Follows the production convention (DataGeneratingNN_Combined_04AMmigration.py):
    PHEN/XO/XL at FINAL_GENS = [G-3, G-2, G-1] and MATES at FINAL_GENS + [G], because
    MATES[i+1] holds the mating pairs formed in generation i.
    """
    hist = results["HISTORY"]
    final_gens = list(range(n_generations - n_final, n_generations))
    n_mates = len(hist["MATES"])
    mates_idx = final_gens + [final_gens[-1] + 1] if final_gens[-1] + 1 < n_mates else final_gens
    return {
        "HISTORY": {
            "PHEN": [hist["PHEN"][i] for i in final_gens],
            "XO": [hist["XO"][i] for i in final_gens],
            "XL": [hist["XL"][i] for i in final_gens],
            "MATES": [hist["MATES"][i] if i < n_mates else None for i in mates_idx],
        }
    }


def feature_vector(results, n_generations, rel_types=None, min_pairs=DEFAULTS["min_pairs"]):
    """PGS1 and Y1 correlations between relatives, for each relationship type.

    Returns a Series indexed ``{REL}_PGS1`` and ``{REL}_Y1``; relationship types
    with fewer than ``min_pairs`` usable pairs are NaN.
    """
    rel_types = REL_TYPES if rel_types is None else rel_types
    trimmed = _trim_to_final_generations(results, n_generations)

    measures = extract_individual_measures(trimmed, ["Y1", "TPO1", "TMO1"])
    measures["PGS1"] = measures["TPO1"] + measures["TMO1"]
    lookup = measures.set_index("ID")[VARIABLES]

    out = {}
    for rel in rel_types:
        try:
            with suppress_output():
                pairs = find_relationship_pairs(trimmed, rel, generations=[0, 1, 2],
                                                output_format="long")
        except Exception:
            pairs = None
        for var in VARIABLES:
            out[f"{rel}_{var}"] = np.nan
        if pairs is None or len(pairs) == 0:
            continue
        for var in VARIABLES:
            col = lookup[var]
            x = pairs["Person_ID"].map(col).to_numpy(dtype=float)
            y = pairs["Relative_ID"].map(col).to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < min_pairs or x[ok].std() < 1e-9 or y[ok].std() < 1e-9:
                continue
            out[f"{rel}_{var}"] = float(np.corrcoef(x[ok], y[ok])[0, 1])
    return pd.Series(out, dtype=float).reindex(
        [f"{r}_{v}" for v in VARIABLES for r in rel_types])


def simulate_features(params, seed, sim_kwargs=None):
    """Run one simulation and return its feature vector (NaNs on failure)."""
    sim_kwargs = dict(sim_kwargs or {})
    min_pairs = sim_kwargs.pop("min_pairs", DEFAULTS["min_pairs"])
    try:
        results = run_island_simulation(params, seed=seed, save_history=True, **sim_kwargs)
        return feature_vector(results, int(params["n_gen"]), min_pairs=min_pairs)
    except Exception as exc:
        print(f"    ! sim failed ({str(exc)[:80]}) -> NaN features", flush=True)
        return pd.Series(np.nan, index=FEATURES, dtype=float)


# ============================================================================
# PARALLEL SIMULATION DESIGN
# ============================================================================

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS")


def single_thread_env():
    """Cap BLAS threads at 1 in this process's environment.

    Must be called in the parent *before* worker processes are spawned: the
    children read these variables when they first import numpy.
    """
    for var in _THREAD_VARS:
        os.environ[var] = "1"


def _design_worker(job):
    """Worker for the process pool: job = (row_id, params, seed, sim_kwargs)."""
    row_id, params, seed, sim_kwargs = job
    feats = simulate_features(params, seed, sim_kwargs)
    return row_id, params, feats


def run_jobs(jobs, n_workers=3, progress_every=25):
    """Run ``_design_worker`` jobs on at most ``n_workers`` processes."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    out = []
    if n_workers <= 1:
        for k, job in enumerate(jobs):
            out.append(_design_worker(job))
            if (k + 1) % progress_every == 0:
                print(f"    simulated {k + 1}/{len(jobs)}", flush=True)
        return out
    single_thread_env()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futs = [ex.submit(_design_worker, job) for job in jobs]
        for k, fut in enumerate(as_completed(futs)):
            out.append(fut.result())
            if (k + 1) % progress_every == 0:
                print(f"    simulated {k + 1}/{len(jobs)}", flush=True)
    return out


def simulate_design(n, out_csv, *, free=None, fixed=None, seed=2026, n_workers=3,
                    sim_kwargs=None, chunk=60):
    """Simulate ``n`` conditions with ``free`` parameters drawn uniformly.

    Rows are appended to ``out_csv`` chunk by chunk, so an interrupted run
    resumes where it stopped. Returns the full design DataFrame
    (``param_*`` columns + feature columns).
    """
    out_csv = Path(out_csv)
    rng = np.random.default_rng(seed)
    all_params = [sample_params(rng, free, fixed) for _ in range(n)]

    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    done_ids = set(done["row_id"]) if "row_id" in done else set()
    todo = [i for i in range(n) if i not in done_ids]
    print(f"[design] {len(done_ids)} rows cached, {len(todo)} to simulate -> {out_csv}", flush=True)

    for start in range(0, len(todo), chunk):
        ids = todo[start:start + chunk]
        jobs = [(i, all_params[i], seed * 1000 + i, sim_kwargs) for i in ids]
        rows = []
        for row_id, params, feats in run_jobs(jobs, n_workers=n_workers, progress_every=10**9):
            row = {"row_id": row_id, **{f"param_{p}": params[p] for p in PARAMS}}
            row.update(feats.to_dict())
            rows.append(row)
        chunk_df = pd.DataFrame(rows).sort_values("row_id")
        chunk_df.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        print(f"[design] {min(start + chunk, len(todo))}/{len(todo)} new rows written", flush=True)

    return pd.read_csv(out_csv).sort_values("row_id").reset_index(drop=True)
