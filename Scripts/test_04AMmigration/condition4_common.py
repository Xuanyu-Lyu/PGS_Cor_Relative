"""
condition4_common.py
====================
Shared helpers for the condition-04 (AM + island migration) test scripts.

This module is imported by the two *independent* test drivers:

    test_equilibrium.py       -- do the variance components reach equilibrium?
    test_identifiability.py   -- which parameters are identifiable from PGS
                                 correlations, and which should be fixed?

It reuses the production simulation engine (IslandMigrationSimulation) and
the relationship/measure extraction utilities under
``Scripts/SimulationFunctions``. Everything here is sized to run quickly on a
laptop (small population, few CVs, few generations).

The model wiring mirrors
``Scripts/run_rc_paper/DataGeneratingNN_Combined_04AMmigration.py``:
    Trait 1 = EA        (within-island assortative-mating trait; am11)
    Trait 2 = Migration (fully latent, prop_h2_latent2 = 1; drives migration)
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


# ============================================================================
# CONDITION-04 PARAMETER SPACE
# ============================================================================

# Estimable / sampled parameters (from DataGeneratingNN_Combined_04AMmigration.py)
PARAM_BOUNDS = {
    "vg2":    (0.10, 0.90),   # migration-trait total genetic variance
    "f11":    (0.05, 0.30),   # within-trait vertical transmission, EA
    "f22":    (0.05, 0.30),   # within-trait vertical transmission, migration
    "re":     (0.00, 0.50),   # environmental correlation between traits
    "am11":   (0.25, 0.75),   # within-island spousal correlation on EA
    "rg":     (0.01, 0.60),   # genetic correlation EA <-> migration (widened range)
    "move_p": (0.01, 0.30),   # fraction migrating per island per generation
}

# Parameters currently held fixed in the production condition.
FIXED_PARAMS = {
    "vg1": 0.45,             # EA total genetic variance
    "prop_h2_latent1": 0.6,  # EA: proportion of h2 that is latent (no PGS)
    "prop_h2_latent2": 1.0,  # migration: entirely latent (no observable PGS)
}

# Exploratory ranges for the otherwise-fixed parameters, used only when the
# equilibrium sweep is asked to vary them.
EXPLORE_BOUNDS = {
    "vg1": (0.20, 0.80),
    "prop_h2_latent1": (0.20, 0.90),
}

# Baseline = midpoint of every sampled range, plus the fixed values, with a
# couple of explicit overrides (f11 = 0.10; vg2 = 0.50).
BASELINE_PARAMS = {**FIXED_PARAMS,
                   **{k: 0.5 * (lo + hi) for k, (lo, hi) in PARAM_BOUNDS.items()},
                   "f11": 0.10, "vg2": 0.50}

# Relationship types used as PGS-correlation features. This is a subset of the
# full production list, chosen so that pairs are still plentiful at small N.
DEFAULT_REL_TYPES = [
    "S", "HSFS", "PSC", "PPSCC", "M", "MS", "SMS", "MSC", "MSM", "SMSC", "SMSM",
]

# Relationships reported in the correlation tables, in the simulation's own
# M/P/C/S relationship-path notation (see find_relationship_pairs). Label == path.
#   S  = full siblings            (1st degree)
#   P  = parent - offspring       (1st degree)
#   PP = grandparent - grandchild (2nd degree)
#   PS = avuncular (parent's sib) (2nd degree)
#   MSMSM, MSMSC                  (5th-degree in-law chains)
#   SMSMSC, MSMSMS               (6th-degree in-law chains)
FIRST_DEGREE_RELS = {"S": "S", "P": "P"}
SECOND_DEGREE_RELS = {"PP": "PP", "PS": "PS"}
FIFTH_DEGREE_RELS = {"MSMSM": "MSMSM", "MSMSC": "MSMSC"}
SIXTH_DEGREE_RELS = {"SMSMSC": "SMSMSC", "MSMSMS": "MSMSMS"}
# Combined, ordered mapping used by the correlation tables.
DEGREE_RELS = {**FIRST_DEGREE_RELS, **SECOND_DEGREE_RELS,
               **FIFTH_DEGREE_RELS, **SIXTH_DEGREE_RELS}

# Small, laptop-friendly simulation defaults.
DEFAULTS = dict(
    pop_size=1000,     # must be divisible by n_islands * 2
    n_islands=5,
    n_cv=400,
    maf_min=0.01,
    maf_max=0.50,
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


def bounds_for(param):
    """Return the (lo, hi) range for a parameter (sampled or exploratory)."""
    if param in PARAM_BOUNDS:
        return PARAM_BOUNDS[param]
    if param in EXPLORE_BOUNDS:
        return EXPLORE_BOUNDS[param]
    raise KeyError("No known range for parameter '%s'." % param)


def make_full_params(overrides=None):
    """Build a complete parameter dict from the baseline plus overrides."""
    params = dict(BASELINE_PARAMS)
    if overrides:
        params.update(overrides)
    return params


def valid_pop_size(pop_size, n_islands):
    """Round pop_size down to a multiple of n_islands*2 (engine requirement)."""
    block = n_islands * 2
    adjusted = (int(pop_size) // block) * block
    return max(block, adjusted)


# ============================================================================
# MATRIX SETUP  (mirrors setup_matrices in the production 04 script)
# ============================================================================

def setup_matrices(params, n_generations):
    """Build the model matrices for the AM+migration condition.

    Returns a dict of keyword arguments for IslandMigrationSimulation,
    including the scalar ``within_island_am`` (which the caller must pop out
    before forwarding the rest to the constructor).
    """
    vg1 = params["vg1"]
    vg2 = params["vg2"]
    rg = params["rg"]
    re = params["re"]
    prop_h2_latent1 = params["prop_h2_latent1"]
    prop_h2_latent2 = params["prop_h2_latent2"]

    k2_matrix = np.array([[1.0, rg], [rg, 1.0]])

    # Observable genetic path matrix (trait 2 has none: prop_h2_latent2 = 1)
    vg_obs1 = vg1 * (1.0 - prop_h2_latent1)
    vg_obs2 = vg2 * (1.0 - prop_h2_latent2)
    d11 = np.sqrt(max(vg_obs1, 0.0))
    d22 = np.sqrt(max(vg_obs2, 0.0))
    delta_mat = np.array([[d11, 0.0], [0.0, d22]])

    # Latent genetic path matrix
    vg_lat1 = vg1 * prop_h2_latent1
    vg_lat2 = vg2 * prop_h2_latent2
    a11 = np.sqrt(max(vg_lat1, 0.0))
    a22 = np.sqrt(max(vg_lat2, 0.0))
    a_mat = np.array([[a11, 0.0], [0.0, a22]])

    covg_mat = (delta_mat @ k2_matrix @ delta_mat.T) + (a_mat @ k2_matrix @ a_mat.T)

    # Environmental covariance (traits scaled so total phenotypic var = 1)
    ve1 = 1.0 - vg1
    ve2 = 1.0 - vg2
    cove = re * np.sqrt(ve1 * ve2)
    cove_mat = np.array([[ve1, cove], [cove, ve2]])

    covy_mat = covg_mat + cove_mat

    # Within-trait vertical transmission only; no shared environment.
    f_mat = np.array([[params["f11"], 0.0], [0.0, params["f22"]]])
    s_mat = np.zeros((2, 2))

    am11 = params["am11"]
    am_list = [am11 for _ in range(n_generations)]

    return {
        "cove_mat": cove_mat,
        "f_mat": f_mat,
        "s_mat": s_mat,
        "a_mat": a_mat,
        "d_mat": delta_mat,
        "am_list": am_list,
        "within_island_am": am11,
        "covy_mat": covy_mat,
        "k2_matrix": k2_matrix,
    }


# ============================================================================
# RUN A SIMULATION
# ============================================================================

def run_island_simulation(params, *, pop_size, n_generations, n_islands,
                          n_cv, seed, save_history,
                          maf_min=DEFAULTS["maf_min"], maf_max=DEFAULTS["maf_max"]):
    """Construct and run one small AM+migration simulation.

    ``save_history=False`` is enough for the equilibrium test (it only reads
    the per-generation variance summary). The identifiability test needs
    ``save_history=True`` so relationship pairs can be enumerated.
    """
    pop_size = valid_pop_size(pop_size, n_islands)
    matrices = setup_matrices(params, n_generations)
    within_island_am = matrices.pop("within_island_am")

    with suppress_output():
        sim = IslandMigrationSimulation(
            n_islands=n_islands,
            move_p=params["move_p"],
            within_island_am=within_island_am,
            migration_trait=2,   # islands sorted by Y2 (migration trait)
            mating_trait=1,      # within-island AM on Y1 (EA)
            n_jobs=1,
            n_CV=n_cv,
            rg_effects=params["rg"],
            maf_min=maf_min,
            maf_max=maf_max,
            num_generations=n_generations,
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

def variance_trajectory(results):
    """Per-generation phenotypic variance and heritability from SUMMARY.RES.

    Returns a DataFrame with columns: gen, var_Y1, var_Y2, h2_1, h2_2.
    """
    rows = []
    for s in results.get("SUMMARY.RES", []):
        vp = np.array(s.get("VP", [[np.nan, np.nan], [np.nan, np.nan]]), dtype=float)
        h2 = s.get("h2", [np.nan, np.nan])
        rows.append({
            "gen": s.get("GEN", np.nan),
            "var_Y1": vp[0, 0],
            "var_Y2": vp[1, 1],
            "h2_1": float(h2[0]) if h2 else np.nan,
            "h2_2": float(h2[1]) if h2 else np.nan,
        })
    return pd.DataFrame(rows)


def _trim_to_final_generations(results, n_generations, n_final=3):
    """Slice the saved HISTORY down to the last ``n_final`` generations."""
    hist = results["HISTORY"]
    final_idx = list(range(n_generations - n_final, n_generations))
    n_mates = len(hist["MATES"])
    return {
        "HISTORY": {
            "PHEN": [hist["PHEN"][i] for i in final_idx],
            "XO": [hist["XO"][i] for i in final_idx],
            "XL": [hist["XL"][i] for i in final_idx],
            "MATES": [hist["MATES"][i] if i < n_mates else None for i in final_idx],
        }
    }


def pgs1_correlations(results, n_generations, rel_types=None, min_pairs=30):
    """Compute PGS1 (EA polygenic score) correlations by relationship type.

    Returns a dict {rel_type: correlation}. Relationship types with fewer than
    ``min_pairs`` usable pairs are returned as NaN.
    """
    if rel_types is None:
        rel_types = DEFAULT_REL_TYPES

    trimmed = _trim_to_final_generations(results, n_generations)

    measures = extract_individual_measures(trimmed, ["Y1", "Y2", "TPO1", "TMO1"])
    measures["PGS1"] = measures["TPO1"] + measures["TMO1"]
    pgs_lookup = measures.set_index("ID")["PGS1"].to_dict()

    out = {}
    for rel in rel_types:
        try:
            with suppress_output():
                pairs = find_relationship_pairs(
                    trimmed, rel, generations=[0, 1, 2], output_format="long"
                )
        except Exception:
            out[rel] = np.nan
            continue

        if pairs is None or len(pairs) == 0:
            out[rel] = np.nan
            continue

        x = pairs["Person_ID"].map(pgs_lookup).to_numpy(dtype=float)
        y = pairs["Relative_ID"].map(pgs_lookup).to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_pairs:
            out[rel] = np.nan
            continue

        xs, ys = x[ok], y[ok]
        if xs.std() < 1e-9 or ys.std() < 1e-9:
            out[rel] = np.nan
            continue
        out[rel] = float(np.corrcoef(xs, ys)[0, 1])
    return out


def relative_correlations(results, n_generations, rel_map, variable="PGS1",
                          min_pairs=30):
    """Correlation of ``variable`` between relatives, for each label -> path.

    ``rel_map`` is a dict of {label: relationship_path} (e.g. FIRST_DEGREE_RELS).
    ``variable`` is "PGS1" (EA polygenic score) or "Y1"/"Y2" (phenotype).
    Returns {label: correlation}; labels with fewer than ``min_pairs`` usable
    pairs are returned as NaN.
    """
    trimmed = _trim_to_final_generations(results, n_generations)

    measures = extract_individual_measures(trimmed, ["Y1", "Y2", "TPO1", "TMO1"])
    measures["PGS1"] = measures["TPO1"] + measures["TMO1"]
    lookup = measures.set_index("ID")[variable].to_dict()

    out = {}
    for label, path in rel_map.items():
        try:
            with suppress_output():
                pairs = find_relationship_pairs(
                    trimmed, path, generations=[0, 1, 2], output_format="long"
                )
        except Exception:
            out[label] = np.nan
            continue

        if pairs is None or len(pairs) == 0:
            out[label] = np.nan
            continue

        x = pairs["Person_ID"].map(lookup).to_numpy(dtype=float)
        y = pairs["Relative_ID"].map(lookup).to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_pairs:
            out[label] = np.nan
            continue

        xs, ys = x[ok], y[ok]
        if xs.std() < 1e-9 or ys.std() < 1e-9:
            out[label] = np.nan
            continue
        out[label] = float(np.corrcoef(xs, ys)[0, 1])
    return out
