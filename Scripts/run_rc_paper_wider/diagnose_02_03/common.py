"""Shared helpers for the diagnostics in this folder: observed targets and training data."""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NPE = HERE.parents[1] / "nn_NPE"
RESULTS = HERE / "results"
TOL = 0.01   # correlation-scale tolerance added to the observed SE in distances


def load_observed(features):
    """Observed correlation and SE for each `<rel>_PGS1` / `<rel>_Y1` feature (trait 1 = EA)."""
    pgs = pd.read_csv(NPE / "observed_correlations_PGS_ea4new.csv").set_index("RelType")
    ph = pd.read_csv(NPE / "observed_correlations_pheno_021126.csv").set_index("RelType")
    obs, se = [], []
    for f in features:
        rel, var = f.rsplit("_", 1)
        row = (pgs if var.startswith("PGS") else ph).loc[rel]
        obs.append(row["Correlation"])
        se.append(row["SE"])
    return np.array(obs, float), np.array(se, float)


def load_training(cond):
    """Training rows, feature names, parameter names, observed values and SEs for one fit."""
    df = pd.read_csv(NPE / f"nn_training_{cond}.csv")
    spec = NPE / "run_specs" / cond
    features = [l.strip() for l in open(spec / "features.txt") if l.strip()]
    params = [l.strip() for l in open(spec / "params.txt") if l.strip()]
    obs, se = load_observed(features)
    return df, features, params, obs, se


def misfit(X, obs, se, tol=TOL):
    """Root-mean-square of (simulated - observed) / sqrt(SE^2 + tol^2) over the features."""
    z = (np.asarray(X, float) - obs) / np.sqrt(se ** 2 + tol ** 2)
    return np.sqrt(np.mean(z ** 2, axis=-1))
