"""
STEP 05 — NPE-based ACE Posterior Recovery Simulation Study

Fits the trained NPE to exactly the same test conditions that STEP 04 fitted
with OpenMx, so STEP 06 can put the two estimators side by side.

For each condition in data/ace_test_conditions.csv and each sample size in
(50, 100, 200, 500, 1000, 2000, 20000):
  1. Simulate N_MZ = N_DZ = N twin pairs from the true ACE covariance matrices.
  2. Compute sample covariance statistics (mz_var, mz_cov, dz_var, dz_cov).
  3. Feed those (+ an N_pairs-derived feature, if the loaded model expects one)
     into the trained NPE posterior.
  4. Draw posterior samples and compute the posterior mean, posterior SD (SE
     analogue), and MAP estimate.

Whether the model expects an N_pairs feature — and how it's encoded
(raw N, log(N), or se_proxy = 1/sqrt(N)) — is auto-detected from the loaded
model's config.json, so the same script evaluates models trained with or
without that feature.

Output:
  results/simulations/npe_simulation_results.csv — one row per
  condition x sample size, with columns mirroring ace_simulation_results.csv.

Usage:
    # The "with N" model
    python 05_simulate_posterior_recovery.py --model_dir se_proxy \
                                             --output npe_simulation_results.csv

    # The "no N" model (STEP 06 compares both)
    python 05_simulate_posterior_recovery.py --model_dir no_n_pairs_gaussian_prior \
                                             --output npe_simulation_results_no_n.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_model import (
    ACEEmbeddingNet,   # noqa: F401 — must be importable to unpickle some runs
    DATA_DIR,
    DEFAULT_MODEL_DIR,
    MODELS_DIR,
    SIM_DIR,
    build_features,
    describe_n_encoding,
    load_posterior,
    posterior_stats,
    resolve,
    simulate_covariances,
    summarize_cov,
)

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================================
# Main simulation loop
# ============================================================================

def run_simulation(conditions_csv: str,
                   model_dir: str,
                   sample_sizes: list[int],
                   n_posterior: int,
                   seed: int,
                   output_csv: str) -> pd.DataFrame:

    conditions_csv = resolve(conditions_csv, DATA_DIR)
    model_dir      = resolve(model_dir, MODELS_DIR)
    output_csv     = resolve(output_csv, SIM_DIR)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("STEP 05 — NPE ACE POSTERIOR RECOVERY SIMULATION")
    print("=" * 70)
    print(f"  Conditions file : {conditions_csv}")
    print(f"  Model dir       : {model_dir}")
    print(f"  Sample sizes    : {sample_sizes}")
    print(f"  Posterior draws : {n_posterior}")
    print(f"  Seed            : {seed}")
    print(f"  Output file     : {output_csv}\n")

    # ---- Load model --------------------------------------------------------
    loaded       = load_posterior(model_dir)
    posterior    = loaded['posterior']
    scaler       = loaded['scaler']
    feature_cols = loaded['feature_cols']
    param_names  = loaded['param_names']
    print(f"Loaded posterior  — features   : {feature_cols}")
    print(f"                  — params     : {param_names}")
    print(f"                  — N encoding : {describe_n_encoding(feature_cols)}\n")

    # ---- Load conditions ---------------------------------------------------
    cond_df = pd.read_csv(conditions_csv)
    required = {"condition_id", "A", "C", "E"}
    missing  = required - set(cond_df.columns)
    if missing:
        raise ValueError(f"Conditions file missing columns: {missing}")

    n_conditions = len(cond_df)
    total_fits   = n_conditions * len(sample_sizes)
    print(f"Running {n_conditions} conditions × {len(sample_sizes)} "
          f"sample sizes = {total_fits} fits\n")

    np.random.seed(seed)

    rows = []
    counter = 0

    for _, cond in cond_df.iterrows():
        cid   = int(cond["condition_id"])
        A_t   = float(cond["A"])
        C_t   = float(cond["C"])
        E_t   = float(cond["E"])

        for N in sample_sizes:
            counter += 1

            # ---- Simulate twin pairs and get sample covariances ------------
            # summarize_cov applies the sufficient reduction (mean of the two
            # diagonal entries), matching how training data is generated.
            S_mz, S_dz = simulate_covariances(A_t, C_t, E_t, N_pairs=N)
            mz_var, mz_cov = summarize_cov(S_mz)
            dz_var, dz_cov = summarize_cov(S_dz)

            # ---- Build feature vector (matching training feature order) -----
            x_raw    = build_features(mz_var, mz_cov, dz_var, dz_cov,
                                      N, feature_cols)
            x_scaled = scaler.transform(x_raw).flatten()

            # ---- Draw posterior samples ------------------------------------
            p_mean, p_std, p_map = posterior_stats(posterior, x_scaled, n_posterior)

            A_est, C_est, E_est = p_mean[0], p_mean[1], p_mean[2]
            A_se,  C_se,  E_se  = p_std[0],  p_std[1],  p_std[2]
            A_map, C_map, E_map = p_map[0],  p_map[1],  p_map[2]

            rows.append({
                "condition_id" : cid,
                "sample_size"  : N,
                "true_A"       : A_t,
                "true_C"       : C_t,
                "true_E"       : E_t,
                # Simulated (noisy) covariance statistics for this draw
                "mz_var"       : mz_var,
                "mz_cov"       : mz_cov,
                "dz_var"       : dz_var,
                "dz_cov"       : dz_cov,
                # NPE posterior means  (= point estimates, analogous to *_est)
                "A_est"        : A_est,
                "C_est"        : C_est,
                "E_est"        : E_est,
                # NPE posterior SDs  (= uncertainty, analogous to *_se)
                "A_se"         : A_se,
                "C_se"         : C_se,
                "E_se"         : E_se,
                # NPE MAP estimates
                "A_map"        : A_map,
                "C_map"        : C_map,
                "E_map"        : E_map,
            })

            if counter % max(1, total_fits // 20) == 0 or counter == total_fits:
                print(f"  [{counter:4d} / {total_fits}]  "
                      f"condition {cid:3d}  N = {N:5d}  "
                      f"A_est={A_est:.3f}  C_est={C_est:.3f}  "
                      f"E_est={E_est:.3f}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 70)
    n_conv   = len(results_df)
    mean_A   = results_df["A_est"].mean()
    mean_C   = results_df["C_est"].mean()
    mean_E   = results_df["E_est"].mean()
    print(f"Done.  {n_conv} rows saved to: {output_csv}")
    print(f"  Grand-mean estimates — A: {mean_A:.4f}  "
          f"C: {mean_C:.4f}  E: {mean_E:.4f}")
    print(f"  Grand-mean true      — A: {results_df['true_A'].mean():.4f}  "
          f"C: {results_df['true_C'].mean():.4f}  "
          f"E: {results_df['true_E'].mean():.4f}")
    print("=" * 70 + "\n")

    return results_df


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NPE ACE posterior recovery simulation across sample sizes"
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="ace_test_conditions.csv",
        help="Conditions CSV, relative to data/ (default: ace_test_conditions.csv). "
             "Produced by 04_fit_openmx_reference.R.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Trained run directory, relative to results/models/ "
             f"(default: {DEFAULT_MODEL_DIR.name})",
    )
    parser.add_argument(
        "--sample_sizes",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500, 1000, 2000, 20000],
        help="List of twin-pair sample sizes to evaluate (default: 50 100 200 500 1000 2000 20000)",
    )
    parser.add_argument(
        "--n_posterior",
        type=int,
        default=1000,
        help="Number of posterior draws per observation (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="NumPy random seed for data simulation (default: 2025)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="npe_simulation_results.csv",
        help="Output CSV, relative to results/simulations/ "
             "(default: npe_simulation_results.csv)",
    )
    args = parser.parse_args()

    run_simulation(
        conditions_csv=args.conditions,
        model_dir=args.model_dir,
        sample_sizes=args.sample_sizes,
        n_posterior=args.n_posterior,
        seed=args.seed,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
