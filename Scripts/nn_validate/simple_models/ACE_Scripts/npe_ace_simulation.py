"""
NPE-based ACE Simulation Study

For each condition in ace_test_conditions.csv and each sample size in
(50, 100, 200, 500, 1000, 2000):
  1. Simulate N_MZ = N_DZ = N twin pairs from the true ACE covariance matrices.
  2. Compute sample covariance statistics (mz_var, mz_cov, dz_var, dz_cov).
  3. Feed those + N_pairs into the trained NPE posterior.
  4. Draw posterior samples and compute posterior mean (point estimate) and
     posterior SD (SE analogue).

Output:
  npe_simulation_results.csv  — one row per condition × sample size, with
  columns mirroring ace_simulation_results.csv from OpenMx.

Usage:
    python npe_ace_simulation.py
    python npe_ace_simulation.py --conditions ace_test_conditions.csv \
                                  --model_dir results_ace_npe \
                                  --n_posterior 1000 \
                                  --seed 2025 \
                                  --output npe_simulation_results.csv
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Allow script to be run from any directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# ACEEmbeddingNet must be importable so pickle can reconstruct the posterior
from train_ace_nn import ACEEmbeddingNet, ACE_PARAM_NAMES  # noqa: F401
from ACEnn import simulate_covariances


# ============================================================================
# Helpers
# ============================================================================

def load_posterior(model_dir: Path):
    """Load the trained NPE posterior, feature scaler, and config."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)

    feature_cols   = config.get("feature_cols",
                                ["mz_var", "mz_cov", "dz_var", "dz_cov"])
    param_names    = config.get("param_names", ACE_PARAM_NAMES)
    # Detect whether the model was trained with log(N_pairs) or raw N_pairs
    include_n_pairs  = ("N_pairs" in feature_cols) or ("log_N_pairs" in feature_cols)
    use_log_n_pairs  = "log_N_pairs" in feature_cols

    posterior_path = model_dir / "posterior.pkl"
    if not posterior_path.exists():
        raise FileNotFoundError(
            f"posterior.pkl not found in {model_dir}. "
            "Run train_ace_nn.py first."
        )

    with open(posterior_path, "rb") as f:
        posterior = pickle.load(f)

    scaler = joblib.load(model_dir / "feature_scaler.pkl")

    # Put the neural net into eval mode if accessible
    if hasattr(posterior, "_neural_net"):
        posterior._neural_net.eval()

    return posterior, scaler, feature_cols, param_names, include_n_pairs, use_log_n_pairs


def posterior_stats(posterior, x_scaled_1d: np.ndarray,
                    n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw `n_samples` posterior samples for a single scaled observation and
    return (mean, std) arrays of shape (3,).

    Parameters
    ----------
    posterior     : trained sbi posterior object
    x_scaled_1d   : 1-D numpy array of scaled features for one observation
    n_samples     : number of posterior draws

    Returns
    -------
    mean : np.ndarray, shape (3,)
    std  : np.ndarray, shape (3,)
    """
    x_t = torch.FloatTensor(x_scaled_1d).unsqueeze(0)
    with torch.no_grad():
        samples = posterior.sample(
            (n_samples,), x=x_t, show_progress_bars=False
        )
    s = samples.numpy()          # (n_samples, 3)
    return s.mean(axis=0), s.std(axis=0)


# ============================================================================
# Main simulation loop
# ============================================================================

def run_simulation(conditions_csv: str,
                   model_dir: str,
                   sample_sizes: list[int],
                   n_posterior: int,
                   seed: int,
                   output_csv: str) -> pd.DataFrame:

    conditions_csv = Path(conditions_csv)
    model_dir      = Path(model_dir)
    if not model_dir.is_absolute():
        model_dir = SCRIPT_DIR / model_dir
    output_csv = Path(output_csv)
    if not output_csv.is_absolute():
        output_csv = SCRIPT_DIR / output_csv

    print("\n" + "=" * 70)
    print("NPE ACE SIMULATION STUDY")
    print("=" * 70)
    print(f"  Conditions file : {conditions_csv}")
    print(f"  Model dir       : {model_dir}")
    print(f"  Sample sizes    : {sample_sizes}")
    print(f"  Posterior draws : {n_posterior}")
    print(f"  Seed            : {seed}")
    print(f"  Output file     : {output_csv}\n")

    # ---- Load model --------------------------------------------------------
    posterior, scaler, feature_cols, param_names, include_n_pairs, use_log_n_pairs = \
        load_posterior(model_dir)
    print(f"Loaded posterior  — features   : {feature_cols}")
    print(f"                  — params     : {param_names}")
    print(f"                  — log(N)     : {use_log_n_pairs}\n")

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
            obs_mz_cov, obs_dz_cov = simulate_covariances(A_t, C_t, E_t,
                                                           N_pairs=N)
            mz_var = float(obs_mz_cov[0, 0])
            mz_cov = float(obs_mz_cov[0, 1])
            dz_var = float(obs_dz_cov[0, 0])
            dz_cov = float(obs_dz_cov[0, 1])

            # ---- Build feature vector (matching training feature order) -----
            base_feats = [mz_var, mz_cov, dz_var, dz_cov]
            if include_n_pairs:
                # Use log(N) if the model was trained that way (recommended),
                # otherwise fall back to raw N for backwards compatibility.
                base_feats.append(np.log(float(N)) if use_log_n_pairs else float(N))
            x_raw    = np.array(base_feats, dtype=np.float32).reshape(1, -1)
            x_scaled = scaler.transform(x_raw).flatten()

            # ---- Draw posterior samples ------------------------------------
            p_mean, p_std = posterior_stats(posterior, x_scaled, n_posterior)

            A_est, C_est, E_est = p_mean[0], p_mean[1], p_mean[2]
            A_se,  C_se,  E_se  = p_std[0],  p_std[1],  p_std[2]

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
        description="NPE ACE simulation study across sample sizes"
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=str(SCRIPT_DIR / "ace_test_conditions.csv"),
        help="Path to conditions CSV (default: ace_test_conditions.csv)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="results_ace_npe",
        help="Directory containing trained NPE posterior (default: results_ace_npe)",
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
        default=str(SCRIPT_DIR / "npe_simulation_results.csv"),
        help="Output CSV path (default: npe_simulation_results.csv)",
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
