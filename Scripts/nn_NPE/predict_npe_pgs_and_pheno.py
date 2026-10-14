"""
Predict Parameters from Observed Correlations using a Trained NPE Posterior
(Neural Posterior Estimation — SNPE-C / sbi)

Loads the trained posterior produced by ``train_npe_pgs_and_pheno.py`` and
draws posterior samples conditioned on your observed PGS and phenotypic (Y1)
correlations.  Unlike the original point-prediction model, this script returns
the **full posterior distribution**, providing:

  - Posterior mean (point estimate)
  - Posterior standard deviation (uncertainty)
  - 95% credible interval  [2.5th – 97.5th percentile samples]

Observed correlation data can be provided via CSV files or directly as
command-line arguments.

Usage
-----
    python predict_npe_pgs_and_pheno.py \
        --model_dir results_npe_unweighted_05AMmigration_noVT2_PGSonly \
        --correlations_pgs observed_correlations_PGS.csv \
        --correlations_pheno observed_correlations_pheno.csv \
        --n_samples 1000

Or directly:
    python predict_npe_pgs_and_pheno.py \\
        --model_dir results_npe_unweighted_04EnvLatentAM \\
        --pgs_S 0.523 --pgs_M 0.041 \\
        --pheno_S 0.312 --pheno_M 0.025
"""

import numpy as np
import pandas as pd
import torch
import argparse
from pathlib import Path
import joblib
import json
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

# Must be imported before pickle.load so the classes are resolvable during unpickling.
# The posterior was pickled with these classes defined in the training script.
from train_npe_pgs_and_pheno import PgsPhenoEmbeddingNet, SNPEFeatureReg, AttentionLayer  # noqa: F401
from train_npe_pgs_and_pheno import set_eval_mode
import npe_count  # noqa: F401  (MixedPosterior, for runs trained with --count_param)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_trained_posterior(model_dir: Path):
    """
    Load the trained NPE posterior and its configuration.

    Parameters
    ----------
    model_dir : Path
        Directory produced by ``train_npe_pgs_and_pheno.py``.

    Returns
    -------
    posterior : sbi NeuralPosterior
    feature_scaler : sklearn StandardScaler
    poly_transformer : sklearn PolynomialFeatures or None
    config : dict
    param_names : list of str
    """
    print(f"\nLoading NPE posterior from: {model_dir}")

    # Config
    with open(model_dir / "config.json", "r") as f:
        config = json.load(f)

    param_names = config["param_names"]

    # Feature scaler
    feature_scaler = joblib.load(model_dir / "feature_scaler.pkl")
    print(f"✓ Loaded feature scaler")

    # Polynomial transformer (if used)
    poly_transformer = None
    if (model_dir / "poly_transformer.pkl").exists():
        poly_transformer = joblib.load(model_dir / "poly_transformer.pkl")
        print(f"✓ Loaded polynomial transformer (degree={config['interaction_degree']})")

    # Posterior
    posterior_path = model_dir / "posterior.pkl"
    if not posterior_path.exists():
        print(f"\n✗ posterior.pkl not found in {model_dir}")
        print("  Did you run train_npe_pgs_and_pheno.py first?")
        sys.exit(1)

    with open(posterior_path, "rb") as f:
        posterior = pickle.load(f)
    print(f"✓ Loaded posterior object")

    print(f"✓ Model predicts {len(param_names)} parameters: {param_names}")

    if config.get("regularization") == "feature_specific":
        print(f"\n✓ Trained with Feature-Specific Regularization:")
        print(f"    pgs_weight_decay:   {config.get('pgs_weight_decay', '?')}")
        print(f"    pheno_weight_decay: {config.get('pheno_weight_decay', '?')}")

    return posterior, feature_scaler, poly_transformer, config, param_names


# ============================================================================
# INPUT PREPARATION   (adapted from original predict script)
# ============================================================================

def prepare_correlation_input(
    pgs_correlations_dict: dict,
    pheno_correlations_dict: dict,
    expected_features: list,
    trait: int = 1,
) -> np.ndarray:
    """
    Assemble the input feature vector in the order expected by the model.

    Parameters
    ----------
    pgs_correlations_dict : dict  {relationship_type → correlation_value}
    pheno_correlations_dict : dict  {relationship_type → correlation_value}
    expected_features : list of str  (from config['original_features'])

    Returns
    -------
    X : np.ndarray of shape (1, n_features)  – NOT yet scaled
    """
    X = np.zeros(len(expected_features))

    used_pgs   = set()
    used_pheno = set()
    matched_pgs = matched_pheno = 0
    missing = []

    pgs_suffix   = f"_PGS{trait}"
    pheno_suffix = f"_Y{trait}"

    for i, feat in enumerate(expected_features):
        if pgs_suffix in feat:
            rel = feat.replace(pgs_suffix, "")
            if rel.startswith("cor_"):
                rel = rel[4:]
            if rel in pgs_correlations_dict:
                X[i] = pgs_correlations_dict[rel]
                matched_pgs += 1
                used_pgs.add(rel)
            else:
                missing.append(f"PGS:{rel}")

        elif pheno_suffix in feat:
            rel = feat.replace(pheno_suffix, "")
            if rel.startswith("cor_"):
                rel = rel[4:]
            if rel in pheno_correlations_dict:
                X[i] = pheno_correlations_dict[rel]
                matched_pheno += 1
                used_pheno.add(rel)
            else:
                missing.append(f"Y{trait}:{rel}")

    unused_pgs   = set(pgs_correlations_dict.keys())   - used_pgs
    unused_pheno = set(pheno_correlations_dict.keys()) - used_pheno

    print(f"\n{'='*70}")
    print("CORRELATION MATCHING SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Matched {matched_pgs + matched_pheno}/{len(expected_features)} features")
    print(f"  PGS features matched:        {matched_pgs}")
    print(f"  Phenotypic features matched:  {matched_pheno}")

    if missing:
        print(f"\n⚠ Model expects {len(missing)} features not provided (filled with 0):")
        for m in missing[:10]:
            print(f"    • {m}")
        if len(missing) > 10:
            print(f"    … and {len(missing)-10} more")

    if unused_pgs or unused_pheno:
        print(f"\n⚠ {len(unused_pgs)+len(unused_pheno)} correlations provided but NOT used:")
        for r in sorted(unused_pgs):
            print(f"    • PGS:{r}  (not in training features)")
        for r in sorted(unused_pheno):
            print(f"    • Y{trait}:{r}  (not in training features)")

    print(f"{'='*70}\n")
    return X.reshape(1, -1)


# ============================================================================
# POSTERIOR SAMPLING AND REPORTING
# ============================================================================

N_PILOT = 20000          # unconstrained draws used to measure how much posterior lies in the prior
MIN_ACCEPTANCE = 0.01    # below this, rejection sampling is too slow: sample with MCMC instead


def _summarise(samples_np: np.ndarray, param_names: list) -> pd.DataFrame:
    """mean, median, std and 95% interval per parameter."""
    rows = []
    for i, param in enumerate(param_names):
        col = samples_np[:, i]
        ci_lo = np.percentile(col, 2.5)
        ci_hi = np.percentile(col, 97.5)
        rows.append({
            "parameter":       param,
            "mean":            float(col.mean()),
            "median":          float(np.median(col)),
            "std":             float(col.std()),
            "ci_lower_2.5":   float(ci_lo),
            "ci_upper_97.5":  float(ci_hi),
            "ci_width_95":    float(ci_hi - ci_lo),
        })
    return pd.DataFrame(rows)


def _inside_prior(posterior, theta: torch.Tensor) -> torch.Tensor:
    """True for samples inside the prior support, i.e. the simulated parameter range."""
    from sbi.utils import within_support
    if isinstance(posterior, npe_count.MixedPosterior):
        if posterior.flow_posterior is None:
            return torch.ones(len(theta), dtype=torch.bool)
        return within_support(posterior.flow_posterior.prior, theta[:, posterior.cont_index])
    return within_support(posterior.prior, theta)


def _sample_in_prior_mcmc(posterior, x_tensor: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Sample the flow density restricted to the prior support with slice-sampling MCMC.

    This is the posterior the rejection sampler targets; MCMC reaches it even when
    almost none of the flow's mass lies inside the prior.
    """
    from sbi.inference import MCMCPosterior
    from sbi.inference.potentials import posterior_estimator_based_potential
    potential_fn, theta_transform = posterior_estimator_based_potential(
        posterior.posterior_estimator, posterior.prior, x_o=x_tensor)
    mcmc = MCMCPosterior(potential_fn, proposal=posterior.prior, theta_transform=theta_transform,
                         method="slice_np_vectorized", num_chains=20, warmup_steps=500,
                         thin=5, init_strategy="resample")
    return mcmc.sample((n_samples,), x=x_tensor, show_progress_bars=False)


def sample_and_summarise(
    posterior,
    x_obs_scaled: np.ndarray,
    param_names: list,
    n_samples: int = 1000,
):
    """
    Draw posterior samples and compute summary statistics.

    The posterior is restricted to the prior support (the simulated parameter
    range): the flow can put mass outside it, where there are no simulations, and
    that mass is not part of the posterior. Unconstrained draws are summarised
    separately so the size of that extrapolation stays visible.

    Parameters
    ----------
    posterior : sbi NeuralPosterior (or npe_count.MixedPosterior)
    x_obs_scaled : np.ndarray  (1, n_features)  – already StandardScaler-transformed
    param_names : list of str
    n_samples : int  – number of posterior samples to draw

    Returns
    -------
    summary_df : pd.DataFrame with columns
        [parameter, mean, median, std, ci_lower_2.5, ci_upper_97.5, ci_width_95]
    samples_df : pd.DataFrame  (n_samples × n_params)  – posterior samples
    unconstrained_df : pd.DataFrame  – same summary for the unconstrained flow
    info : dict  – share of the unconstrained flow inside the prior, sampling method
    """
    x_tensor = torch.FloatTensor(x_obs_scaled)
    set_eval_mode(posterior)

    with torch.no_grad():
        raw = posterior.sample((N_PILOT,), x=x_tensor, show_progress_bars=False,
                               reject_outside_prior=False)
    inside = _inside_prior(posterior, raw)
    acceptance = float(inside.float().mean())
    unconstrained_df = _summarise(raw.cpu().numpy(), param_names)
    print(f"Unconstrained flow: {acceptance:.2%} of {N_PILOT:,} draws lie inside the prior "
          f"(simulated parameter range)")

    if acceptance >= MIN_ACCEPTANCE:
        method = "rejection"
        print(f"Drawing {n_samples:,} posterior samples (rejection outside the prior) …")
        with torch.no_grad():
            samples = posterior.sample((n_samples,), x=x_tensor, show_progress_bars=False,
                                       reject_outside_prior=True)
    elif isinstance(posterior, npe_count.MixedPosterior):
        raise RuntimeError(f"only {acceptance:.4%} of the flow lies inside the prior; MCMC "
                           "sampling is not implemented for count-parameter posteriors")
    else:
        method = "mcmc"
        print(f"⚠ Too little mass inside the prior for rejection sampling: drawing "
              f"{n_samples:,} samples with slice-sampling MCMC on the restricted density …")
        samples = _sample_in_prior_mcmc(posterior, x_tensor, n_samples)

    samples_np = samples.cpu().numpy()   # shape: (n_samples, n_params)
    summary_df = _summarise(samples_np, param_names)
    samples_df = pd.DataFrame(samples_np, columns=param_names)
    info = {"share_of_unconstrained_flow_inside_prior": acceptance,
            "n_pilot_draws": N_PILOT, "sampling_method": method, "n_samples": n_samples}
    return summary_df, samples_df, unconstrained_df, info


def display_posterior_summary(summary_df: pd.DataFrame):
    """Print a formatted table of posterior summary statistics."""
    print("\n" + "=" * 90)
    print("POSTERIOR SUMMARY  (Neural Posterior Estimation)")
    print("=" * 90)
    print(f"\n{'Parameter':<22} {'Mean':>10} {'Median':>10} {'Std':>10}  "
          f"{'95% CI Lower':>15} {'95% CI Upper':>15} {'CI Width':>10}")
    print("-" * 90)

    descriptions = {
        "f11":             "Vertical transmission (trait 1)",
        "f22":             "Vertical transmission (trait 2)",
        "f12":             "Cross-trait VT (trait 1 → trait 2)",
        "f21":             "Cross-trait VT (trait 2 → trait 1)",
        "prop_h2_latent1": "Proportion of h² that is latent (trait 1)",
        "vg1":             "Genetic variance (trait 1)",
        "vg2":             "Genetic variance (trait 2)",
        "am22":            "Assortative mating coefficient",
        "rg":              "Genetic correlation",
    }

    for _, row in summary_df.iterrows():
        p = row["parameter"]
        print(f"  {p:<20} {row['mean']:>10.4f} {row['median']:>10.4f} "
              f"{row['std']:>10.4f}  {row['ci_lower_2.5']:>15.4f} "
              f"{row['ci_upper_97.5']:>15.4f} {row['ci_width_95']:>10.4f}")

    print("=" * 90)
    print("\n  Interpretation:")
    print("  • Mean / Median  = point estimate (use mean for symmetric, median for skewed posteriors)")
    print("  • Std            = posterior uncertainty (wider = data less informative for this param)")
    print("  • 95% CI         = credible interval (95% probability the true value lies here)")
    print("=" * 90 + "\n")

    print("Parameter descriptions:")
    for _, row in summary_df.iterrows():
        p = row["parameter"]
        if p in descriptions:
            print(f"  {p:<22} {descriptions[p]}")


def plot_posterior_distributions(samples_df: pd.DataFrame, output_dir: Path, param_names: list):
    """Plot marginal posterior histograms for each predicted parameter."""
    import math
    import matplotlib.pyplot as plt

    n_params = len(param_names)
    n_cols   = min(4, n_params)
    n_rows   = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_params > 1 else [axes]

    for i, param in enumerate(param_names):
        ax = axes[i]
        col = samples_df[param].values

        ax.hist(col, bins=50, color="steelblue", edgecolor="white", alpha=0.8, density=True)
        ax.axvline(col.mean(), color="red",    linestyle="--", lw=2, label=f"Mean={col.mean():.4f}")
        ax.axvline(np.percentile(col, 2.5),  color="orange", linestyle=":",  lw=1.5, label="95% CI")
        ax.axvline(np.percentile(col, 97.5), color="orange", linestyle=":",  lw=1.5)
        ax.set_title(param, fontsize=11, fontweight="bold")
        ax.set_xlabel("Parameter value", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(n_params, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.suptitle("Posterior Distributions (marginal)", fontsize=13)
    plt.tight_layout()
    out = output_dir / "posterior_distributions.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved posterior distribution plots to {out}")


def plot_posterior_pairplot(samples_df: pd.DataFrame, output_dir: Path, max_params: int = 6):
    """Plot a pair-plot of the joint posterior for the first ``max_params`` parameters."""
    import matplotlib.pyplot as plt

    cols = samples_df.columns[:max_params].tolist()
    n    = len(cols)

    fig, axes = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n))

    for i, p_i in enumerate(cols):
        for j, p_j in enumerate(cols):
            ax = axes[i][j]
            if i == j:
                ax.hist(samples_df[p_i], bins=40, color="steelblue", edgecolor="none", density=True)
                ax.set_title(p_i, fontsize=8)
            elif j < i:
                ax.scatter(samples_df[p_j], samples_df[p_i],
                           alpha=0.05, s=2, color="steelblue", rasterized=True)
                ax.set_xlabel(p_j, fontsize=7)
                ax.set_ylabel(p_i, fontsize=7)
            else:
                ax.axis("off")
            ax.tick_params(labelsize=6)

    plt.suptitle("Joint Posterior Pair-Plot", fontsize=13)
    plt.tight_layout()
    out = output_dir / "posterior_pairplot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved pair-plot to {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sample posterior distributions from a trained NPE model "
                    "(sbi SNPE-C) given observed PGS + phenotypic correlations"
    )
    parser.add_argument("--model_dir", type=str, default="results_npe_weighted",
                        help="Directory containing trained NPE model")
    parser.add_argument("--correlations_pgs", type=str, default=None,
                        help="CSV file with PGS correlations (columns: RelType, Correlation)")
    parser.add_argument("--correlations_pheno", type=str, default=None,
                        help="CSV file with phenotypic (Y1) correlations (columns: RelType, Correlation)")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of posterior samples to draw (default: 1000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results (default: <model_dir>/predictions/)")
    parser.add_argument("--no_pairplot", action="store_true",
                        help="Skip pair-plot (can be slow for many parameters)")
    parser.add_argument("--trait", type=int, choices=[1, 2], default=1,
                        help="Trait number the model was trained on (1 or 2, default: 1)")

    # PGS correlations as CLI arguments
    for rel in ["S", "HSFS", "PSC", "PPSCC", "M", "MS", "SMS", "MSC", "MSM",
                "SMSC", "SMSM", "SMSMS", "MSMSM", "MSMSC", "PSMSC", "SMSMSC", "MSMSMS"]:
        parser.add_argument(f"--pgs_{rel}", type=float, default=None)

    # Phenotypic correlations as CLI arguments
    for rel in ["S", "HSFS", "PSC", "PPSCC", "M", "MS", "SMS", "MSC", "MSM",
                "SMSC", "SMSM", "SMSMS", "MSMSM", "MSMSC", "PSMSC", "SMSMSC", "MSMSMS"]:
        parser.add_argument(f"--pheno_{rel}", type=float, default=None)

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("PARAMETER INFERENCE — NPE POSTERIOR SAMPLING (sbi SNPE-C)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Resolve model directory
    # ------------------------------------------------------------------
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = Path(__file__).parent / args.model_dir
    if not model_dir.exists():
        print(f"\n✗ Model directory not found: {model_dir}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    posterior, feature_scaler, poly_transformer, config, param_names = \
        load_trained_posterior(model_dir)

    # ------------------------------------------------------------------
    # Collect observed correlations
    # ------------------------------------------------------------------
    rel_types = ["S", "HSFS", "PSC", "PPSCC", "M", "MS", "SMS", "MSC", "MSM",
                 "SMSC", "SMSM", "SMSMS", "MSMSM", "MSMSC", "PSMSC", "SMSMSC", "MSMSMS"]

    pgs_correlations_dict   = {}
    pheno_correlations_dict = {}

    if args.correlations_pgs:
        csv_path = Path(args.correlations_pgs)
        if not csv_path.is_absolute():
            alt = Path(__file__).parent / args.correlations_pgs
            if alt.exists():
                csv_path = alt
        print(f"\nLoading PGS correlations from: {csv_path}")
        df_pgs = pd.read_csv(csv_path)
        if "RelType" not in df_pgs or "Correlation" not in df_pgs.columns:
            print("✗ PGS CSV must have 'RelType' and 'Correlation' columns.")
            sys.exit(1)
        df_pgs = df_pgs.groupby("RelType")["Correlation"].mean().reset_index()
        pgs_correlations_dict = dict(zip(df_pgs["RelType"], df_pgs["Correlation"]))
    else:
        for r in rel_types:
            val = getattr(args, f"pgs_{r}", None)
            if val is not None:
                pgs_correlations_dict[r] = val

    if args.correlations_pheno:
        csv_path = Path(args.correlations_pheno)
        if not csv_path.is_absolute():
            alt = Path(__file__).parent / args.correlations_pheno
            if alt.exists():
                csv_path = alt
        print(f"\nLoading phenotypic (Y1) correlations from: {csv_path}")
        df_ph = pd.read_csv(csv_path)
        if "RelType" not in df_ph or "Correlation" not in df_ph.columns:
            print("✗ Phenotypic CSV must have 'RelType' and 'Correlation' columns.")
            sys.exit(1)
        df_ph = df_ph.groupby("RelType")["Correlation"].mean().reset_index()
        pheno_correlations_dict = dict(zip(df_ph["RelType"], df_ph["Correlation"]))
    else:
        for r in rel_types:
            val = getattr(args, f"pheno_{r}", None)
            if val is not None:
                pheno_correlations_dict[r] = val

    if not pgs_correlations_dict and not pheno_correlations_dict:
        print("\n✗ No correlations provided!")
        print("  Use --correlations_pgs <file> or --pgs_<REL> <value>")
        print("  Use --correlations_pheno <file> or --pheno_<REL> <value>")
        print("\nExample:")
        print("  python predict_npe_pgs_and_pheno.py \\")
        print("      --model_dir results_npe_weighted \\")
        print("      --correlations_pgs observed_pgs.csv \\")
        print("      --correlations_pheno observed_pheno.csv")
        sys.exit(1)

    # Echo inputs
    print("\nInput PGS correlations:")
    for r, v in sorted(pgs_correlations_dict.items()):
        print(f"  PGS  {r:<12} {v:.6f}")
    print(f"\nInput phenotypic (Y{args.trait}) correlations:")
    for r, v in sorted(pheno_correlations_dict.items()):
        print(f"  Y{args.trait}   {r:<12} {v:.6f}")

    # ------------------------------------------------------------------
    # Prepare and scale observation
    # ------------------------------------------------------------------
    X_raw = prepare_correlation_input(
        pgs_correlations_dict, pheno_correlations_dict, config["original_features"],
        trait=args.trait,
    )

    if poly_transformer is not None and config.get("interaction_degree", 1) > 1:
        X_raw = poly_transformer.transform(X_raw)
        print(f"  Applied polynomial features: {X_raw.shape[1]} features")

    X_scaled = feature_scaler.transform(X_raw)

    # ------------------------------------------------------------------
    # Sample posterior
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"POSTERIOR SAMPLING  (n_samples={args.n_samples:,})")
    print("=" * 70)

    torch.manual_seed(2026)
    np.random.seed(2026)
    summary_df, samples_df, unconstrained_df, sampling_info = sample_and_summarise(
        posterior, X_scaled, param_names, n_samples=args.n_samples
    )

    # ------------------------------------------------------------------
    # Display and save results
    # ------------------------------------------------------------------
    display_posterior_summary(summary_df)

    output_dir = Path(args.output) if args.output else model_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "posterior_summary.csv"
    samples_path = output_dir / "posterior_samples.csv"

    summary_df.to_csv(summary_path, index=False)
    samples_df.to_csv(samples_path, index=False)
    unconstrained_df.to_csv(output_dir / "posterior_summary_unconstrained.csv", index=False)
    with open(output_dir / "posterior_sampling.json", "w") as f:
        json.dump(sampling_info, f, indent=2)
    print(f"\n✓ Posterior summary  → {summary_path}")
    print(f"✓ Raw samples        → {samples_path}")
    print(f"✓ Unconstrained flow summary → {output_dir / 'posterior_summary_unconstrained.csv'} "
          f"({sampling_info['share_of_unconstrained_flow_inside_prior']:.2%} inside the prior; "
          f"sampled by {sampling_info['sampling_method']})")

    # Count parameter (e.g. n_gen): exact posterior probability of each value
    if config.get("count_param"):
        pmf = posterior.count_pmf(X_scaled)[0]
        pmf_df = pd.DataFrame({config["count_param"]: config["count_values"], "probability": pmf})
        pmf_path = output_dir / "posterior_count_pmf.csv"
        pmf_df.to_csv(pmf_path, index=False)
        print(f"\nPosterior of {config['count_param']} (mode = "
              f"{pmf_df.iloc[int(np.argmax(pmf)), 0]}):")
        print(pmf_df.round(3).to_string(index=False))
        print(f"✓ Count posterior    → {pmf_path}")

    # Plots
    plot_posterior_distributions(samples_df, output_dir, param_names)
    if not args.no_pairplot:
        plot_posterior_pairplot(samples_df, output_dir)

    print("\n✓ Inference complete.\n")
    print("Key outputs:")
    print(f"  posterior_summary.csv   – mean, median, std, 95% CI per parameter")
    print(f"  posterior_summary_unconstrained.csv – the same, without restricting to the prior")
    print(f"  posterior_samples.csv   – {args.n_samples:,} raw samples for further analysis")
    print(f"  posterior_distributions.png  – marginal density plots")
    if not args.no_pairplot:
        print(f"  posterior_pairplot.png       – joint posterior pair-plot\n")

    return summary_df, samples_df


if __name__ == "__main__":
    main()
