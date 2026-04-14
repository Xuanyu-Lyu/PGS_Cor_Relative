"""
Neural Posterior Estimation (NPE) for PGS + Phenotypic (Y1) Correlations

Converts the point-prediction neural network into a full Bayesian posterior
estimator using the ``sbi`` library (Sequential Neural Posterior Estimation,
SNPE-C / APT).

Architecture overview
---------------------
The model is split into two specialised components that are trained end-to-end:

1. **Embedding network** (``PgsPhenoEmbeddingNet``):
   Your existing ``FeatureAwarePredictorPgsPheno`` architecture, minus its
   final output layer.  It reads the raw (scaled) correlations, applies the
   attention mechanism, and compresses the information into a dense
   ``hidden_sizes[-1]``-dimensional summary vector.

2. **Normalizing Flow** (Neural Spline Flow, via ``sbi``):
   Takes the embedding vector as its condition and warps a simple base
   distribution into the full posterior over your simulation parameters.
   Trained with the Negative Log-Likelihood (NLL) of the flow.

Feature-Specific Regularization
--------------------------------
Because ``sbi`` owns the training loop, we inject feature-specific L2 penalties
by subclassing ``SNPE`` and overriding ``_train_epoch``.  Manual L2 terms are
added to each mini-batch NLL loss exactly as in the original code:

  - PGS-connected weights of the first embedding layer  →  ``--pgs_weight_decay``
  - Y1-connected weights of the first embedding layer   →  ``--pheno_weight_decay``
  - All other parameters                                 →  ``--weight_decay``

BatchNorm → LayerNorm
---------------------
``BatchNorm1d`` breaks for batch-size 1, which ``sbi`` uses when evaluating a
single observation during posterior sampling.  The embedding network therefore
uses ``LayerNorm`` instead, which normalises over the feature dimension and is
agnostic to batch size.

Usage
-----
    python train_npe_pgs_and_pheno.py \
        --data nn_training_01DirAM.csv \
        --output ./results_npe_unweighted_01DirAM_AE \
        --epochs 500 \
        --features_file ./results_npe_unweighted_01DirAM_AE/features.txt \
        --params_file ./results_npe_unweighted_01DirAM_AE/params.txt \
        --pheno_weight_decay 0 \
        --pgs_weight_decay 0
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import argparse
from pathlib import Path
import joblib
import json
import pickle

from sbi.inference import SNPE
from sbi.utils import BoxUniform
from sbi.neural_nets import posterior_nn

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# ATTENTION LAYER  (unchanged from original)
# ============================================================================

class AttentionLayer(nn.Module):
    """Self-attention layer to weight hidden features."""

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


# ============================================================================
# EMBEDDING NETWORK
# ============================================================================

class PgsPhenoEmbeddingNet(nn.Module):
    """
    Embedding network for NPE.

    Identical architecture to ``FeatureAwarePredictorPgsPheno`` but with the
    final linear output layer removed.  The network returns the last hidden
    representation (``deep_features``), which ``sbi``'s Normalizing Flow uses
    as its conditioning vector.

    ``LayerNorm`` replaces ``BatchNorm1d`` so that the network works correctly
    when called with a single sample during ``posterior.sample()``.

    The ``output_dim`` attribute is required by ``sbi`` to dimension the
    first layer of the Normalizing Flow.

    Parameters
    ----------
    n_features : int
        Total number of input features (PGS + Y1).
    hidden_sizes : list of int, length 4
        Widths of the four hidden layers
        [feat_layer1, feat_layer2, deep_layer1, deep_layer2].
    dropout_rate : float
        Dropout probability applied after every activation.
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list = None,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128, 128]

        self.feature_extractor = nn.Sequential(
            nn.Linear(n_features, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.attention = AttentionLayer(hidden_sizes[1])

        self.deeper = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.LayerNorm(hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.LayerNorm(hidden_sizes[3]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # sbi reads this attribute to set the first Flow layer's input size.
        self.output_dim = hidden_sizes[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        attended = self.attention(features)
        return self.deeper(attended)


# ============================================================================
# SNPE SUBCLASS WITH FEATURE-SPECIFIC REGULARIZATION
# ============================================================================

class SNPEFeatureReg(SNPE):
    """
    SNPE subclass that adds Feature-Specific L2 regularisation to the NLL loss.

    The base ``SNPE._train_epoch`` minimises only the Normalizing Flow's NLL.
    This override appends manual L2 penalties targeting specific columns of
    the embedding network's first linear layer, mirroring the original
    ``train_model`` approach from ``train_realistic_pgs_and_pheno_weighted.py``.

    Parameters
    ----------
    n_pgs : int
        Number of PGS feature columns (first ``n_pgs`` columns of input).
    n_y1 : int
        Number of Y1 phenotypic feature columns (columns ``n_pgs`` to
        ``n_pgs + n_y1``).
    pgs_weight_decay : float
        L2 coefficient for PGS-connected weights in the first embedding layer.
    pheno_weight_decay : float
        L2 coefficient for Y1-connected weights in the first embedding layer.
        Set higher than ``pgs_weight_decay`` to downweight Y1 dominance.
    global_weight_decay : float
        L2 coefficient applied to all other network parameters.
    """

    def __init__(
        self,
        prior,
        density_estimator,
        *,
        n_pgs: int = 0,
        n_y1: int = 0,
        pgs_weight_decay: float = 0.0001,
        pheno_weight_decay: float = 0.01,
        global_weight_decay: float = 0.0001,
        **kwargs,
    ):
        super().__init__(prior=prior, density_estimator=density_estimator, **kwargs)
        self._n_pgs = n_pgs
        self._n_y1 = n_y1
        self._pgs_wd = pgs_weight_decay
        self._pheno_wd = pheno_weight_decay
        self._global_wd = global_weight_decay

    def _train_epoch(self, train_loader, clip_max_norm, loss_args):
        """
        Single training epoch with feature-specific L2 regularisation.

        Adds manual L2 penalty terms to the NLL loss per feature group.
        The reported (and validated) loss is the raw NLL only, for
        comparability with the validation loss which has no regularisation.
        """
        assert self._neural_net is not None

        pgs_end = self._n_pgs
        y1_end = self._n_pgs + self._n_y1

        train_loss_sum = 0.0
        for batch in train_loader:
            self.optimizer.zero_grad()

            if loss_args is None:
                train_losses = self._get_losses(batch=batch)
            else:
                train_losses = self._get_losses(batch=batch, loss_args=loss_args)

            nll_loss = torch.mean(train_losses)

            # ------ Feature-Specific L2 Regularisation ------
            reg_loss = torch.tensor(0.0, device=self._device)
            try:
                first_w = self._neural_net.embedding_net.feature_extractor[0].weight

                if pgs_end > 0 and self._pgs_wd > 0:
                    reg_loss = reg_loss + self._pgs_wd * first_w[:, :pgs_end].pow(2).sum()

                if self._n_y1 > 0 and self._pheno_wd > 0:
                    reg_loss = reg_loss + self._pheno_wd * first_w[:, pgs_end:y1_end].pow(2).sum()

                if self._global_wd > 0:
                    for name, param in self._neural_net.named_parameters():
                        if (
                            name != "embedding_net.feature_extractor.0.weight"
                            and param.requires_grad
                        ):
                            reg_loss = reg_loss + self._global_wd * param.pow(2).sum()
            except (AttributeError, IndexError):
                # Architecture doesn't expose the expected path – skip feature-reg.
                pass

            total_loss = nll_loss + reg_loss
            total_loss.backward()

            if clip_max_norm is not None:
                clip_grad_norm_(self._neural_net.parameters(), max_norm=clip_max_norm)

            self.optimizer.step()

            # Track raw NLL (no reg) so the training-loss curve is on the same
            # scale as the validation-loss curve.
            train_loss_sum += train_losses.sum().item()

        return train_loss_sum / (len(train_loader) * train_loader.batch_size)


# ============================================================================
# DATA LOADING AND CLEANING  (preserved from original)
# ============================================================================

def load_and_clean_data(data_path, missing_threshold=0.95):
    """Load CSV and apply basic quality filters."""
    print("\n" + "=" * 70)
    print("LOADING AND CLEANING DATA")
    print("=" * 70)

    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} columns")

    initial_samples = len(df)
    cor_cols = [col for col in df.columns if col.startswith("cor_")]

    if len(cor_cols) == 0:
        print("\n⚠ Warning: No correlation columns found!")
        return df

    missing_per_row = df[cor_cols].isnull().sum(axis=1) / len(cor_cols)

    print(f"\nMissing data distribution:")
    print(f"  Min: {missing_per_row.min():.1%}")
    print(f"  Mean: {missing_per_row.mean():.1%}")
    print(f"  Median: {missing_per_row.median():.1%}")
    print(f"  Max: {missing_per_row.max():.1%}")

    df = df[missing_per_row < missing_threshold].copy()
    print(f"\n✓ Removed {initial_samples - len(df)} rows with >{missing_threshold:.0%} missing correlations")
    print(f"  Remaining: {len(df)} samples")

    if len(df) == 0:
        print("\n✗ ERROR: All rows removed during cleaning!")
        print("  Try adjusting --missing_threshold parameter")
        return df

    param_cols = [col for col in df.columns if col.startswith("param_")]
    dedup_cols = param_cols + (["Condition", "Iteration"] if "Condition" in df.columns else ["Iteration"])
    before_dedup = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first")
    print(f"✓ Removed {before_dedup - len(df)} duplicate rows. Remaining: {len(df)} samples")

    outlier_mask = (df[cor_cols].abs() > 1).any(axis=1)
    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        print(f"⚠ Warning: Removed {n_outliers} rows with |r| > 1")
        df = df[~outlier_mask].copy()
        print(f"  Remaining: {len(df)} samples")

    return df


def engineer_features(df, feature_cols, interaction_degree=1):
    """Optionally add polynomial interaction features."""
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    X = df[feature_cols].values
    X_filled = np.nan_to_num(X, nan=0.0)
    print(f"\nOriginal features: {X.shape[1]}")

    if interaction_degree > 1:
        poly = PolynomialFeatures(degree=interaction_degree, include_bias=False)
        X_poly = poly.fit_transform(X_filled)
        print(f"After polynomial (degree={interaction_degree}): {X_poly.shape[1]}")
        feature_names = (
            list(poly.get_feature_names_out(feature_cols))
            if hasattr(poly, "get_feature_names_out")
            else [f"poly_{i}" for i in range(X_poly.shape[1])]
        )
        return X_poly, feature_names, poly

    return X_filled, list(feature_cols), None


# ============================================================================
# NPE EVALUATION  (replaces evaluate_model + plot_predictions)
# ============================================================================

def evaluate_npe(posterior, X_test, y_test, param_names, output_dir, n_posterior_samples=500):
    """
    Evaluate the trained NPE on the held-out test set.

    For each test observation we draw ``n_posterior_samples`` samples from
    the posterior and use their mean as the point estimate and their standard
    deviation as the uncertainty.

    Parameters
    ----------
    posterior : sbi NeuralPosterior
    X_test : np.ndarray  (n_test, n_features)  – already scaled
    y_test : np.ndarray  (n_test, n_params)    – original (unscaled) parameter values
    param_names : list of str
    output_dir : Path
    n_posterior_samples : int

    Returns
    -------
    results : dict mapping param_name → dict(r2, rmse, mae, true, pred_mean, pred_std)
    """
    print("\n" + "=" * 70)
    print(f"EVALUATING NPE ON TEST SET  (n_posterior_samples={n_posterior_samples})")
    print("=" * 70)

    n_test = min(len(X_test), 500)   # cap to keep evaluation tractable
    if n_test < len(X_test):
        print(f"  ⚠ Evaluating on first {n_test}/{len(X_test)} test samples for speed.")

    pred_means = []
    pred_stds = []

    # DirectPosterior is not a nn.Module; set the underlying density estimator to eval mode
    if hasattr(posterior, '_neural_net'):
        posterior._neural_net.eval()
    for i in range(n_test):
        x_obs = torch.FloatTensor(X_test[i]).unsqueeze(0)
        with torch.no_grad():
            samples = posterior.sample((n_posterior_samples,), x=x_obs, show_progress_bars=False)
        pred_means.append(samples.mean(0).numpy())
        pred_stds.append(samples.std(0).numpy())

    pred_means = np.array(pred_means)
    pred_stds  = np.array(pred_stds)
    y_true     = y_test[:n_test]

    results = {}
    print(f"\n{'Parameter':<22} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'Mean Posterior σ'}")
    print("-" * 70)

    for i, param in enumerate(param_names):
        y_t = y_true[:, i]
        y_p = pred_means[:, i]
        y_s = pred_stds[:, i]

        r2   = r2_score(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae  = mean_absolute_error(y_t, y_p)

        results[param] = dict(r2=r2, rmse=rmse, mae=mae, mean_sigma=y_s.mean(),
                              true=y_t, pred_mean=y_p, pred_std=y_s)
        print(f"{param:<22} {r2:<10.4f} {rmse:<10.4f} {mae:<10.4f} {y_s.mean():.4f}")

    overall_r2   = r2_score(y_true.flatten(), pred_means.flatten())
    overall_rmse = np.sqrt(mean_squared_error(y_true.flatten(), pred_means.flatten()))
    print("-" * 70)
    print(f"{'OVERALL':<22} {overall_r2:<10.4f} {overall_rmse:<10.4f}")
    print("=" * 70 + "\n")

    # Save metrics
    metrics_dict = {
        p: {k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in vals.items()}
        for p, vals in results.items()
    }
    metrics_dict["overall"] = {"r2": float(overall_r2), "rmse": float(overall_rmse)}
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    return results


def plot_npe_results(results, output_dir, param_names):
    """Generate prediction scatter plots and posterior-uncertainty bar plots."""
    import math

    # --- Prediction vs True ---
    n_params = len(param_names)
    n_cols = 4
    n_rows = math.ceil(n_params / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, param in enumerate(param_names):
        ax = axes[i]
        y_t = results[param]["true"]
        y_p = results[param]["pred_mean"]
        y_s = results[param]["pred_std"]
        r2  = results[param]["r2"]

        ax.errorbar(y_t, y_p, yerr=y_s, fmt="o", alpha=0.3, markersize=3, elinewidth=0.5)
        lo = min(y_t.min(), y_p.min())
        hi = max(y_t.max(), y_p.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=2)
        ax.set_xlabel("True", fontsize=9)
        ax.set_ylabel("Predicted (posterior mean)", fontsize=9)
        ax.set_title(f"{param}  (R²={r2:.3f})", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

    for j in range(n_params, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.suptitle("NPE: Posterior Mean vs True (error bars = posterior σ)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "predictions_vs_true.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Posterior width per parameter ---
    fig, ax = plt.subplots(figsize=(max(8, n_params), 4))
    mean_sigmas = [results[p]["mean_sigma"] for p in param_names]
    bars = ax.bar(param_names, mean_sigmas, color="steelblue", edgecolor="white")
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Mean Posterior σ")
    ax.set_title("Posterior Uncertainty per Parameter")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, mean_sigmas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "posterior_uncertainty.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved prediction plots to", output_dir)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train NPE (sbi SNPE) for PGS + Phenotypic correlation data"
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Path to training CSV")
    parser.add_argument("--output", type=str, default="results_npe_weighted",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Maximum training epochs")
    parser.add_argument("--stop_after_epochs", type=int, default=50,
                        help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training mini-batch size")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate for Adam optimiser")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="Global L2 penalty for non-first-layer parameters")
    parser.add_argument("--pgs_weight_decay", type=float, default=0.0001,
                        help="L2 penalty for PGS-connected first-layer weights")
    parser.add_argument("--pheno_weight_decay", type=float, default=0.01,
                        help="L2 penalty for Y1-connected first-layer weights "
                             "(set higher than pgs_weight_decay to downweight Y1 influence)")
    parser.add_argument("--hidden_sizes", type=int, nargs="+",
                        default=[256, 256, 128, 128],
                        help="Embedding network hidden layer widths (4 values)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate in embedding network")
    parser.add_argument("--flow_type", type=str, default="nsf",
                        choices=["nsf", "maf", "maf_rqs", "mdn"],
                        help="Normalizing flow architecture")
    parser.add_argument("--flow_hidden", type=int, default=64,
                        help="Hidden features within the normalizing flow")
    parser.add_argument("--flow_transforms", type=int, default=5,
                        help="Number of flow transforms")
    parser.add_argument("--interaction_degree", type=int, default=1,
                        help="Polynomial degree for feature interactions (1=no interactions). "
                             "Feature-Specific Regularization requires degree=1.")
    parser.add_argument("--features_file", type=str, required=True,
                        help="Path to a plain-text file listing the full input feature column "
                             "names (one per line). Columns containing '_PGS' are treated as "
                             "PGS features; all others as phenotypic (Y1) features.")
    parser.add_argument("--params_file", type=str, required=True,
                        help="Path to a plain-text file listing the parameter names to predict "
                             "(one per line). Names must match param_<name> columns in the CSV.")
    parser.add_argument("--missing_threshold", type=float, default=0.95,
                        help="Max fraction of missing correlations per row")
    parser.add_argument("--prior_buffer", type=float, default=0.1,
                        help="Fraction of parameter range added as buffer around "
                             "training-data min/max when defining the BoxUniform prior")
    parser.add_argument("--n_posterior_samples", type=int, default=500,
                        help="Posterior samples per test observation for evaluation")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("NEURAL POSTERIOR ESTIMATION — PGS + PHENOTYPIC (SNPE + FEATURE-REG)")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print("=" * 70)

    if args.interaction_degree > 1:
        print("\n⚠ WARNING: --interaction_degree > 1 detected.")
        print("  Feature-Specific Regularization targets the original PGS / Y1 feature")
        print("  columns in the first layer.  With polynomial expansion, interaction")
        print("  terms (PGS×Y1 etc.) receive only the global weight_decay penalty.")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"\nUsing device: {device}")

    params_path = Path(args.params_file)
    if not params_path.exists():
        print(f"\n✗ ERROR: Params file not found: {params_path}")
        return
    active_param_names = [ln.strip() for ln in params_path.read_text().splitlines() if ln.strip()]
    if not active_param_names:
        print(f"\n✗ ERROR: No parameter names found in {params_path}")
        return
    print(f"\nParameter set: {len(active_param_names)} params from {params_path.name}")
    print(f"  {active_param_names}")

    # ------------------------------------------------------------------
    # 1. Load and clean data
    # ------------------------------------------------------------------
    df = load_and_clean_data(args.data, missing_threshold=args.missing_threshold)
    if len(df) == 0:
        print("\n✗ No data available after cleaning. Exiting.")
        return

    # ------------------------------------------------------------------
    # 2. Select feature columns
    # ------------------------------------------------------------------
    features_path = Path(args.features_file)
    if not features_path.exists():
        print(f"\n✗ ERROR: Features file not found: {features_path}")
        return
    combined_feature_cols = [ln.strip() for ln in features_path.read_text().splitlines() if ln.strip()]
    if not combined_feature_cols:
        print(f"\n✗ ERROR: No feature names found in {features_path}")
        return
    missing_feat = [c for c in combined_feature_cols if c not in df.columns]
    if missing_feat:
        print(f"\n✗ ERROR: Feature columns not found in data: {missing_feat}")
        print(f"  Available columns (sample): {list(df.columns[:20])}")
        return

    # Split for feature-specific regularization: columns containing '_PGS' are PGS features
    pgs1_cols = [c for c in combined_feature_cols if "_PGS" in c]
    y1_cols   = [c for c in combined_feature_cols if "_PGS" not in c]

    print(f"\nFeatures loaded from {features_path.name}: {len(combined_feature_cols)} total")
    print(f"  PGS features ({len(pgs1_cols)}): {pgs1_cols}")
    print(f"  Y1  features ({len(y1_cols)}):  {y1_cols}")

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    X_engineered, feature_names, poly_transformer = engineer_features(
        df, combined_feature_cols, args.interaction_degree
    )

    # ------------------------------------------------------------------
    # 4. Prepare targets (theta)
    # ------------------------------------------------------------------
    has_param_prefix = any(f"param_{p}" in df.columns for p in active_param_names)
    target_cols = [f"param_{p}" for p in active_param_names] if has_param_prefix else active_param_names
    missing_cols = [c for c in target_cols if c not in df.columns]
    if missing_cols:
        print(f"\n✗ ERROR: Target columns not found in data: {missing_cols}")
        print(f"  Available param columns: {[c for c in df.columns if c.startswith('param_')]}")
        return
    y = df[target_cols].values  # raw, unscaled – sbi handles internal normalization

    # ------------------------------------------------------------------
    # 5. Train/val/test split
    # ------------------------------------------------------------------
    n_samples = len(df)
    n_train = int(n_samples * 0.70)
    n_val   = int(n_samples * 0.15)

    rng = np.random.default_rng(42)
    indices = rng.permutation(n_samples)

    X_train = X_engineered[indices[:n_train]]
    y_train = y[indices[:n_train]]
    X_val   = X_engineered[indices[n_train:n_train + n_val]]
    y_val   = y[indices[n_train:n_train + n_val]]
    X_test  = X_engineered[indices[n_train + n_val:]]
    y_test  = y[indices[n_train + n_val:]]

    print(f"\nData split: {len(X_train)} train | {len(X_val)} val | {len(X_test)} test")

    # ------------------------------------------------------------------
    # 6. Scale features  (theta NOT scaled — sbi handles that internally)
    # ------------------------------------------------------------------
    feature_scaler = StandardScaler()
    X_train_s = feature_scaler.fit_transform(X_train)
    X_val_s   = feature_scaler.transform(X_val)
    X_test_s  = feature_scaler.transform(X_test)

    joblib.dump(feature_scaler, output_dir / "feature_scaler.pkl")
    if poly_transformer:
        joblib.dump(poly_transformer, output_dir / "poly_transformer.pkl")

    # ------------------------------------------------------------------
    # 7. Define BoxUniform prior from training data range (+buffer)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PRIOR DEFINITION (BoxUniform from training data range)")
    print("=" * 70)

    y_min = y_train.min(axis=0)
    y_max = y_train.max(axis=0)
    buffer = args.prior_buffer * (y_max - y_min)
    prior_lower = torch.tensor(y_min - buffer, dtype=torch.float32)
    prior_upper = torch.tensor(y_max + buffer, dtype=torch.float32)

    print(f"\n{'Parameter':<22} {'Lower':<12} {'Upper':<12}")
    print("-" * 50)
    for p, lo, hi in zip(active_param_names, prior_lower, prior_upper):
        print(f"  {p:<20} {lo.item():+.4f}      {hi.item():+.4f}")

    prior = BoxUniform(low=prior_lower, high=prior_upper, device=str(device))

    # ------------------------------------------------------------------
    # 8. Build embedding network and density estimator
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE")
    print("=" * 70)

    n_features = X_engineered.shape[1]
    embedding_net = PgsPhenoEmbeddingNet(
        n_features=n_features,
        hidden_sizes=args.hidden_sizes,
        dropout_rate=args.dropout,
    )

    print(f"\nEmbedding network:")
    print(f"  Input features : {n_features}  ({len(pgs1_cols)} PGS + {len(y1_cols)} Y1)")
    print(f"  Hidden layers  : {args.hidden_sizes}")
    print(f"  Output dim     : {embedding_net.output_dim}  (→ Normalizing Flow condition)")
    total_emb = sum(p.numel() for p in embedding_net.parameters())
    print(f"  Parameters     : {total_emb:,}")

    density_estimator_fn = posterior_nn(
        model=args.flow_type,
        embedding_net=embedding_net,
        hidden_features=args.flow_hidden,
        num_transforms=args.flow_transforms,
        z_score_theta="independent",  # sbi z-scores theta internally
        z_score_x="independent",      # normalize embedding output before the flow (prevents NSF spline instability)
    )

    print(f"\nNormalizing Flow:")
    print(f"  Type           : {args.flow_type.upper()}")
    print(f"  Hidden features: {args.flow_hidden}")
    print(f"  Transforms     : {args.flow_transforms}")

    # ------------------------------------------------------------------
    # 9. Train SNPE with feature-specific regularisation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING (SNPE + Feature-Specific Regularization)")
    print("=" * 70)
    print(f"\n  Global L2 (non-first-layer):           {args.weight_decay}")
    print(f"  PGS first-layer L2:                    {args.pgs_weight_decay}")
    print(f"  Y1 phenotypic first-layer L2:          {args.pheno_weight_decay}")
    print(f"  n_pgs_features: {len(pgs1_cols)}  |  n_y1_features: {len(y1_cols)}")

    inference = SNPEFeatureReg(
        prior=prior,
        density_estimator=density_estimator_fn,
        device=str(device),
        n_pgs=len(pgs1_cols),
        n_y1=len(y1_cols),
        pgs_weight_decay=args.pgs_weight_decay,
        pheno_weight_decay=args.pheno_weight_decay,
        global_weight_decay=args.weight_decay,
    )

    # Combine train + val for sbi (it manages its own validation split)
    X_all = np.vstack([X_train_s, X_val_s])
    y_all = np.vstack([y_train, y_val])

    inference.append_simulations(
        theta=torch.tensor(y_all, dtype=torch.float32),
        x=torch.tensor(X_all, dtype=torch.float32),
    )

    density_estimator = inference.train(
        training_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_num_epochs=args.epochs,
        stop_after_epochs=args.stop_after_epochs,
        show_train_summary=True,
        validation_fraction=len(X_val_s) / len(X_all),
    )

    # ------------------------------------------------------------------
    # 10. Build and save posterior
    # ------------------------------------------------------------------
    posterior = inference.build_posterior(density_estimator)

    print("\n✓ Posterior built successfully.")

    # Save density estimator weights + architecture so it can be reconstructed
    torch.save(
        {
            "density_estimator_state_dict": density_estimator.state_dict(),
            "prior_lower": prior_lower,
            "prior_upper": prior_upper,
        },
        output_dir / "density_estimator.pt",
    )

    # Save the full posterior object (for direct use in predict script)
    with open(output_dir / "posterior.pkl", "wb") as f:
        pickle.dump(posterior, f)
    print(f"✓ Saved posterior  → {output_dir / 'posterior.pkl'}")

    # ------------------------------------------------------------------
    # 11. Evaluate on held-out test set
    # ------------------------------------------------------------------
    results = evaluate_npe(
        posterior, X_test_s, y_test, active_param_names,
        output_dir, n_posterior_samples=args.n_posterior_samples
    )
    plot_npe_results(results, output_dir, active_param_names)

    # ------------------------------------------------------------------
    # 12. Save config
    # ------------------------------------------------------------------
    config = {
        "model_type": "NPE",
        "flow_type": args.flow_type,
        "flow_hidden": args.flow_hidden,
        "flow_transforms": args.flow_transforms,
        "n_original_pgs1_features": len(pgs1_cols),
        "n_original_y1_features": len(y1_cols),
        "n_original_features": len(combined_feature_cols),
        "n_engineered_features": n_features,
        "interaction_degree": args.interaction_degree,
        "hidden_sizes": args.hidden_sizes,
        "dropout_rate": args.dropout,
        "param_names": active_param_names,
        "original_features": combined_feature_cols,
        "pgs1_features": pgs1_cols,
        "y1_features": y1_cols,
        "regularization": "feature_specific",
        "weight_decay": args.weight_decay,
        "pgs_weight_decay": args.pgs_weight_decay,
        "pheno_weight_decay": args.pheno_weight_decay,
        "prior_lower": prior_lower.tolist(),
        "prior_upper": prior_upper.tolist(),
        "note": (
            "NPE (SNPE-C) model: PgsPhenoEmbeddingNet (LayerNorm, no output layer) "
            "feeds a Neural Spline Flow.  Feature-Specific Regularization is applied "
            "by overriding SNPE._train_epoch with manual L2 penalties."
        ),
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved config     → {output_dir / 'config.json'}")

    print("\n✓ NPE Training complete!\n")
    print("Next step: run predict_npe_pgs_and_pheno.py to get posterior samples")
    print("  for your observed data with full uncertainty quantification.\n")


if __name__ == "__main__":
    main()
