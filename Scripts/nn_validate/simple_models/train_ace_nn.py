"""
Neural Posterior Estimation (NPE) for ACE Parameter Prediction

Uses sbi (SNPE-C / APT) to learn the full posterior over A, C, E variance
components given MZ and DZ twin covariance matrix elements.

Architecture overview
---------------------
1. **Embedding network** (``ACEEmbeddingNet``):
   A small LayerNorm MLP that compresses the 4 (or 5) input features into a
   dense summary vector.  LayerNorm is used instead of BatchNorm1d so that
   the network works correctly when called with batch_size=1 during posterior
   sampling.

2. **Normalizing Flow** (Neural Spline Flow, via ``sbi``):
   Conditions on the embedding vector and outputs the full posterior over the
   3 ACE parameters.

Input features  (4):  mz_var, mz_cov, dz_var, dz_cov
                      Optionally include N_pairs with --include_n_pairs
Output targets  (3):  A (additive genetic), C (shared env), E (unique env)

Usage:
    # 1. Generate training data first
    python generate_ace_data.py --n_samples 20000

    # 2. Train
    python train_ace_nn.py --data ace_training_data.csv --epochs 500

    # 3. Optionally add N_pairs as a feature
    python train_ace_nn.py --data ace_training_data.csv --include_n_pairs
"""

import math
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sbi.inference import SNPE
from sbi.utils import BoxUniform
from sbi.neural_nets import posterior_nn

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

ACE_PARAM_NAMES = ['A', 'C', 'E']


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_clean_data(data_path):
    print("\n" + "="*70)
    print("LOADING AND CLEANING DATA")
    print("="*70)

    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} columns")

    initial_samples = len(df)
    feature_cols  = ['mz_var', 'mz_cov', 'dz_var', 'dz_cov']
    required_cols = feature_cols + ACE_PARAM_NAMES

    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Required columns missing from data: {missing_required}")

    df = df.dropna(subset=required_cols).copy()
    print(f"✓ Removed {initial_samples - len(df)} rows with missing values")

    invalid_mz = (df['mz_cov'].abs() > df['mz_var'].abs())
    invalid_dz = (df['dz_cov'].abs() > df['dz_var'].abs())
    n_invalid = (invalid_mz | invalid_dz).sum()
    if n_invalid > 0:
        df = df[~(invalid_mz | invalid_dz)].copy()
        print(f"✓ Removed {n_invalid} rows with invalid covariance matrices")

    print(f"\nFinal sample size: {len(df)}")
    print("="*70)
    return df


# ============================================================================
# EMBEDDING NETWORK
# ============================================================================

class ACEEmbeddingNet(nn.Module):
    """
    Lightweight embedding network for NPE on the ACE model.

    Uses LayerNorm instead of BatchNorm1d so that it works correctly
    when called with a single sample during posterior.sample().
    The ``output_dim`` attribute is required by sbi.
    """

    def __init__(self, n_features=4, hidden_sizes=None, dropout_rate=0.2):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64, 32]

        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_size, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
            in_size = h

        self.network = nn.Sequential(*layers)
        self.output_dim = in_size   # required by sbi

    def forward(self, x):
        return self.network(x)


# ============================================================================
# NPE EVALUATION
# ============================================================================

def evaluate_npe(posterior, X_test, y_test, param_names, output_dir,
                 n_posterior_samples=500, device=None):
    """
    Evaluate the trained NPE on the held-out test set.

    For each test observation, draw ``n_posterior_samples`` samples from the
    posterior and use their mean as the point estimate and std as uncertainty.
    """
    print("\n" + "="*70)
    print(f"EVALUATING NPE ON TEST SET  (n_posterior_samples={n_posterior_samples})")
    print("="*70)

    n_test = min(len(X_test), 500)
    if n_test < len(X_test):
        print(f"  ⚠ Evaluating on first {n_test}/{len(X_test)} test samples for speed.")

    if hasattr(posterior, '_neural_net'):
        posterior._neural_net.eval()

    pred_means, pred_stds = [], []
    for i in range(n_test):
        x_obs = torch.FloatTensor(X_test[i]).unsqueeze(0)
        if device is not None:
            x_obs = x_obs.to(device)
        with torch.no_grad():
            samples = posterior.sample(
                (n_posterior_samples,), x=x_obs, show_progress_bars=False
            )
        pred_means.append(samples.cpu().mean(0).numpy())
        pred_stds.append(samples.cpu().std(0).numpy())

    pred_means = np.array(pred_means)
    pred_stds  = np.array(pred_stds)
    y_true     = y_test[:n_test]

    results = {}
    print(f"\n{'Parameter':<22} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'Mean Posterior σ'}")
    print("-"*70)

    for i, param in enumerate(param_names):
        y_t = y_true[:, i]
        y_p = pred_means[:, i]
        y_s = pred_stds[:, i]
        r2   = r2_score(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae  = mean_absolute_error(y_t, y_p)
        results[param] = dict(r2=r2, rmse=rmse, mae=mae, mean_sigma=float(y_s.mean()),
                              true=y_t, pred_mean=y_p, pred_std=y_s)
        print(f"{param:<22} {r2:<10.4f} {rmse:<10.4f} {mae:<10.4f} {y_s.mean():.4f}")

    overall_r2   = r2_score(y_true.flatten(), pred_means.flatten())
    overall_rmse = np.sqrt(mean_squared_error(y_true.flatten(), pred_means.flatten()))
    print("-"*70)
    print(f"{'OVERALL':<22} {overall_r2:<10.4f} {overall_rmse:<10.4f}")
    print("="*70 + "\n")

    metrics_dict = {
        p: {k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in vals.items()}
        for p, vals in results.items()
    }
    metrics_dict['overall'] = {'r2': float(overall_r2), 'rmse': float(overall_rmse)}
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_npe_results(results, output_dir, param_names):
    """Prediction scatter plots and posterior uncertainty bar plot."""
    n_params = len(param_names)
    n_cols   = min(n_params, 4)
    n_rows   = math.ceil(n_params / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    for i, param in enumerate(param_names):
        ax  = axes[i]
        y_t = results[param]['true']
        y_p = results[param]['pred_mean']
        y_s = results[param]['pred_std']
        r2  = results[param]['r2']

        ax.errorbar(y_t, y_p, yerr=y_s, fmt='o', alpha=0.4, markersize=3, elinewidth=0.5)
        lo, hi = min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=2)
        ax.set_xlabel('True', fontsize=10)
        ax.set_ylabel('Predicted (posterior mean)', fontsize=10)
        ax.set_title(f'{param}  (R²={r2:.3f})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    for j in range(n_params, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.suptitle('NPE: Posterior Mean vs True (error bars = posterior σ)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_vs_true.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Posterior uncertainty bar chart
    fig, ax = plt.subplots(figsize=(max(6, n_params * 2), 4))
    mean_sigmas = [results[p]['mean_sigma'] for p in param_names]
    bars = ax.bar(param_names, mean_sigmas, color='steelblue', edgecolor='white')
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Mean Posterior σ')
    ax.set_title('Posterior Uncertainty per Parameter')
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, mean_sigmas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / 'posterior_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved prediction plots to {output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train NPE (sbi SNPE-C) for ACE parameter estimation'
    )
    parser.add_argument('--data', type=str, default='ace_training_data.csv',
                        help='Path to training CSV (default: ace_training_data.csv)')
    parser.add_argument('--output', type=str, default='results_ace_npe',
                        help='Output directory (default: results_ace_npe)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Maximum training epochs (default: 500)')
    parser.add_argument('--stop_after_epochs', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training mini-batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='AdamW weight decay (default: 1e-4)')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[64, 64, 32],
                        help='Embedding network hidden layer widths (default: 64 64 32)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--flow_type', type=str, default='nsf',
                        choices=['nsf', 'maf', 'maf_rqs', 'mdn'],
                        help='Normalizing flow type (default: nsf)')
    parser.add_argument('--flow_hidden', type=int, default=64,
                        help='Hidden features within the flow (default: 64)')
    parser.add_argument('--flow_transforms', type=int, default=5,
                        help='Number of flow transforms (default: 5)')
    parser.add_argument('--prior_buffer', type=float, default=0.1,
                        help='Buffer fraction around training data range for prior (default: 0.1)')
    parser.add_argument('--n_posterior_samples', type=int, default=500,
                        help='Posterior samples per test observation for evaluation (default: 500)')
    parser.add_argument('--include_n_pairs', action='store_true',
                        help='Include N_pairs as an input feature')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, mps (default: auto)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_path  = Path(args.data)
    if not data_path.is_absolute():
        data_path = script_dir / data_path
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("ACE NEURAL POSTERIOR ESTIMATION (SNPE-C)")
    print("="*70)
    print(f"Output: {output_dir}")
    print("="*70)

    # ---- Device ----
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    # ---- Load data ----
    df = load_and_clean_data(data_path)

    feature_cols = ['mz_var', 'mz_cov', 'dz_var', 'dz_cov']
    if args.include_n_pairs and 'N_pairs' in df.columns:
        feature_cols.append('N_pairs')
        print("  Including N_pairs as feature")

    X = df[feature_cols].values
    y = df[ACE_PARAM_NAMES].values

    # ---- Train/val/test split (70/15/15) ----
    n_samples = len(df)
    rng       = np.random.default_rng(42)
    indices   = rng.permutation(n_samples)
    n_train   = int(n_samples * 0.70)
    n_val     = int(n_samples * 0.15)

    X_train = X[indices[:n_train]]
    y_train = y[indices[:n_train]]
    X_val   = X[indices[n_train:n_train + n_val]]
    y_val   = y[indices[n_train:n_train + n_val]]
    X_test  = X[indices[n_train + n_val:]]
    y_test  = y[indices[n_train + n_val:]]
    print(f"\nData split: {len(X_train)} train | {len(X_val)} val | {len(X_test)} test")

    # ---- Scale features  (theta NOT scaled — sbi handles that internally) ----
    feature_scaler = StandardScaler()
    X_train_s = feature_scaler.fit_transform(X_train)
    X_val_s   = feature_scaler.transform(X_val)
    X_test_s  = feature_scaler.transform(X_test)
    joblib.dump(feature_scaler, output_dir / 'feature_scaler.pkl')

    # ---- Define BoxUniform prior from training data range ----
    print("\n" + "="*70)
    print("PRIOR DEFINITION (BoxUniform from training data range)")
    print("="*70)

    y_min  = y_train.min(axis=0)
    y_max  = y_train.max(axis=0)
    buffer = args.prior_buffer * (y_max - y_min)
    prior_lower = torch.tensor(y_min - buffer, dtype=torch.float32)
    prior_upper = torch.tensor(y_max + buffer, dtype=torch.float32)

    print(f"\n{'Parameter':<12} {'Lower':<12} {'Upper':<12}")
    print("-"*38)
    for p, lo, hi in zip(ACE_PARAM_NAMES, prior_lower, prior_upper):
        print(f"  {p:<10} {lo.item():+.4f}      {hi.item():+.4f}")

    prior = BoxUniform(low=prior_lower, high=prior_upper, device=str(device))

    # ---- Build embedding network and density estimator ----
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)

    n_features    = X_train.shape[1]
    embedding_net = ACEEmbeddingNet(
        n_features=n_features,
        hidden_sizes=args.hidden_sizes,
        dropout_rate=args.dropout,
    )

    print(f"\nEmbedding network:")
    print(f"  Input features : {n_features}")
    print(f"  Hidden layers  : {args.hidden_sizes}")
    print(f"  Output dim     : {embedding_net.output_dim}  (→ Normalizing Flow condition)")
    print(f"  Parameters     : {sum(p.numel() for p in embedding_net.parameters()):,}")

    density_estimator_fn = posterior_nn(
        model=args.flow_type,
        embedding_net=embedding_net,
        hidden_features=args.flow_hidden,
        num_transforms=args.flow_transforms,
        z_score_theta='independent',
        z_score_x='independent',
    )

    print(f"\nNormalizing Flow:")
    print(f"  Type           : {args.flow_type.upper()}")
    print(f"  Hidden features: {args.flow_hidden}")
    print(f"  Transforms     : {args.flow_transforms}")

    # ---- Train ----
    print("\n" + "="*70)
    print("TRAINING (SNPE-C)")
    print("="*70)

    inference = SNPE(
        prior=prior,
        density_estimator=density_estimator_fn,
        device=str(device),
    )

    X_all = np.vstack([X_train_s, X_val_s])
    y_all = np.vstack([y_train,   y_val])

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

    # ---- Build and save posterior ----
    posterior = inference.build_posterior(density_estimator)
    print("\n✓ Posterior built successfully.")

    torch.save(
        {
            'density_estimator_state_dict': density_estimator.state_dict(),
            'prior_lower': prior_lower,
            'prior_upper': prior_upper,
        },
        output_dir / 'density_estimator.pt',
    )
    with open(output_dir / 'posterior.pkl', 'wb') as f:
        pickle.dump(posterior, f)
    print(f"✓ Saved posterior  → {output_dir / 'posterior.pkl'}")

    # ---- Evaluate and plot ----
    results = evaluate_npe(
        posterior, X_test_s, y_test, ACE_PARAM_NAMES,
        output_dir, n_posterior_samples=args.n_posterior_samples,
        device=device,
    )
    plot_npe_results(results, output_dir, ACE_PARAM_NAMES)

    # ---- Save config ----
    config = {
        'model_type': 'NPE',
        'flow_type': args.flow_type,
        'flow_hidden': args.flow_hidden,
        'flow_transforms': args.flow_transforms,
        'n_features': n_features,
        'feature_cols': feature_cols,
        'hidden_sizes': args.hidden_sizes,
        'dropout_rate': args.dropout,
        'param_names': ACE_PARAM_NAMES,
        'prior_lower': prior_lower.tolist(),
        'prior_upper': prior_upper.tolist(),
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved config     → {output_dir / 'config.json'}")

    print("\n✓ NPE Training complete!\n")


if __name__ == "__main__":
    main()
