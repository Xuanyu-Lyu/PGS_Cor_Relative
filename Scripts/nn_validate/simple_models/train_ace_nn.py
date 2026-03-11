"""
Train Neural Network for ACE Parameter Prediction

Trains a lightweight MLP to predict A, C, E variance components
from MZ and DZ twin covariance matrix elements.

Input features  (4):  mz_var, mz_cov, dz_var, dz_cov
                      Optionally include N_pairs with --include_n_pairs
Output targets  (3):  A (additive genetic), C (shared env), E (unique env)

The network is intentionally lean (~few hundred parameters) since
the prediction problem has a near-analytical solution; a small
NN learns it quickly from simulated data.

Usage:
    # 1. Generate training data first
    python generate_ace_data.py --n_samples 20000

    # 2. Train
    python train_ace_nn.py --data ace_training_data.csv --epochs 500

    # 3. Optionally add N_pairs as a feature
    python train_ace_nn.py --data ace_training_data.csv --include_n_pairs
"""

import sys
import math
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Parameter names (targets to predict)
ACE_PARAM_NAMES = ['A', 'C', 'E']


# ============================================================================
# DATASET
# ============================================================================

class TwinCovDataset(Dataset):
    """PyTorch Dataset for twin covariance-to-ACE parameter prediction."""

    def __init__(self, features, targets):
        """
        Args:
            features: numpy array of shape (n_samples, n_features)
            targets:  numpy array of shape (n_samples, n_targets)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_clean_data(data_path):
    """
    Load data and perform initial cleaning.

    Args:
        data_path: Path to the CSV file

    Returns:
        df: Cleaned DataFrame
    """
    print("\n" + "="*70)
    print("LOADING AND CLEANING DATA")
    print("="*70)

    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} columns")

    initial_samples = len(df)

    # Remove rows with any NaN in feature or target columns
    feature_cols = ['mz_var', 'mz_cov', 'dz_var', 'dz_cov']
    target_cols  = ACE_PARAM_NAMES
    required_cols = feature_cols + target_cols

    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Required columns missing from data: {missing_required}")

    df = df.dropna(subset=required_cols).copy()
    print(f"✓ Removed {initial_samples - len(df)} rows with missing values")
    print(f"  Remaining: {len(df)} samples")

    # Remove rows where covariance matrices look invalid (|r| > 1 implied by |cov| > var)
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
# MODEL
# ============================================================================

class ACEPredictor(nn.Module):
    """
    Lightweight MLP for predicting A, C, E from MZ/DZ covariance matrices.

    Default hidden_sizes=[64, 64, 32] gives ~6 k parameters — much
    lighter than the PGS/phenotypic models.  The problem nearly has a
    closed-form solution so a small network suffices.
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
        layers.append(nn.Linear(in_size, len(ACE_PARAM_NAMES)))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, epochs, learning_rate,
                weight_decay, device, output_dir):
    """Train model with best practices."""

    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7
    )

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 50

    print(f"\nStarting training...")
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Best Val':<15}")
    print("-" * 70)

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                val_loss += criterion(model(features), targets).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {'model_state_dict': model.state_dict(),
                 'val_loss': val_loss,
                 'epoch': epoch},
                output_dir / 'best_model.pt'
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"{epoch+1:<8} {train_loss:<15.6f} {val_loss:<15.6f} {best_val_loss:<15.6f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print(f"\n{'='*70}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"{'='*70}\n")

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training', linewidth=2)
    plt.plot(val_losses, label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History (ACE Model)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'training_history.png', dpi=300)
    plt.close()

    return best_val_loss


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_loader, target_scaler, device, output_dir,
                   param_names=None):
    """
    Evaluate the model on the test set and compute metrics.

    Args:
        param_names: list of parameter names matching model outputs.
                     Defaults to ACE_PARAM_NAMES if not provided.
    """
    param_names = param_names or ACE_PARAM_NAMES

    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST SET")
    print("="*70)

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            all_preds.append(model(features).cpu().numpy())
            all_targets.append(targets.numpy())

    predictions = np.vstack(all_preds)
    targets     = np.vstack(all_targets)

    # Inverse-transform to original scale
    predictions = target_scaler.inverse_transform(predictions)
    targets     = target_scaler.inverse_transform(targets)

    results = {}
    print(f"\n{'Parameter':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
    print("-" * 70)

    for i, name in enumerate(param_names):
        y_true = targets[:, i]
        y_pred = predictions[:, i]
        r2   = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        results[name] = {'r2': r2, 'rmse': rmse, 'mae': mae,
                         'true': y_true, 'pred': y_pred}
        print(f"{name:<20} {r2:<10.4f} {rmse:<10.4f} {mae:<10.4f}")

    overall_r2 = r2_score(targets.flatten(), predictions.flatten())
    print("-" * 70)
    print(f"{'OVERALL':<20} {overall_r2:<10.4f}")
    print("="*70 + "\n")

    # Save metrics JSON
    metrics_dict = {
        name: {k: float(v) if k not in ('true', 'pred') else v.tolist()
               for k, v in vals.items()}
        for name, vals in results.items()
    }
    metrics_dict['overall'] = {'r2': float(overall_r2)}
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_predictions(results, output_dir, param_names=None):
    """Plot predicted vs true values for each parameter.

    Args:
        param_names: list of parameter names. Defaults to ACE_PARAM_NAMES if not provided.
    """
    param_names = param_names or ACE_PARAM_NAMES
    output_dir  = Path(output_dir)

    n_params = len(param_names)
    n_cols   = min(n_params, 4)
    n_rows   = math.ceil(n_params / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    for i, name in enumerate(param_names):
        ax     = axes[i]
        y_true = results[name]['true']
        y_pred = results[name]['pred']
        r2     = results[name]['r2']

        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect prediction')
        ax.set_xlabel('True Value', fontsize=10)
        ax.set_ylabel('Predicted Value', fontsize=10)
        ax.set_title(f'{name} (R² = {r2:.3f})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n_params, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_vs_true.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved predictions vs true values plot")

    # Individual high-res plots
    for name in param_names:
        fig, ax = plt.subplots(figsize=(8, 8))
        y_true = results[name]['true']
        y_pred = results[name]['pred']
        r2     = results[name]['r2']
        rmse   = results[name]['rmse']

        ax.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', lw=3, label='Perfect prediction')
        ax.set_xlabel('True Value', fontsize=14)
        ax.set_ylabel('Predicted Value', fontsize=14)
        ax.set_title(f'{name}', fontsize=16, fontweight='bold')
        ax.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'prediction_{name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✓ Saved individual parameter plots")


def plot_residuals(results, output_dir, param_names=None):
    """Plot residuals for each parameter.

    Args:
        param_names: list of parameter names. Defaults to ACE_PARAM_NAMES if not provided.
    """
    param_names = param_names or ACE_PARAM_NAMES
    output_dir  = Path(output_dir)

    n_params = len(param_names)
    n_cols   = min(n_params, 4)
    n_rows   = math.ceil(n_params / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    for i, name in enumerate(param_names):
        ax        = axes[i]
        y_pred    = results[name]['pred']
        residuals = results[name]['true'] - y_pred

        ax.scatter(y_pred, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Value', fontsize=10)
        ax.set_ylabel('Residual', fontsize=10)
        ax.set_title(f'{name}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    for j in range(n_params, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_dir / 'residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved residuals plot")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train ACE neural network')
    parser.add_argument('--data', type=str, default='ace_training_data.csv',
                        help='Path to training CSV (default: ace_training_data.csv)')
    parser.add_argument('--output', type=str, default='results_ace',
                        help='Output directory (default: results_ace)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Max training epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='AdamW weight decay (default: 1e-4)')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[64, 64, 32],
                        help='Hidden layer sizes (default: 64 64 32)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--include_n_pairs', action='store_true',
                        help='Include N_pairs as an input feature')
    args = parser.parse_args()

    # Resolve paths relative to this script's directory
    script_dir = Path(__file__).parent
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = script_dir / data_path
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("ACE NEURAL NETWORK TRAINING")
    print("="*70)
    print(f"Output: {output_dir}")
    print("="*70)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"\nUsing device: {device}")

    # ---- Load data ----
    df = load_and_clean_data(data_path)

    feature_cols = ['mz_var', 'mz_cov', 'dz_var', 'dz_cov']
    if args.include_n_pairs and 'N_pairs' in df.columns:
        feature_cols.append('N_pairs')
        print("  Including N_pairs as feature")
    target_cols = ACE_PARAM_NAMES

    X = df[feature_cols].values
    y = df[target_cols].values

    # ---- Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42)
    print(f"\nSplit  —  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ---- Scale ----
    feature_scaler = StandardScaler()
    target_scaler  = StandardScaler()

    X_train_s = feature_scaler.fit_transform(X_train)
    X_val_s   = feature_scaler.transform(X_val)
    X_test_s  = feature_scaler.transform(X_test)
    y_train_s = target_scaler.fit_transform(y_train)
    y_val_s   = target_scaler.transform(y_val)
    y_test_s  = target_scaler.transform(y_test)

    # ---- DataLoaders ----
    train_loader = DataLoader(TwinCovDataset(X_train_s, y_train_s),
                              batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TwinCovDataset(X_val_s, y_val_s),
                              batch_size=args.batch_size)
    test_loader  = DataLoader(TwinCovDataset(X_test_s, y_test_s),
                              batch_size=args.batch_size)

    # ---- Model ----
    model = ACEPredictor(
        n_features=len(feature_cols),
        hidden_sizes=args.hidden_sizes,
        dropout_rate=args.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    arch_str = ' -> '.join([str(len(feature_cols))] +
                           [str(h) for h in args.hidden_sizes] +
                           [str(len(ACE_PARAM_NAMES))])
    print(f"\nModel: {n_params} parameters")
    print(f"Architecture: {arch_str}")

    # ---- Save config and scalers ----
    config = {
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'hidden_sizes': args.hidden_sizes,
        'dropout_rate': args.dropout,
        'n_features': len(feature_cols),
        'param_names': target_cols,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    joblib.dump(feature_scaler, output_dir / 'feature_scaler.pkl')
    joblib.dump(target_scaler,  output_dir / 'target_scaler.pkl')

    # ---- Train ----
    train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, learning_rate=args.lr, weight_decay=args.weight_decay,
        device=device, output_dir=output_dir
    )

    # ---- Load best checkpoint ----
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model — epoch {checkpoint['epoch']+1}, "
          f"val_loss = {checkpoint['val_loss']:.6f}")

    # Update config with training info
    config['epochs_trained'] = int(checkpoint['epoch']) + 1
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ---- Evaluate ----
    results = evaluate_model(model, test_loader, target_scaler, device, output_dir)
    plot_predictions(results, output_dir)
    plot_residuals(results, output_dir)


if __name__ == "__main__":
    main()
