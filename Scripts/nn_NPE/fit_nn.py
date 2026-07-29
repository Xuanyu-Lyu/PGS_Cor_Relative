"""
Shared Utilities for Neural Network Parameter Prediction

This module provides shared constants, dataset class, evaluation functions,
and visualization utilities used by the training scripts:
  - train_realistic_pgs_only.py       (PGS-only model)
  - train_realistic_pgs_and_pheno.py  (PGS + phenotypic model)

Shared exports:
  - PARAM_NAMES:        Target parameter names
  - CorrelationDataset: PyTorch Dataset class
  - evaluate_model:     Model evaluation on test set
  - plot_predictions:   Prediction vs true scatter plots
  - plot_residuals:     Residual diagnostic plots
"""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Parameter names (targets to predict)
PARAM_NAMES = ['f11', 'prop_h2_latent1', 'vg1', 'vg2', 'f22', 'am22', 'rg']
PARAM_NAMES_EXT = ['f11','f22', 'f12', 'f21', 'prop_h2_latent1', 'vg1', 'vg2',  'am22', 'rg']
# ============================================================================
# DATASET CLASS
# ============================================================================

class CorrelationDataset(Dataset):
    """
    PyTorch Dataset for correlation-to-parameter prediction.
    """
    def __init__(self, features, targets):
        """
        Args:
            features: numpy array of shape (n_samples, n_features)
            targets: numpy array of shape (n_samples, n_targets)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_loader, target_scaler, device, output_dir, param_names=None):
    """
    Evaluate the model on test set and compute metrics.
    
    Args:
        param_names: list of parameter names matching model outputs.
                     Defaults to PARAM_NAMES if not provided.
    """
    param_names = param_names or PARAM_NAMES

    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST SET")
    print("="*70)
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            outputs = model(features)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
    
    # Concatenate all batches
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Inverse transform to original scale
    predictions = target_scaler.inverse_transform(predictions)
    targets = target_scaler.inverse_transform(targets)
    
    # Compute metrics for each parameter
    results = {}
    print(f"\n{'Parameter':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
    print("-" * 70)
    
    for i, param_name in enumerate(param_names):
        y_true = targets[:, i]
        y_pred = predictions[:, i]
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        results[param_name] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'true': y_true,
            'pred': y_pred
        }
        
        print(f"{param_name:<20} {r2:<10.4f} {rmse:<10.4f} {mae:<10.4f}")
    
    # Overall metrics
    overall_r2 = r2_score(targets.flatten(), predictions.flatten())
    overall_rmse = np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten()))
    print("-" * 70)
    print(f"{'OVERALL':<20} {overall_r2:<10.4f} {overall_rmse:<10.4f}")
    print("="*70 + "\n")
    
    # Save metrics
    output_dir = Path(output_dir)
    metrics_dict = {
        param: {k: float(v) if k not in ['true', 'pred'] else v.tolist() 
                for k, v in vals.items()}
        for param, vals in results.items()
    }
    metrics_dict['overall'] = {'r2': float(overall_r2), 'rmse': float(overall_rmse)}
    
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_predictions(results, output_dir, param_names=None):
    """Plot predicted vs true values for each parameter.
    
    Args:
        param_names: list of parameter names. Defaults to PARAM_NAMES if not provided.
    """
    param_names = param_names or PARAM_NAMES
    output_dir = Path(output_dir)

    # Overview plot (all parameters) — dynamic grid
    n_params = len(param_names)
    n_cols = 4
    n_rows = math.ceil(n_params / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    
    for i, param_name in enumerate(param_names):
        ax = axes[i]
        y_true = results[param_name]['true']
        y_pred = results[param_name]['pred']
        r2 = results[param_name]['r2']
        
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        ax.set_xlabel('True Value', fontsize=10)
        ax.set_ylabel('Predicted Value', fontsize=10)
        ax.set_title(f'{param_name} (R² = {r2:.3f})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide any unused subplot slots
    for j in range(n_params, n_rows * n_cols):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_vs_true.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved predictions vs true values plot")
    
    # Individual high-res plots
    for param_name in param_names:
        fig, ax = plt.subplots(figsize=(8, 8))
        y_true = results[param_name]['true']
        y_pred = results[param_name]['pred']
        r2 = results[param_name]['r2']
        rmse = results[param_name]['rmse']
        
        ax.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect prediction')
        
        ax.set_xlabel('True Value', fontsize=14)
        ax.set_ylabel('Predicted Value', fontsize=14)
        ax.set_title(f'{param_name}', fontsize=16, fontweight='bold')
        ax.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'prediction_{param_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved individual parameter plots")

def plot_residuals(results, output_dir, param_names=None):
    """Plot residuals for each parameter.
    
    Args:
        param_names: list of parameter names. Defaults to PARAM_NAMES if not provided.
    """
    param_names = param_names or PARAM_NAMES
    output_dir = Path(output_dir)

    n_params = len(param_names)
    n_cols = 4
    n_rows = math.ceil(n_params / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    
    for i, param_name in enumerate(param_names):
        ax = axes[i]
        y_true = results[param_name]['true']
        y_pred = results[param_name]['pred']
        residuals = y_true - y_pred
        
        ax.scatter(y_pred, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Value', fontsize=10)
        ax.set_ylabel('Residual', fontsize=10)
        ax.set_title(f'{param_name}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    for j in range(n_params, n_rows * n_cols):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved residuals plot")
