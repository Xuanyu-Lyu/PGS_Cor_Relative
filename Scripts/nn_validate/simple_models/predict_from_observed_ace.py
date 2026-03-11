"""
Out-of-Sample Validation for the ACE Neural Network

Generates N fresh samples (default 200) that were NOT part of the training data,
runs the trained model on them, and checks whether predictions are unbiased by
comparing predicted vs true A, C, E values.

Usage:
    python predict_from_observed_ace.py
    python predict_from_observed_ace.py --n_samples 500 --model_dir results_ace
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from train_ace_nn import ACEPredictor, ACE_PARAM_NAMES
from generate_ace_data import generate_training_data


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_trained_model(model_dir):
    """Load trained ACE model, scalers, and configuration."""
    model_dir = Path(model_dir)

    print(f"\nLoading model from: {model_dir}")

    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)

    param_names = config.get('param_names', ACE_PARAM_NAMES)

    feature_scaler = joblib.load(model_dir / 'feature_scaler.pkl')
    target_scaler  = joblib.load(model_dir / 'target_scaler.pkl')

    device = torch.device('cpu')
    model  = ACEPredictor(
        n_features=config['n_features'],
        hidden_sizes=config['hidden_sizes'],
        dropout_rate=config['dropout_rate'],
    )

    checkpoint = torch.load(model_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epochs_trained = config.get('epochs_trained', checkpoint.get('epoch', '?'))
    print(f"✓ Loaded model (trained for {epochs_trained} epochs)")
    print(f"✓ Best validation loss: {checkpoint['val_loss']:.6f}")
    print(f"✓ Input features: {config['feature_cols']}")
    print(f"✓ Predicting {len(param_names)} parameters: {param_names}")

    return model, feature_scaler, target_scaler, config, device, param_names


# ============================================================================
# OUT-OF-SAMPLE VALIDATION
# ============================================================================

def run_oos_validation(n_samples=200, model_dir='results_ace', seed=999, output_dir=None):
    """
    Generate n_samples fresh ACE samples (distinct seed from training data),
    predict A, C, E with the trained model, and evaluate bias.

    Args:
        n_samples:   Number of out-of-sample predictions to make
        model_dir:   Path to trained model directory
        seed:        Random seed (use a value different from training seed=42)
        output_dir:  Where to save results; defaults to model_dir

    Returns:
        pd.DataFrame with columns [A_true, C_true, E_true, A_pred, C_pred, E_pred]
    """
    script_dir = Path(__file__).parent
    model_dir  = Path(model_dir)
    if not model_dir.is_absolute():
        model_dir = script_dir / model_dir
    if output_dir is None:
        output_dir = model_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("OUT-OF-SAMPLE VALIDATION (ACE MODEL)")
    print("="*70)
    print(f"  Samples:    {n_samples}")
    print(f"  Seed:       {seed}  (different from training seed)")
    print(f"  Model dir:  {model_dir}")

    # Load model
    model, feature_scaler, target_scaler, config, device, param_names = \
        load_trained_model(model_dir)

    # Generate fresh out-of-sample data
    print(f"\nGenerating {n_samples} out-of-sample observations...")
    df_oos = generate_training_data(n_samples=n_samples, seed=seed)

    feature_cols = config['feature_cols']
    X_oos = df_oos[feature_cols].values
    y_true = df_oos[param_names].values  # shape (n_samples, 3)

    # Predict
    X_scaled = feature_scaler.transform(X_oos)
    X_tensor  = torch.FloatTensor(X_scaled).to(device)

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()

    y_pred = target_scaler.inverse_transform(y_pred_scaled)  # shape (n_samples, 3)

    # ---- Bias summary ----
    print("\n" + "="*70)
    print("BIAS ANALYSIS")
    print("="*70)
    print(f"\n{'Parameter':<12} {'Mean True':<14} {'Mean Pred':<14} "
          f"{'Bias':<12} {'|Bias|/SD':<12} {'R²':<8} {'RMSE':<8}")
    print("-" * 80)

    results_rows = []
    for i, name in enumerate(param_names):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        bias   = np.mean(pred_i - true_i)
        sd     = np.std(true_i)
        rel_bias = abs(bias) / sd if sd > 0 else np.nan
        r2     = r2_score(true_i, pred_i)
        rmse   = np.sqrt(mean_squared_error(true_i, pred_i))
        mae    = mean_absolute_error(true_i, pred_i)

        print(f"{name:<12} {np.mean(true_i):<14.4f} {np.mean(pred_i):<14.4f} "
              f"{bias:<12.4f} {rel_bias:<12.4f} {r2:<8.4f} {rmse:<8.4f}")

        results_rows.append({
            'param':      name,
            'mean_true':  float(np.mean(true_i)),
            'mean_pred':  float(np.mean(pred_i)),
            'bias':       float(bias),
            'rel_bias':   float(rel_bias),
            'r2':         float(r2),
            'rmse':       float(rmse),
            'mae':        float(mae),
        })

    print("="*70)
    print("  bias      = mean(predicted) - mean(true)")
    print("  |bias|/SD = bias relative to SD of true values (< 0.1 indicates low bias)")

    # ---- Save detailed predictions ----
    df_out = pd.DataFrame(
        np.hstack([y_true, y_pred]),
        columns=[f"{p}_true" for p in param_names] + [f"{p}_pred" for p in param_names]
    )
    # Also record the sum of predictions (should be close to 1)
    df_out['sum_pred'] = df_out[[f"{p}_pred" for p in param_names]].sum(axis=1)
    df_out['sum_true'] = df_out[[f"{p}_true" for p in param_names]].sum(axis=1)

    predictions_path = output_dir / 'oos_predictions.csv'
    df_out.to_csv(predictions_path, index=False)
    print(f"\n✓ Detailed predictions saved to: {predictions_path}")

    # ---- Save bias summary ----
    summary_df = pd.DataFrame(results_rows)
    summary_path = output_dir / 'oos_bias_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Bias summary saved to: {summary_path}")

    # ---- Plots ----
    _plot_oos(y_true, y_pred, param_names, output_dir)

    return df_out


def _plot_oos(y_true, y_pred, param_names, output_dir):
    """Scatter plots (predicted vs true) and bias distribution plots."""
    n = len(param_names)

    # --- Predicted vs True ---
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    axes = np.array(axes).flatten()

    for i, name in enumerate(param_names):
        ax     = axes[i]
        t, p   = y_true[:, i], y_pred[:, i]
        r2     = r2_score(t, p)
        bias   = np.mean(p - t)

        ax.scatter(t, p, alpha=0.5, s=30, edgecolors='k', linewidth=0.3)
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect prediction')
        ax.set_xlabel('True Value', fontsize=12)
        ax.set_ylabel('Predicted Value', fontsize=12)
        ax.set_title(f'{name}  (R² = {r2:.3f},  bias = {bias:+.4f})',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('ACE Model — Out-of-Sample Predictions vs True',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'oos_predictions_vs_true.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved predictions vs true plot")

    # --- Bias distribution (prediction error histogram) ---
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    axes = np.array(axes).flatten()

    for i, name in enumerate(param_names):
        ax    = axes[i]
        error = y_pred[:, i] - y_true[:, i]
        ax.hist(error, bins=30, edgecolor='k', alpha=0.7)
        ax.axvline(0,                color='r',      lw=2, linestyle='--', label='Zero bias')
        ax.axvline(np.mean(error),   color='orange', lw=2, linestyle='-',
                   label=f'Mean bias = {np.mean(error):+.4f}')
        ax.set_xlabel('Predicted − True', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('ACE Model — Prediction Error Distribution (Out-of-Sample)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'oos_bias_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved bias distribution plot")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Out-of-sample bias validation for the ACE neural network'
    )
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Number of out-of-sample predictions (default: 200)')
    parser.add_argument('--model_dir', type=str, default='results_ace',
                        help='Directory containing trained model (default: results_ace)')
    parser.add_argument('--seed', type=int, default=999,
                        help='Random seed for OOS data generation (default: 999)')
    args = parser.parse_args()

    run_oos_validation(
        n_samples=args.n_samples,
        model_dir=args.model_dir,
        seed=args.seed,
    )
