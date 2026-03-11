"""
Predict Parameters from Observed PGS + Phenotypic (Y1) Correlations
(Feature-Specific Regularization Model)

Use trained model to predict simulation parameters from real PGS and phenotypic
correlation data. The model was trained with Feature-Specific Regularization
(Weighted Weight Decay) via train_realistic_pgs_and_pheno_weighted.py.

Observed data are loaded from two separate CSV files:
- observed_correlations_PGS.csv: PGS correlations
- observed_correlations_pheno.csv: Phenotypic (Y1) correlations

Usage:
    python predict_from_observed_pgs_and_pheno_weighted.py \\
        --model_dir results_pgs_and_pheno_weighted \\
        --correlations_pgs observed_correlations_PGS.csv \\
        --correlations_pheno observed_correlations_pheno.csv

Or provide correlations directly as arguments (prefixed with --pgs_ or --pheno_).
"""

import numpy as np
import pandas as pd
import torch
import argparse
from pathlib import Path
import joblib
import json
import sys

from train_realistic_pgs_and_pheno_weighted import FeatureAwarePredictorPgsPheno
from fit_nn import PARAM_NAMES, PARAM_NAMES_EXT


def load_trained_model(model_dir):
    """Load trained model, scalers, and configuration."""
    model_dir = Path(model_dir)

    print(f"\nLoading model from: {model_dir}")

    # Load config
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)

    # Determine parameter names from config (supports both 7 and 9 output models)
    param_names = config.get('param_names', PARAM_NAMES)
    n_outputs = len(param_names)

    # Load scalers
    feature_scaler = joblib.load(model_dir / 'feature_scaler.pkl')
    target_scaler = joblib.load(model_dir / 'target_scaler.pkl')

    # Load polynomial transformer if exists
    poly_transformer = None
    if (model_dir / 'poly_transformer.pkl').exists():
        poly_transformer = joblib.load(model_dir / 'poly_transformer.pkl')
        print(f"✓ Loaded polynomial transformer (degree={config['interaction_degree']})")

    # Create model
    device = torch.device('cpu')  # Use CPU for prediction
    model = FeatureAwarePredictorPgsPheno(
        n_features=config['n_engineered_features'],
        hidden_sizes=config['hidden_sizes'],
        dropout_rate=config['dropout_rate'],
        n_outputs=n_outputs
    )

    # Load weights
    checkpoint = torch.load(model_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded model (trained for {config['epochs_trained']} epochs)")
    print(f"✓ Best validation loss: {checkpoint['val_loss']:.6f}")
    print(f"✓ Model uses {config.get('n_original_pgs1_features', '?')} PGS1 features "
          f"and {config.get('n_original_y1_features', '?')} Y1 features")
    print(f"✓ Predicting {n_outputs} parameters: {param_names}")

    # Print regularization settings if available
    if config.get('regularization') == 'feature_specific':
        print(f"✓ Trained with Feature-Specific Regularization:")
        print(f"    pgs_weight_decay:   {config.get('pgs_weight_decay', '?')}")
        print(f"    pheno_weight_decay: {config.get('pheno_weight_decay', '?')}")

    return model, feature_scaler, target_scaler, poly_transformer, config, device, param_names


def prepare_correlation_input(pgs_correlations_dict, pheno_correlations_dict, expected_features):
    """
    Prepare combined correlation input for prediction.

    Args:
        pgs_correlations_dict: Dict mapping relationship type to PGS correlation value
        pheno_correlations_dict: Dict mapping relationship type to phenotypic correlation value
        expected_features: List of expected feature names from training

    Returns:
        numpy array with correlations in correct order
    """
    X = np.zeros(len(expected_features))

    used_pgs_rels = set()
    used_pheno_rels = set()

    matched_pgs = 0
    matched_pheno = 0
    missing = []

    for i, feature_name in enumerate(expected_features):
        # Handle two naming conventions:
        # 1. "cor_S_PGS1" -> "S" (PGS feature)
        # 2. "cor_S_Y1"   -> "S" (Y1 phenotypic feature)

        if '_PGS1' in feature_name:
            rel_type = feature_name.replace('_PGS1', '')
            if rel_type.startswith('cor_'):
                rel_type = rel_type.replace('cor_', '')

            if rel_type in pgs_correlations_dict:
                X[i] = pgs_correlations_dict[rel_type]
                matched_pgs += 1
                used_pgs_rels.add(rel_type)
            else:
                missing.append(f"PGS:{rel_type}")
                X[i] = 0.0

        elif '_Y1' in feature_name:
            rel_type = feature_name.replace('_Y1', '')
            if rel_type.startswith('cor_'):
                rel_type = rel_type.replace('cor_', '')

            if rel_type in pheno_correlations_dict:
                X[i] = pheno_correlations_dict[rel_type]
                matched_pheno += 1
                used_pheno_rels.add(rel_type)
            else:
                missing.append(f"Y1:{rel_type}")
                X[i] = 0.0

    unused_pgs = set(pgs_correlations_dict.keys()) - used_pgs_rels
    unused_pheno = set(pheno_correlations_dict.keys()) - used_pheno_rels

    total_expected = len(expected_features)
    total_matched = matched_pgs + matched_pheno

    print(f"\n{'='*70}")
    print("CORRELATION MATCHING SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Matched {total_matched}/{total_expected} features")
    print(f"  - PGS features matched: {matched_pgs}")
    print(f"  - Phenotypic features matched: {matched_pheno}")

    if missing:
        print(f"\n⚠ WARNING: Model expects {len(missing)} features not provided (filled with 0):")
        for m in missing[:10]:
            print(f"    • {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing)-10} more")

    if unused_pgs or unused_pheno:
        print(f"\n⚠ WARNING: {len(unused_pgs) + len(unused_pheno)} correlations provided but NOT used by model:")
        for rel in sorted(unused_pgs):
            print(f"    • PGS:{rel} (not in training features)")
        for rel in sorted(unused_pheno):
            print(f"    • Y1:{rel} (not in training features)")

    print(f"{'='*70}\n")

    return X.reshape(1, -1)


def predict_parameters(model, feature_scaler, target_scaler, poly_transformer,
                       X, device, config=None):
    """Make prediction and inverse transform to original scale."""

    # Apply polynomial transformation if needed
    if poly_transformer is not None and config is not None:
        interaction_degree = config.get('interaction_degree', 2)
        if interaction_degree > 1:
            X = poly_transformer.transform(X)
            print(f"  Applied polynomial features: {X.shape[1]} features")

    X_normalized = feature_scaler.transform(X)
    X_tensor = torch.FloatTensor(X_normalized).to(device)

    with torch.no_grad():
        predictions_normalized = model(X_tensor).cpu().numpy()

    predictions = target_scaler.inverse_transform(predictions_normalized)

    return predictions[0]


def display_predictions(predictions, param_names):
    """Display predicted parameters in a nice format.

    Args:
        predictions: array of predicted values
        param_names: list of parameter names matching the model outputs
    """
    print("\n" + "="*70)
    print("PREDICTED PARAMETERS (PGS + Phenotypic Model, Feature-Specific Regularization)")
    print("="*70)

    print(f"\n{'Parameter':<20} {'Predicted Value':<20} {'Description'}")
    print("-" * 70)

    descriptions = {
        'f11': 'Vertical transmission (trait 1)',
        'f22': 'Vertical transmission (trait 2)',
        'f12': 'Cross-trait VT (trait 1 -> trait 2)',
        'f21': 'Cross-trait VT (trait 2 -> trait 1)',
        'prop_h2_latent1': 'Prop. h² latent (trait 1)',
        'vg1': 'Genetic variance (trait 1)',
        'vg2': 'Genetic variance (trait 2)',
        'am22': 'Assortative mating coef.',
        'rg': 'Genetic correlation'
    }

    for param, value in zip(param_names, predictions):
        desc = descriptions.get(param, '')
        print(f"{param:<20} {value:<20.4f} {desc}")

    print("="*70)

    pred_df = pd.DataFrame([predictions], columns=param_names)

    return pred_df


def main():
    parser = argparse.ArgumentParser(
        description='Predict parameters from observed PGS + phenotypic correlations '
                    '(Feature-Specific Regularization model)'
    )
    parser.add_argument('--model_dir', type=str, default='results_pgs_and_pheno_weighted',
                       help='Directory containing trained model (absolute or relative to script dir)')
    parser.add_argument('--correlations_pgs', type=str, default=None,
                       help='Path to CSV file with PGS correlations (columns: RelType, Correlation)')
    parser.add_argument('--correlations_pheno', type=str, default=None,
                       help='Path to CSV file with phenotypic (Y1) correlations (columns: RelType, Correlation)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for predictions (default: predictions_observed.csv in model_dir)')

    # Alternative: provide PGS correlations as command-line arguments
    parser.add_argument('--pgs_S', type=float, help='PGS Siblings correlation')
    parser.add_argument('--pgs_HSFS', type=float, help='PGS Half-siblings correlation')
    parser.add_argument('--pgs_PSC', type=float, help='PGS Avuncular correlation')
    parser.add_argument('--pgs_PPSCC', type=float, help='PGS First cousins correlation')
    parser.add_argument('--pgs_M', type=float, help='PGS Mates correlation')
    parser.add_argument('--pgs_MS', type=float, help='PGS Siblings-in-law correlation')
    parser.add_argument('--pgs_SMS', type=float, help='PGS Sibling\'s mate\'s sibling')
    parser.add_argument('--pgs_MSC', type=float, help='PGS Mate\'s sibling\'s child')
    parser.add_argument('--pgs_MSM', type=float, help='PGS Mate\'s sibling\'s mate')
    parser.add_argument('--pgs_SMSC', type=float, help='PGS Sibling\'s mate\'s sibling\'s child')
    parser.add_argument('--pgs_SMSM', type=float, help='PGS Sibling\'s mate\'s sibling\'s mate')
    parser.add_argument('--pgs_SMSMS', type=float, help='PGS SMSMS correlation')
    parser.add_argument('--pgs_MSMSM', type=float, help='PGS MSMSM correlation')
    parser.add_argument('--pgs_MSMSC', type=float, help='PGS MSMSC correlation')
    parser.add_argument('--pgs_PSMSC', type=float, help='PGS PSMSC correlation')
    parser.add_argument('--pgs_SMSMSC', type=float, help='PGS SMSMSC correlation')
    parser.add_argument('--pgs_MSMSMS', type=float, help='PGS MSMSMS correlation')

    # Alternative: provide phenotypic correlations as command-line arguments
    parser.add_argument('--pheno_S', type=float, help='Phenotypic Siblings correlation')
    parser.add_argument('--pheno_HSFS', type=float, help='Phenotypic Half-siblings correlation')
    parser.add_argument('--pheno_PSC', type=float, help='Phenotypic Avuncular correlation')
    parser.add_argument('--pheno_PPSCC', type=float, help='Phenotypic First cousins correlation')
    parser.add_argument('--pheno_M', type=float, help='Phenotypic Mates correlation')
    parser.add_argument('--pheno_MS', type=float, help='Phenotypic Siblings-in-law correlation')
    parser.add_argument('--pheno_SMS', type=float, help='Phenotypic Sibling\'s mate\'s sibling')
    parser.add_argument('--pheno_MSC', type=float, help='Phenotypic Mate\'s sibling\'s child')
    parser.add_argument('--pheno_MSM', type=float, help='Phenotypic Mate\'s sibling\'s mate')
    parser.add_argument('--pheno_SMSC', type=float, help='Phenotypic Sibling\'s mate\'s sibling\'s child')
    parser.add_argument('--pheno_SMSM', type=float, help='Phenotypic Sibling\'s mate\'s sibling\'s mate')
    parser.add_argument('--pheno_SMSMS', type=float, help='Phenotypic SMSMS correlation')
    parser.add_argument('--pheno_MSMSM', type=float, help='Phenotypic MSMSM correlation')
    parser.add_argument('--pheno_MSMSC', type=float, help='Phenotypic MSMSC correlation')
    parser.add_argument('--pheno_PSMSC', type=float, help='Phenotypic PSMSC correlation')
    parser.add_argument('--pheno_SMSMSC', type=float, help='Phenotypic SMSMSC correlation')
    parser.add_argument('--pheno_MSMSMS', type=float, help='Phenotypic MSMSMS correlation')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("PARAMETER PREDICTION — PGS + PHENOTYPIC (FEATURE-SPECIFIC REGULARIZATION)")
    print("="*70)

    # Resolve model directory path relative to script location if not absolute
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        script_dir = Path(__file__).parent
        model_dir = script_dir / args.model_dir

    if not model_dir.exists():
        print(f"\nError: Model directory not found: {model_dir}")
        print(f"Script directory: {Path(__file__).parent}")
        print(f"Current directory: {Path.cwd()}")
        return

    # Load model
    model, feature_scaler, target_scaler, poly_transformer, config, device, param_names = \
        load_trained_model(model_dir)

    # Get PGS correlations
    pgs_correlations_dict = {}
    pheno_correlations_dict = {}

    if args.correlations_pgs:
        cors_path = Path(args.correlations_pgs)
        if not cors_path.is_absolute():
            script_dir = Path(__file__).parent
            if (script_dir / args.correlations_pgs).exists():
                cors_path = script_dir / args.correlations_pgs

        print(f"\nLoading PGS correlations from: {cors_path}")
        cors_df = pd.read_csv(cors_path)

        if 'RelType' in cors_df.columns and 'Correlation' in cors_df.columns:
            cors_df = cors_df.groupby('RelType')['Correlation'].mean().reset_index()
            pgs_correlations_dict = dict(zip(cors_df['RelType'], cors_df['Correlation']))
        else:
            print("Error: PGS CSV must have 'RelType' and 'Correlation' columns")
            return
    else:
        rel_types = ['S', 'HSFS', 'PSC', 'PPSCC', 'M', 'MS', 'SMS', 'MSC', 'MSM',
                     'SMSC', 'SMSM', 'SMSMS', 'MSMSM', 'MSMSC', 'PSMSC', 'SMSMSC', 'MSMSMS']
        for rel_type in rel_types:
            value = getattr(args, f'pgs_{rel_type}', None)
            if value is not None:
                pgs_correlations_dict[rel_type] = value

    if args.correlations_pheno:
        cors_path = Path(args.correlations_pheno)
        if not cors_path.is_absolute():
            script_dir = Path(__file__).parent
            if (script_dir / args.correlations_pheno).exists():
                cors_path = script_dir / args.correlations_pheno

        print(f"\nLoading phenotypic (Y1) correlations from: {cors_path}")
        cors_df = pd.read_csv(cors_path)

        if 'RelType' in cors_df.columns and 'Correlation' in cors_df.columns:
            cors_df = cors_df.groupby('RelType')['Correlation'].mean().reset_index()
            pheno_correlations_dict = dict(zip(cors_df['RelType'], cors_df['Correlation']))
        else:
            print("Error: Phenotypic CSV must have 'RelType' and 'Correlation' columns")
            return
    else:
        rel_types = ['S', 'HSFS', 'PSC', 'PPSCC', 'M', 'MS', 'SMS', 'MSC', 'MSM',
                     'SMSC', 'SMSM', 'SMSMS', 'MSMSM', 'MSMSC', 'PSMSC', 'SMSMSC', 'MSMSMS']
        for rel_type in rel_types:
            value = getattr(args, f'pheno_{rel_type}', None)
            if value is not None:
                pheno_correlations_dict[rel_type] = value

    if not pgs_correlations_dict and not pheno_correlations_dict:
        print("\nError: No correlations provided!")
        print("Provide PGS correlations via --correlations_pgs <file> or --pgs_<REL> <value>")
        print("Provide phenotypic correlations via --correlations_pheno <file> or --pheno_<REL> <value>")
        print("\nExample:")
        print("  python predict_from_observed_pgs_and_pheno_weighted.py \\")
        print("      --correlations_pgs observed_correlations_PGS.csv \\")
        print("      --correlations_pheno observed_correlations_pheno.csv")
        return

    print("\nInput PGS correlations:")
    for rel_type, corr in sorted(pgs_correlations_dict.items()):
        print(f"  PGS  {rel_type:<10} {corr:.6f}")

    print("\nInput phenotypic (Y1) correlations:")
    for rel_type, corr in sorted(pheno_correlations_dict.items()):
        print(f"  Y1   {rel_type:<10} {corr:.6f}")

    # Prepare input
    X = prepare_correlation_input(
        pgs_correlations_dict, pheno_correlations_dict, config['original_features']
    )

    # Make prediction
    print("\nMaking prediction...")
    predictions = predict_parameters(
        model, feature_scaler, target_scaler, poly_transformer, X, device, config
    )

    # Display results
    pred_df = display_predictions(predictions, param_names)

    # Save predictions
    if args.output is None:
        output_path = Path(args.model_dir) / 'predictions_observed.csv'
    else:
        output_path = Path(args.output)

    pred_df.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}\n")

    return pred_df


if __name__ == "__main__":
    main()
