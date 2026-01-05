"""
Predict Parameters from Observed PGS Correlations

Use trained model to predict simulation parameters from real PGS correlation data.

Usage:
    python predict_from_observed.py --model_dir results_realistic_pgs --correlations observed_cors.csv
    
Or provide correlations directly as arguments.
"""

import numpy as np
import pandas as pd
import torch
import argparse
from pathlib import Path
import joblib
import json
import sys

from train_realistic_pgs_only import FeatureAwarePredictor
from fit_nn import PARAM_NAMES

def load_trained_model(model_dir):
    """Load trained model, scalers, and configuration."""
    model_dir = Path(model_dir)
    
    print(f"\nLoading model from: {model_dir}")
    
    # Load config
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
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
    model = FeatureAwarePredictor(
        n_features=config['n_engineered_features'],
        hidden_sizes=config['hidden_sizes'],
        dropout_rate=config['dropout_rate']
    )
    
    # Load weights
    checkpoint = torch.load(model_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model (trained for {config['epochs_trained']} epochs)")
    print(f"✓ Best validation loss: {checkpoint['val_loss']:.6f}")
    
    return model, feature_scaler, target_scaler, poly_transformer, config, device

def prepare_correlation_input(correlations_dict, expected_features):
    """
    Prepare correlation input for prediction.
    
    Args:
        correlations_dict: Dict mapping relationship type to correlation value
        expected_features: List of expected feature names from training
        
    Returns:
        numpy array with correlations in correct order
    """
    # Create feature vector
    X = np.zeros(len(expected_features))
    
    # Map correlations to feature positions
    matched = 0
    missing = []
    
    for i, feature_name in enumerate(expected_features):
        # Extract relationship type from feature name
        # Feature names are like "cor_S_PGS1"
        if feature_name.startswith('cor_') and '_PGS1' in feature_name:
            rel_type = feature_name.replace('cor_', '').replace('_PGS1', '')
            
            if rel_type in correlations_dict:
                X[i] = correlations_dict[rel_type]
                matched += 1
            else:
                missing.append(rel_type)
                X[i] = 0.0  # Fill missing with 0
    
    print(f"\n✓ Matched {matched}/{len(expected_features)} features")
    if missing:
        print(f"⚠ Missing correlations for: {', '.join(missing[:10])}")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more")
    
    return X.reshape(1, -1)  # Reshape to (1, n_features) for single prediction

def predict_parameters(model, feature_scaler, target_scaler, poly_transformer, 
                      X, device):
    """Make prediction and inverse transform to original scale."""
    
    # Apply polynomial transformation if needed
    if poly_transformer is not None:
        X = poly_transformer.transform(X)
    
    # Normalize features
    X_normalized = feature_scaler.transform(X)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_normalized).to(device)
    
    # Predict
    with torch.no_grad():
        predictions_normalized = model(X_tensor).cpu().numpy()
    
    # Inverse transform to original scale
    predictions = target_scaler.inverse_transform(predictions_normalized)
    
    return predictions[0]  # Return single prediction

def display_predictions(predictions, confidence=False):
    """Display predicted parameters in a nice format."""
    
    print("\n" + "="*70)
    print("PREDICTED PARAMETERS")
    print("="*70)
    
    print(f"\n{'Parameter':<20} {'Predicted Value':<20} {'Description'}")
    print("-" * 70)
    
    descriptions = {
        'f11': 'Vertical transmission (trait 1)',
        'prop_h2_latent1': 'Prop. h² latent (trait 1)',
        'vg1': 'Genetic variance (trait 1)',
        'vg2': 'Genetic variance (trait 2)',
        'f22': 'Vertical transmission (trait 2)',
        'am22': 'Assortative mating coef.',
        'rg': 'Genetic correlation'
    }
    
    for param, value in zip(PARAM_NAMES, predictions):
        desc = descriptions.get(param, '')
        print(f"{param:<20} {value:<20.4f} {desc}")
    
    print("="*70)
    
    # Create DataFrame for easy export
    pred_df = pd.DataFrame([predictions], columns=PARAM_NAMES)
    
    return pred_df

def main():
    parser = argparse.ArgumentParser(description='Predict parameters from observed PGS correlations')
    parser.add_argument('--model_dir', type=str, default='results_realistic_pgs',
                       help='Directory containing trained model (absolute or relative to script dir)')
    parser.add_argument('--correlations', type=str, default=None,
                       help='Path to CSV file with correlations (columns: RelType, Correlation)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for predictions (default: predictions.csv in model_dir)')
    
    # Alternative: provide correlations as command-line arguments
    parser.add_argument('--S', type=float, help='Siblings correlation')
    parser.add_argument('--HSFS', type=float, help='Half-siblings correlation')
    parser.add_argument('--PSC', type=float, help='Avuncular correlation')
    parser.add_argument('--PPSCC', type=float, help='First cousins correlation')
    parser.add_argument('--M', type=float, help='Mates correlation')
    parser.add_argument('--MS', type=float, help='Siblings-in-law correlation')
    parser.add_argument('--SMS', type=float, help='Sibling\'s mate\'s sibling')
    parser.add_argument('--MSC', type=float, help='Mate\'s sibling\'s child')
    parser.add_argument('--MSM', type=float, help='Mate\'s sibling\'s mate')
    parser.add_argument('--SMSC', type=float, help='Sibling\'s mate\'s sibling\'s child')
    parser.add_argument('--SMSM', type=float, help='Sibling\'s mate\'s sibling\'s mate')
    parser.add_argument('--SMSMS', type=float, help='SMSMS correlation')
    parser.add_argument('--MSMSM', type=float, help='MSMSM correlation')
    parser.add_argument('--MSMSC', type=float, help='MSMSC correlation')
    parser.add_argument('--PSMSC', type=float, help='PSMSC correlation')
    parser.add_argument('--SMSMSC', type=float, help='SMSMSC correlation')
    parser.add_argument('--MSMSMS', type=float, help='MSMSMS correlation')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PARAMETER PREDICTION FROM OBSERVED PGS CORRELATIONS")
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
    model, feature_scaler, target_scaler, poly_transformer, config, device = \
        load_trained_model(model_dir)
    
    # Get correlations
    correlations_dict = {}
    
    if args.correlations:
        # Load from file
        cors_path = Path(args.correlations)
        if not cors_path.is_absolute():
            # Try relative to script directory first
            script_dir = Path(__file__).parent
            if (script_dir / args.correlations).exists():
                cors_path = script_dir / args.correlations
            # Otherwise use as-is (relative to cwd)
        
        print(f"\nLoading correlations from: {cors_path}")
        cors_df = pd.read_csv(cors_path)
        
        # Assuming columns are 'RelType' and 'Correlation' or similar
        if 'RelType' in cors_df.columns and 'Correlation' in cors_df.columns:
            # Handle duplicate relationship types by taking mean
            cors_df = cors_df.groupby('RelType')['Correlation'].mean().reset_index()
            correlations_dict = dict(zip(cors_df['RelType'], cors_df['Correlation']))
        else:
            print("Error: CSV must have 'RelType' and 'Correlation' columns")
            return
    else:
        # Get from command-line arguments
        rel_types = ['S', 'HSFS', 'PSC', 'PPSCC', 'M', 'MS', 'SMS', 'MSC', 'MSM', 
                     'SMSC', 'SMSM', 'SMSMS', 'MSMSM', 'MSMSC', 'PSMSC', 'SMSMSC', 'MSMSMS']
        
        for rel_type in rel_types:
            value = getattr(args, rel_type, None)
            if value is not None:
                correlations_dict[rel_type] = value
    
    if not correlations_dict:
        print("\nError: No correlations provided!")
        print("Either use --correlations <file> or provide values via command-line args")
        print("\nExample:")
        print("  python predict_from_observed.py --S 0.559 --M 0.135 --MS 0.096 ...")
        return
    
    print("\nInput correlations:")
    for rel_type, corr in sorted(correlations_dict.items()):
        print(f"  {rel_type:<10} {corr:.6f}")
    
    # Prepare input
    X = prepare_correlation_input(correlations_dict, config['original_features'])
    
    # Make prediction
    print("\nMaking prediction...")
    predictions = predict_parameters(
        model, feature_scaler, target_scaler, poly_transformer, X, device
    )
    
    # Display results
    pred_df = display_predictions(predictions)
    
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
