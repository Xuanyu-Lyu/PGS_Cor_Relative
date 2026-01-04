"""
Prediction Script for Trained Neural Network

Use a trained neural network model to predict simulation parameters from
new correlation data.

Usage:
    python predict.py --model results/best_model.pt --data new_correlations.csv
"""

import torch
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import joblib
import json
import sys

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).parent))
from fit_nn import ParameterPredictor, PARAM_NAMES

def load_model(model_path, scalers_dir, device='auto'):
    """
    Load trained model and scalers.
    
    Returns:
        model, feature_scaler, target_scaler, config
    """
    # Determine device
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)
    
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    # Load config
    config_path = Path(scalers_dir) / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load scalers
    feature_scaler = joblib.load(Path(scalers_dir) / 'feature_scaler.pkl')
    target_scaler = joblib.load(Path(scalers_dir) / 'target_scaler.pkl')
    
    # Create model
    model = ParameterPredictor(
        n_features=config['n_features'],
        hidden_sizes=config['hidden_sizes'],
        dropout_rate=config['dropout_rate']
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded (trained for {config['epochs_trained']} epochs)")
    print(f"✓ Best validation loss: {checkpoint['val_loss']:.6f}")
    
    return model, feature_scaler, target_scaler, config, device

def predict(model, feature_scaler, target_scaler, data, device):
    """
    Make predictions on new data.
    
    Parameters:
    -----------
    model : PyTorch model
    feature_scaler : StandardScaler for features
    target_scaler : StandardScaler for targets
    data : DataFrame or numpy array
        Input features (correlations)
    device : torch device
    
    Returns:
    --------
    predictions : DataFrame with predicted parameters
    """
    # Convert to numpy if DataFrame
    if isinstance(data, pd.DataFrame):
        feature_cols = [col for col in data.columns if col not in PARAM_NAMES + ['Iteration']]
        X = data[feature_cols].values
    else:
        X = data
    
    # Handle missing values
    if np.isnan(X).any():
        print("⚠ Warning: Found NaN values, filling with 0")
        X = np.nan_to_num(X, nan=0.0)
    
    # Normalize features
    X_normalized = feature_scaler.transform(X)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_normalized).to(device)
    
    # Predict
    with torch.no_grad():
        predictions_normalized = model(X_tensor).cpu().numpy()
    
    # Inverse transform to original scale
    predictions = target_scaler.inverse_transform(predictions_normalized)
    
    # Create DataFrame
    predictions_df = pd.DataFrame(predictions, columns=PARAM_NAMES)
    
    return predictions_df

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Predict parameters from correlations using trained model')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to trained model (.pt file)')
    parser.add_argument('--scalers_dir', type=str, required=True,
                      help='Directory containing scalers and config')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to CSV file with correlation data')
    parser.add_argument('--output', type=str, default=None,
                      help='Output file for predictions (default: predictions.csv in scalers_dir)')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (cpu, cuda, mps, or auto)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PARAMETER PREDICTION FROM CORRELATIONS")
    print("="*70)
    
    # Load model
    model, feature_scaler, target_scaler, config, device = load_model(
        args.model, args.scalers_dir, args.device
    )
    
    # Load data
    print(f"\nLoading data from: {args.data}")
    data = pd.read_csv(args.data)
    print(f"✓ Loaded {len(data)} samples")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = predict(model, feature_scaler, target_scaler, data, device)
    print(f"✓ Predictions completed for {len(predictions)} samples")
    
    # Display summary statistics
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    print("\nPredicted parameter ranges:")
    print(predictions.describe().loc[['mean', 'std', 'min', 'max']].T)
    
    # Save predictions
    if args.output is None:
        output_path = Path(args.scalers_dir) / 'predictions.csv'
    else:
        output_path = Path(args.output)
    
    predictions.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")
    
    # If original data has true parameters, compute accuracy
    if all(param in data.columns for param in PARAM_NAMES):
        print("\n" + "="*70)
        print("PREDICTION ACCURACY (vs true values)")
        print("="*70)
        
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        print(f"\n{'Parameter':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
        print("-" * 70)
        
        for param in PARAM_NAMES:
            y_true = data[param].values
            y_pred = predictions[param].values
            
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            print(f"{param:<20} {r2:<10.4f} {rmse:<10.4f} {mae:<10.4f}")
        
        # Overall metrics
        y_true_all = data[PARAM_NAMES].values.flatten()
        y_pred_all = predictions.values.flatten()
        overall_r2 = r2_score(y_true_all, y_pred_all)
        overall_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        
        print("-" * 70)
        print(f"{'OVERALL':<20} {overall_r2:<10.4f} {overall_rmse:<10.4f}")
    
    print("\n" + "="*70)
    print("PREDICTION COMPLETE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
