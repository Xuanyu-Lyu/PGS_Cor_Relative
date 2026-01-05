"""
Training Script for PGS1-Only Neural Network

This script preprocesses the data, filters to only PGS1 correlations,
and trains a neural network to predict parameters.

Usage:
    python train_pgs1_only.py --data nn_training_combined.csv --output results_pgs1/
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import joblib
import json
import sys

# Import from fit_nn.py
from fit_nn import (
    ParameterPredictor, 
    CorrelationDataset,
    PARAM_NAMES,
    train_epoch,
    validate,
    evaluate_model,
    plot_training_history,
    plot_predictions,
    plot_residuals
)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def load_and_clean_data(data_path):
    """
    Load data and perform initial cleaning.
    
    Returns:
        df: Cleaned DataFrame
    """
    print("\n" + "="*70)
    print("LOADING AND CLEANING DATA")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Show initial data quality
    initial_samples = len(df)
    
    # 1. Remove rows with too many missing values (>50% of correlation columns)
    cor_cols = [col for col in df.columns if col.startswith('cor_')]
    missing_per_row = df[cor_cols].isnull().sum(axis=1) / len(cor_cols)
    df = df[missing_per_row < 0.5].copy()
    print(f"\n✓ Removed {initial_samples - len(df)} rows with >50% missing correlations")
    print(f"  Remaining: {len(df)} samples")
    
    # 2. Remove duplicate rows based on parameters and iteration
    param_cols = [col for col in df.columns if col.startswith('param_')]
    if 'Condition' in df.columns:
        dedup_cols = param_cols + ['Condition', 'Iteration']
    else:
        dedup_cols = param_cols + ['Iteration']
    
    before_dedup = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep='first')
    print(f"✓ Removed {before_dedup - len(df)} duplicate rows")
    print(f"  Remaining: {len(df)} samples")
    
    # 3. Check for extreme outliers in correlations (abs > 1.5, which shouldn't exist)
    outlier_mask = (df[cor_cols].abs() > 1.5).any(axis=1)
    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        print(f"⚠ Warning: Found {n_outliers} rows with impossible correlation values (|r| > 1.5)")
        df = df[~outlier_mask].copy()
        print(f"  Removed these outliers. Remaining: {len(df)} samples")
    
    # 4. Summary statistics
    print(f"\n{'='*70}")
    print("DATA QUALITY SUMMARY")
    print(f"{'='*70}")
    print(f"Final sample size: {len(df)}")
    print(f"\nMissing values per column (top 10):")
    missing = df[cor_cols].isnull().sum().sort_values(ascending=False).head(10)
    for col, count in missing.items():
        pct = (count / len(df)) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")
    
    return df

def filter_pgs1_features(df):
    """
    Filter to only keep PGS1 correlation features.
    
    Returns:
        feature_cols: List of PGS1 correlation column names
    """
    print("\n" + "="*70)
    print("FILTERING TO PGS1 CORRELATIONS ONLY")
    print("="*70)
    
    # Get all PGS1 correlation columns
    pgs1_cols = [col for col in df.columns if 'cor_' in col and '_PGS1' in col]
    
    print(f"\nFound {len(pgs1_cols)} PGS1 correlation features:")
    # Group by relationship type
    relationships = {}
    for col in pgs1_cols:
        rel = col.split('_')[1]  # Extract relationship type (e.g., 'S', 'PSC', etc.)
        if rel not in relationships:
            relationships[rel] = []
        relationships[rel].append(col)
    
    for rel, cols in sorted(relationships.items()):
        print(f"  {rel}: {len(cols)} feature(s)")
    
    return pgs1_cols

def prepare_training_data(df, feature_cols, train_ratio=0.7, val_ratio=0.15, batch_size=32):
    """
    Prepare data for training with PGS1 features only.
    
    Returns:
        train_loader, val_loader, test_loader, feature_scaler, target_scaler, n_features
    """
    print("\n" + "="*70)
    print("PREPARING TRAINING DATA")
    print("="*70)
    
    # Get target columns
    if 'param_f11' in df.columns:
        target_cols = [f'param_{p}' for p in PARAM_NAMES]
    else:
        target_cols = PARAM_NAMES
    
    print(f"\nFeatures: {len(feature_cols)} PGS1 correlations")
    print(f"Targets: {len(target_cols)} parameters")
    
    # Extract features and targets
    X = df[feature_cols].values
    y = df[target_cols].values
    
    # Handle missing values (fill with 0 for correlations)
    if np.isnan(X).any():
        n_missing = np.isnan(X).sum()
        print(f"\n⚠ Warning: Found {n_missing} missing values in features")
        print(f"  Filling with 0 (assumes no correlation)")
        X = np.nan_to_num(X, nan=0.0)
    
    # Split data
    n_samples = len(df)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    # Random split
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples ({train_ratio*100:.0f}%)")
    print(f"  Validation: {len(X_val)} samples ({val_ratio*100:.0f}%)")
    print(f"  Test: {len(X_test)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    # Normalize features
    print("\nNormalizing features...")
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_val = feature_scaler.transform(X_val)
    X_test = feature_scaler.transform(X_test)
    print("✓ Features normalized (mean=0, std=1)")
    
    # Normalize targets
    print("Normalizing targets...")
    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(y_train)
    y_val = target_scaler.transform(y_val)
    y_test = target_scaler.transform(y_test)
    print("✓ Targets normalized (mean=0, std=1)")
    
    # Create datasets and loaders
    train_dataset = CorrelationDataset(X_train, y_train)
    val_dataset = CorrelationDataset(X_val, y_val)
    test_dataset = CorrelationDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, feature_scaler, target_scaler, len(feature_cols)

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, epochs, learning_rate, weight_decay, device, output_dir):
    """
    Train the neural network.
    """
    print("\n" + "="*70)
    print("TRAINING NEURAL NETWORK")
    print("="*70)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 50
    
    print(f"\nHyperparameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Device: {device}")
    
    print(f"\nStarting training...")
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Best Val':<15}")
    print("-" * 70)
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_dir / 'best_model.pt')
        else:
            patience_counter += 1
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"{epoch+1:<8} {train_loss:<15.6f} {val_loss:<15.6f} {best_val_loss:<15.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"{'='*70}\n")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, output_dir)
    
    return train_losses, val_losses

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Train NN with PGS1 correlations only')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to training data CSV')
    parser.add_argument('--output', type=str, default='results_pgs1',
                      help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=500,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay (L2 regularization)')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[256, 128, 64],
                      help='Hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.3,
                      help='Dropout rate')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (cpu, cuda, mps, or auto)')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("NEURAL NETWORK TRAINING - PGS1 CORRELATIONS ONLY")
    print("="*70)
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\nUsing device: {device}")
    
    # Step 1: Load and clean data
    df = load_and_clean_data(args.data)
    
    # Step 2: Filter to PGS1 features only
    feature_cols = filter_pgs1_features(df)
    
    # Step 3: Prepare training data
    train_loader, val_loader, test_loader, feature_scaler, target_scaler, n_features = \
        prepare_training_data(df, feature_cols, batch_size=args.batch_size)
    
    # Save scalers
    joblib.dump(feature_scaler, output_dir / 'feature_scaler.pkl')
    joblib.dump(target_scaler, output_dir / 'target_scaler.pkl')
    print(f"\n✓ Saved scalers to {output_dir}")
    
    # Step 4: Create model
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)
    model = ParameterPredictor(n_features, hidden_sizes=args.hidden_sizes, dropout_rate=args.dropout)
    model = model.to(device)
    
    print(f"\nInput features: {n_features} (PGS1 correlations only)")
    print(f"Output parameters: {len(PARAM_NAMES)}")
    print(f"Hidden layers: {args.hidden_sizes}")
    print(f"Dropout rate: {args.dropout}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Save model architecture
    with open(output_dir / 'model_architecture.txt', 'w') as f:
        f.write(str(model))
    
    # Step 5: Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=output_dir
    )
    
    # Step 6: Load best model for evaluation
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✓ Loaded best model (epoch {checkpoint['epoch']+1})")
    
    # Step 7: Evaluate on test set
    results = evaluate_model(model, test_loader, target_scaler, device, output_dir)
    
    # Step 8: Plot residuals
    plot_residuals(results, output_dir)
    
    # Step 9: Save configuration
    config = {
        'data_path': args.data,
        'n_features': n_features,
        'feature_type': 'PGS1_only',
        'n_samples_total': len(df),
        'n_train': len(train_loader.dataset),
        'n_val': len(val_loader.dataset),
        'n_test': len(test_loader.dataset),
        'hidden_sizes': args.hidden_sizes,
        'dropout_rate': args.dropout,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'epochs_trained': checkpoint['epoch'] + 1,
        'best_val_loss': float(checkpoint['val_loss']),
        'device': str(device),
        'feature_columns': feature_cols
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Saved configuration to {output_dir / 'config.json'}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey files:")
    print(f"  - best_model.pt: Trained model weights")
    print(f"  - feature_scaler.pkl: Feature normalization scaler")
    print(f"  - target_scaler.pkl: Target normalization scaler")
    print(f"  - config.json: Full configuration and metadata")
    print(f"  - test_metrics.json: Detailed test set metrics")
    print(f"  - training_history.png: Loss curves")
    print(f"  - predictions_vs_true.png: Prediction plots")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
