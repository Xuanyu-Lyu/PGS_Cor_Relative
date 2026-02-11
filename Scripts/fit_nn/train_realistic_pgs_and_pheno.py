"""
Realistic Training for PGS + Phenotypic (Y1) Predictions

Uses both PGS correlations and phenotypic (Y1) correlations as input features
to predict simulation parameters. This extends the PGS-only model by
incorporating phenotypic correlations of trait 1 as additional features.

Features used:
- cor_*_PGS1: PGS correlations for trait 1
- cor_*_Y1: Phenotypic correlations for trait 1

Improvements over PGS-only:
1. Additional phenotypic features provide complementary information
2. Feature engineering (interactions, polynomials)
3. More relationship types
4. Parameter-specific models (some parameters predict better than others)
5. Uncertainty quantification

Usage:
    python train_realistic_pgs_and_pheno.py --data nn_training_combined400.csv --device cpu --interaction_degree 1 --epochs 1000
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

from fit_nn import CorrelationDataset, PARAM_NAMES
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================


def load_and_clean_data(data_path, missing_threshold=0.95):
    """
    Load data and perform initial cleaning.
    
    Args:
        data_path: Path to the CSV file
        missing_threshold: Maximum proportion of missing correlations allowed (default: 0.95)
        
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
    
    # 1. Remove rows with too many missing values
    cor_cols = [col for col in df.columns if col.startswith('cor_')]
    
    if len(cor_cols) == 0:
        print("\n⚠ Warning: No correlation columns found!")
        return df
    
    missing_per_row = df[cor_cols].isnull().sum(axis=1) / len(cor_cols)
    
    # Show distribution of missing data
    print(f"\nMissing data distribution:")
    print(f"  Min: {missing_per_row.min():.1%}")
    print(f"  Mean: {missing_per_row.mean():.1%}")
    print(f"  Median: {missing_per_row.median():.1%}")
    print(f"  Max: {missing_per_row.max():.1%}")
    
    # Apply threshold
    df = df[missing_per_row < missing_threshold].copy()
    print(f"\n✓ Removed {initial_samples - len(df)} rows with >{missing_threshold:.0%} missing correlations")
    print(f"  Remaining: {len(df)} samples")
    
    if len(df) == 0:
        print("\n✗ ERROR: All rows removed during cleaning!")
        print("  Try adjusting --missing_threshold parameter")
        return df
    
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

def engineer_features(df, feature_cols, interaction_degree=2):
    """
    Create engineered features from PGS + phenotypic correlations.
    
    Adds:
    - Polynomial features (interactions)
    - Ratios between relationship types
    - Differences
    """
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    
    X = df[feature_cols].values
    
    # Fill NaN with 0 for feature engineering
    X_filled = np.nan_to_num(X, nan=0.0)
    
    print(f"\nOriginal features: {X.shape[1]}")
    
    # Add polynomial features (interactions)
    if interaction_degree > 1:
        poly = PolynomialFeatures(degree=interaction_degree, include_bias=False)
        X_poly = poly.fit_transform(X_filled)
        print(f"After polynomial (degree={interaction_degree}): {X_poly.shape[1]}")
        
        # Get feature names
        if hasattr(poly, 'get_feature_names_out'):
            feature_names = poly.get_feature_names_out(feature_cols)
        else:
            feature_names = [f"poly_{i}" for i in range(X_poly.shape[1])]
        
        return X_poly, list(feature_names), poly
    
    return X_filled, feature_cols, None

# ============================================================================
# PARAMETER-SPECIFIC ANALYSIS
# ============================================================================

def analyze_parameter_predictability(df, feature_cols):
    """
    Analyze which parameters are predictable from PGS + phenotypic correlations.
    """
    print("\n" + "="*70)
    print("PARAMETER PREDICTABILITY ANALYSIS (PGS + Phenotypic)")
    print("="*70)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    
    # Get target columns
    if 'param_f11' in df.columns:
        target_cols = [f'param_{p}' for p in PARAM_NAMES]
    else:
        target_cols = PARAM_NAMES
    
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    
    print(f"\nTesting predictability with Random Forest (5-fold CV):\n")
    print(f"{'Parameter':<20} {'Mean R²':<12} {'Std R²':<12} {'Predictable?'}")
    print("-" * 70)
    
    predictability = {}
    for target_col, param_name in zip(target_cols, PARAM_NAMES):
        y = df[target_col].values
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
        
        mean_r2 = scores.mean()
        std_r2 = scores.std()
        is_predictable = mean_r2 > 0.5  # Threshold for "predictable"
        
        predictability[param_name] = {
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'predictable': is_predictable
        }
        
        status = "✓ Yes" if is_predictable else "✗ Limited"
        print(f"{param_name:<20} {mean_r2:<12.4f} {std_r2:<12.4f} {status}")
    
    print("\n" + "="*70)
    
    return predictability

# ============================================================================
# IMPROVED MODEL FOR PGS + PHENOTYPIC FEATURES
# ============================================================================

class AttentionLayer(nn.Module):
    """Self-attention layer to weight features."""
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size), # Output matches input size
            nn.Sigmoid() # Scale features between 0 and 1 independently
        )
        
    def forward(self, x):
        weights = self.attention(x)
        return x * weights

class FeatureAwarePredictorPgsPheno(nn.Module):
    """
    Model designed for PGS + phenotypic features:
    - Attention mechanism to focus on important correlations
    - Deeper initial layers to extract patterns
    - Parameter-specific branches
    """
    def __init__(self, n_features, hidden_sizes=[512, 512, 256, 256], dropout_rate=0.4):
        super(FeatureAwarePredictorPgsPheno, self).__init__()
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_features, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_sizes[1])
        
        # Deeper processing
        self.deeper = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.BatchNorm1d(hidden_sizes[3]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Output layer
        self.output = nn.Linear(hidden_sizes[3], len(PARAM_NAMES))
        
    def forward(self, x):
        features = self.feature_extractor(x)
        attended = self.attention(features)
        deep_features = self.deeper(attended)
        output = self.output(deep_features)
        return output

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, epochs, learning_rate, weight_decay, 
                device, output_dir):
    """Train model with best practices."""
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Reduce LR on plateau with more patience
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-7
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 150
    
    print(f"\nStarting training...")
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Best Val':<15}")
    print("-" * 70)
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        
        val_loss = total_val_loss / len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
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
        
        if (epoch + 1) % 20 == 0:
            print(f"{epoch+1:<8} {train_loss:<15.6f} {val_loss:<15.6f} {best_val_loss:<15.6f}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*70}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"{'='*70}\n")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training', linewidth=2)
    plt.plot(val_losses, label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History (PGS + Phenotypic)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'training_history.png', dpi=300)
    plt.close()
    
    return train_losses, val_losses

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='results_pgs_and_pheno')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[256, 256, 128, 128])
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--interaction_degree', type=int, default=2,
                       help='Polynomial degree for feature interactions (1=no interactions, 2=pairwise)')
    parser.add_argument('--analyze_predictability', action='store_true',
                       help='Run predictability analysis before training')
    parser.add_argument('--missing_threshold', type=float, default=0.95,
                       help='Maximum proportion of missing correlations allowed (default: 0.95)')
    parser.add_argument('--y1_features', type=str, nargs='+', default=None,
                       help='Y1 phenotypic features to use (default: all features). Specify subset like: S PSC M MS')
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("REALISTIC PGS + PHENOTYPIC (Y1) NEURAL NETWORK TRAINING")
    print("="*70)
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    
    # Load data
    df = load_and_clean_data(args.data, missing_threshold=args.missing_threshold)
    
    if len(df) == 0:
        print("\n✗ No data available after cleaning. Exiting.")
        return
    
    # Get PGS1 features - check for both naming conventions
    pgs1_cols = [col for col in df.columns if 'cor_' in col and '_PGS1' in col]
    
    # If no 'cor_' prefix columns, look for columns ending with _PGS1 (but not _N suffix)
    if len(pgs1_cols) == 0:
        pgs1_cols = [col for col in df.columns 
                     if '_PGS1' in col and not col.endswith('_N') 
                     and not col.startswith('param_')]
    
    # Get Y1 (phenotypic trait 1) features
    if args.y1_features is None:
        # Use all Y1 features (default)
        print("\nUsing ALL Y1 phenotypic features")
        y1_cols = [col for col in df.columns if 'cor_' in col and '_Y1' in col]
        
        # If no 'cor_' prefix columns, look for columns ending with _Y1 (but not _N suffix)
        if len(y1_cols) == 0:
            y1_cols = [col for col in df.columns 
                       if '_Y1' in col and not col.endswith('_N') 
                       and not col.startswith('param_') and not col.startswith('n_')]
    else:
        # Use customized subset
        allowed_y1_rels = args.y1_features
        print(f"\nUsing CUSTOMIZED Y1 phenotypic features: {allowed_y1_rels}")
        y1_cols = [col for col in df.columns 
                   if 'cor_' in col and '_Y1' in col 
                   and any(f'cor_{rel}_Y1' == col for rel in allowed_y1_rels)]
        
        # If no 'cor_' prefix columns, look for columns ending with _Y1 (but not _N suffix)
        if len(y1_cols) == 0:
            y1_cols = [col for col in df.columns 
                       if '_Y1' in col and not col.endswith('_N') 
                       and not col.startswith('param_') and not col.startswith('n_')
                       and any(f'{rel}_Y1' == col for rel in allowed_y1_rels)]
    
    # Combine PGS1 and Y1 features
    combined_feature_cols = pgs1_cols + y1_cols
    
    print(f"\nFound {len(pgs1_cols)} PGS1 correlation features")
    print(f"Found {len(y1_cols)} Y1 (phenotypic) correlation features")
    print(f"Total combined features: {len(combined_feature_cols)}")
    
    if len(pgs1_cols) > 0:
        print(f"\nPGS1 features: {pgs1_cols}")
    if len(y1_cols) > 0:
        print(f"\nY1 features: {y1_cols}")
    
    # Predictability analysis
    if args.analyze_predictability:
        predictability = analyze_parameter_predictability(df, combined_feature_cols)
        with open(output_dir / 'predictability_analysis.json', 'w') as f:
            json.dump(predictability, f, indent=2)
    
    # Feature engineering
    X_engineered, feature_names, poly_transformer = engineer_features(
        df, combined_feature_cols, args.interaction_degree
    )
    
    # Prepare targets
    if 'param_f11' in df.columns:
        target_cols = [f'param_{p}' for p in PARAM_NAMES]
    else:
        target_cols = PARAM_NAMES
    
    y = df[target_cols].values
    
    # Split data
    n_samples = len(df)
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)
    
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    
    X_train, y_train = X_engineered[indices[:n_train]], y[indices[:n_train]]
    X_val, y_val = X_engineered[indices[n_train:n_train+n_val]], y[indices[n_train:n_train+n_val]]
    X_test, y_test = X_engineered[indices[n_train+n_val:]], y[indices[n_train+n_val:]]
    
    print(f"\nData split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    
    # Normalize
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_val = feature_scaler.transform(X_val)
    X_test = feature_scaler.transform(X_test)
    
    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(y_train)
    y_val = target_scaler.transform(y_val)
    y_test = target_scaler.transform(y_test)
    
    # Save scalers and transformer
    joblib.dump(feature_scaler, output_dir / 'feature_scaler.pkl')
    joblib.dump(target_scaler, output_dir / 'target_scaler.pkl')
    if poly_transformer:
        joblib.dump(poly_transformer, output_dir / 'poly_transformer.pkl')
    
    # Create data loaders
    train_dataset = CorrelationDataset(X_train, y_train)
    val_dataset = CorrelationDataset(X_val, y_val)
    test_dataset = CorrelationDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)
    
    n_features = X_engineered.shape[1]
    model = FeatureAwarePredictorPgsPheno(n_features, args.hidden_sizes, args.dropout)
    model = model.to(device)
    
    print(f"\nInput features: {n_features} (after engineering)")
    print(f"  - PGS1 features: {len(pgs1_cols)}")
    print(f"  - Y1 features: {len(y1_cols)}")
    print(f"Hidden layers: {args.hidden_sizes}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=output_dir
    )
    
    # Load best and evaluate
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    from fit_nn import evaluate_model, plot_predictions, plot_residuals
    results = evaluate_model(model, test_loader, target_scaler, device, output_dir)
    plot_predictions(results, output_dir)
    plot_residuals(results, output_dir)
    
    # Save config
    config = {
        'n_original_pgs1_features': len(pgs1_cols),
        'n_original_y1_features': len(y1_cols),
        'n_original_features': len(combined_feature_cols),
        'n_engineered_features': n_features,
        'interaction_degree': args.interaction_degree,
        'hidden_sizes': args.hidden_sizes,
        'dropout_rate': args.dropout,
        'epochs_trained': checkpoint['epoch'] + 1,
        'best_val_loss': float(checkpoint['val_loss']),
        'original_features': combined_feature_cols,
        'pgs1_features': pgs1_cols,
        'y1_features': y1_cols,
        'note': 'PGS1 + Y1 phenotypic model with feature engineering for realistic deployment'
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✓ Training complete!\n")
    print("💡 RECOMMENDATION:")
    print("   This model uses both PGS1 and phenotypic (Y1) correlations.")
    print("   If R² is still low for some parameters, consider:")
    print("   1. Adding more relationship types (rerun analysis with new types)")
    print("   2. Training separate models for well-predicted parameters")
    print("   3. Using ensemble methods")
    print("   4. Adjusting interaction degree for feature engineering\n")

if __name__ == "__main__":
    main()
