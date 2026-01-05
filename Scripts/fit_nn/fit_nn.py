"""
Neural Network Training Script for Parameter Prediction

This script trains a PyTorch neural network to predict simulation parameters
(f11, prop_h2_latent1, vg1, vg2, f22, am22, rg) from relative PGS correlations.

The model takes correlation values as input features and outputs the 7 parameters
that generated those correlations.

Usage:
    python fit_nn.py --data path/to/nn_training_data.csv --epochs 500 --batch_size 32
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.preprocessing import StandardScaler
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

# Default paths
DEFAULT_DATA_PATH = "/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/DataGeneratingNN/combined/nn_training_data.csv"
DEFAULT_OUTPUT_DIR = "results"  # Use relative path to work both locally and on cluster

# Parameter names (targets to predict)
PARAM_NAMES = ['f11', 'prop_h2_latent1', 'vg1', 'vg2', 'f22', 'am22', 'rg']

# Training hyperparameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 500
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (remaining)

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
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class ParameterPredictor(nn.Module):
    """
    Neural network to predict simulation parameters from correlations.
    
    Architecture:
        - Input layer: n_features (correlation values)
        - Hidden layers: 3 layers with decreasing size
        - Output layer: 7 (parameters)
        - Activation: ReLU for hidden layers
        - Regularization: Dropout, BatchNorm
    """
    def __init__(self, n_features, hidden_sizes=[256, 128, 64], dropout_rate=0.3):
        super(ParameterPredictor, self).__init__()
        
        self.n_features = n_features
        self.n_targets = len(PARAM_NAMES)
        
        # Build architecture
        layers = []
        input_size = n_features
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, self.n_targets))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(data_path, train_ratio=0.7, val_ratio=0.15):
    """
    Load data, split into train/val/test, and normalize features.
    
    Returns:
        train_loader, val_loader, test_loader, feature_scaler, target_scaler, n_features
    """
    print("\n" + "="*70)
    print("LOADING AND PREPROCESSING DATA")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples")
    
    # Separate features and targets
    # Check if parameters have 'param_' prefix
    if 'param_f11' in df.columns:
        target_cols = [f'param_{p}' for p in PARAM_NAMES]
        exclude_cols = target_cols + ['Iteration', 'Condition']
    else:
        target_cols = PARAM_NAMES
        exclude_cols = PARAM_NAMES + ['Iteration', 'Condition']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\nFeatures: {len(feature_cols)} correlation values")
    print(f"Targets: {len(target_cols)} parameters")
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    # Handle missing values (fill with mean)
    if np.isnan(X).any():
        print("\n⚠ Warning: Found NaN values in features, filling with column mean")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    
    # Split data
    n_samples = len(df)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    # Random split
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
    
    # Normalize features (fit on training data only)
    print("\nNormalizing features...")
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_val = feature_scaler.transform(X_val)
    X_test = feature_scaler.transform(X_test)
    print("✓ Features normalized (mean=0, std=1)")
    
    # Normalize targets (helps with training stability)
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
    
    train_loader = DataLoader(train_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, feature_scaler, target_scaler, X_train.shape[1]

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for features, targets in val_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train_model(model, train_loader, val_loader, epochs, learning_rate, weight_decay, device, output_dir):
    """
    Train the neural network.
    """
    print("\n" + "="*70)
    print("TRAINING NEURAL NETWORK")
    print("="*70)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 50  # Early stopping patience
    
    print(f"\nHyperparameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Batch size: {DEFAULT_BATCH_SIZE}")
    print(f"  Optimizer: Adam")
    print(f"  Loss: MSE")
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
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch+1:<8} {train_loss:<15.6f} {val_loss:<15.6f} {best_val_loss:<15.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"{'='*70}\n")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, output_dir)
    
    return train_losses, val_losses

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_loader, target_scaler, device, output_dir):
    """
    Evaluate the model on test set and compute metrics.
    """
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
    
    for i, param_name in enumerate(PARAM_NAMES):
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
    metrics_dict = {
        param: {k: float(v) if k not in ['true', 'pred'] else v.tolist() 
                for k, v in vals.items()}
        for param, vals in results.items()
    }
    metrics_dict['overall'] = {'r2': float(overall_r2), 'rmse': float(overall_rmse)}
    
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Plot predictions vs true values
    plot_predictions(results, output_dir)
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(train_losses, val_losses, output_dir):
    """Plot training and validation loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training history plot")

def plot_predictions(results, output_dir):
    """Plot predicted vs true values for each parameter."""
    n_params = len(PARAM_NAMES)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, param_name in enumerate(PARAM_NAMES):
        ax = axes[i]
        y_true = results[param_name]['true']
        y_pred = results[param_name]['pred']
        r2 = results[param_name]['r2']
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        ax.set_xlabel('True Value', fontsize=10)
        ax.set_ylabel('Predicted Value', fontsize=10)
        ax.set_title(f'{param_name} (R² = {r2:.3f})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_vs_true.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved predictions vs true values plot")
    
    # Also create individual high-res plots
    for param_name in PARAM_NAMES:
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

def plot_residuals(results, output_dir):
    """Plot residuals for each parameter."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, param_name in enumerate(PARAM_NAMES):
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
    
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved residuals plot")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Train neural network for parameter prediction')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA_PATH,
                      help='Path to training data CSV')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                      help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                      help='Batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY,
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
    print("NEURAL NETWORK TRAINING FOR PARAMETER PREDICTION")
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
    
    # Load and preprocess data
    train_loader, val_loader, test_loader, feature_scaler, target_scaler, n_features = \
        load_and_preprocess_data(args.data)
    
    # Save scalers
    import joblib
    joblib.dump(feature_scaler, output_dir / 'feature_scaler.pkl')
    joblib.dump(target_scaler, output_dir / 'target_scaler.pkl')
    print(f"\n✓ Saved scalers to {output_dir}")
    
    # Create model
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)
    model = ParameterPredictor(n_features, hidden_sizes=args.hidden_sizes, dropout_rate=args.dropout)
    model = model.to(device)
    
    print(f"\nInput features: {n_features}")
    print(f"Output parameters: {len(PARAM_NAMES)}")
    print(f"Hidden layers: {args.hidden_sizes}")
    print(f"Dropout rate: {args.dropout}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Save model architecture
    with open(output_dir / 'model_architecture.txt', 'w') as f:
        f.write(str(model))
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=output_dir
    )
    
    # Load best model for evaluation
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✓ Loaded best model (epoch {checkpoint['epoch']+1})")
    
    # Evaluate on test set
    results = evaluate_model(model, test_loader, target_scaler, device, output_dir)
    
    # Plot residuals
    plot_residuals(results, output_dir)
    
    # Save final config
    config = {
        'data_path': args.data,
        'n_features': n_features,
        'n_targets': len(PARAM_NAMES),
        'target_names': PARAM_NAMES,
        'hidden_sizes': args.hidden_sizes,
        'dropout_rate': args.dropout,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'epochs_trained': len(train_losses),
        'best_epoch': checkpoint['epoch'] + 1,
        'best_val_loss': checkpoint['val_loss'],
        'device': str(device)
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - best_model.pt")
    print(f"  - feature_scaler.pkl")
    print(f"  - target_scaler.pkl")
    print(f"  - test_metrics.json")
    print(f"  - training_history.png")
    print(f"  - predictions_vs_true.png")
    print(f"  - residuals.png")
    print(f"  - Individual parameter plots")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
