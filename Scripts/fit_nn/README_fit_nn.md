# Neural Network Training for Parameter Prediction

This directory contains PyTorch-based neural network models to predict simulation parameters from relative PGS correlations.

## Overview

The neural network takes correlation values (from siblings, cousins, in-laws, etc.) as input and predicts the 7 parameters that generated those correlations:
- `f11`: Vertical transmission for trait 1
- `prop_h2_latent1`: Proportion of h² that's latent for trait 1
- `vg1`: Genetic variance for trait 1
- `vg2`: Genetic variance for trait 2
- `f22`: Vertical transmission for trait 2
- `am22`: Assortative mating coefficient for trait 2
- `rg`: Genetic correlation between traits

## Files

- **`fit_nn.py`**: Main training script
- **`predict.py`**: Prediction script for using trained models
- **`README_fit_nn.md`**: This documentation

## Requirements

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn joblib
```

Or use your existing conda environment:
```bash
conda activate /projects/xuly4739/general_env
```

## Model Architecture

**ParameterPredictor** - Fully connected neural network:
- Input layer: ~40-50 features (correlation values)
- Hidden layers: [256, 128, 64] neurons (configurable)
- Activation: ReLU
- Regularization: BatchNorm + Dropout (0.3)
- Output layer: 7 parameters
- Loss function: MSE (Mean Squared Error)
- Optimizer: Adam with learning rate scheduling

## Training

### Basic Usage

```bash
python fit_nn.py --data path/to/nn_training_data.csv
```

### Advanced Options

```bash
python fit_nn.py \
    --data /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/DataGeneratingNN/combined/nn_training_data.csv \
    --output results/ \
    --epochs 500 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --hidden_sizes 256 128 64 \
    --dropout 0.3 \
    --device auto
```

### Parameters

- `--data`: Path to training data CSV (required)
- `--output`: Output directory for results (default: `results/`)
- `--epochs`: Maximum training epochs (default: 500)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: L2 regularization (default: 1e-5)
- `--hidden_sizes`: Hidden layer sizes (default: 256 128 64)
- `--dropout`: Dropout rate (default: 0.3)
- `--device`: Device to use - auto/cpu/cuda/mps (default: auto)

## Training Features

### Data Preprocessing
- Train/validation/test split: 70%/15%/15%
- Feature normalization (StandardScaler)
- Target normalization for training stability
- Missing value imputation

### Training Enhancements
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping (patience=50 epochs)
- Model checkpointing (saves best model)
- Progress monitoring

### Output Files

After training, the following files are saved to the output directory:

```
results/
├── best_model.pt                    # Trained model weights
├── feature_scaler.pkl              # Feature normalization scaler
├── target_scaler.pkl               # Target normalization scaler
├── config.json                     # Training configuration
├── model_architecture.txt          # Model architecture summary
├── test_metrics.json               # Detailed test metrics
├── training_history.png            # Loss curves
├── predictions_vs_true.png         # All parameters overview
├── residuals.png                   # Residual plots
└── prediction_{param}.png          # Individual parameter plots (7 files)
```

## Making Predictions

Use the trained model to predict parameters from new correlation data:

```bash
python predict.py \
    --model results/best_model.pt \
    --scalers_dir results/ \
    --data new_correlations.csv \
    --output predictions.csv
```

### Input Data Format

The input CSV should have the same correlation columns as the training data:
- Column names: `{RelationshipType}_{Variable}_cor`
- Example: `S_PGS1_cor`, `PPSCC_PGS2_cor`, `M_Y1_cor`
- Variables: PGS1, PGS2, Y1, Y2
- Relationships: S, PSC, PPSCC, M, MS, SMS, MSC, MSM, SMSC, SMSM, SMSMS

### Output Format

The output CSV contains predicted values for all 7 parameters:
```csv
f11,prop_h2_latent1,vg1,vg2,f22,am22,rg
0.1234,0.7890,0.6543,0.8765,0.2109,0.5432,0.7654
```

## Example Workflow

### 1. Train Model

```bash
cd /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Scripts/fit_nn

python fit_nn.py \
    --data /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/DataGeneratingNN/combined/nn_training_data.csv \
    --output results_v1/ \
    --epochs 500
```

**Expected output:**
- Training completes in ~20-30 minutes on GPU
- Best validation loss: ~0.01-0.05 (normalized scale)
- Test R² scores: typically 0.85-0.95 for most parameters

### 2. Evaluate Results

Check the generated plots:
```bash
open results_v1/predictions_vs_true.png
open results_v1/training_history.png
```

Review metrics:
```bash
cat results_v1/test_metrics.json
```

### 3. Make Predictions

```bash
python predict.py \
    --model results_v1/best_model.pt \
    --scalers_dir results_v1/ \
    --data /path/to/new/correlation/data.csv \
    --output predicted_parameters.csv
```

## Performance Optimization

### GPU Training

The script automatically detects and uses available GPUs. To force CPU/GPU:
```bash
# Force CPU
python fit_nn.py --device cpu ...

# Force CUDA GPU
python fit_nn.py --device cuda ...

# Force Apple Silicon GPU
python fit_nn.py --device mps ...
```

### Hyperparameter Tuning

Experiment with different architectures:
```bash
# Deeper network
python fit_nn.py --hidden_sizes 512 256 128 64 ...

# Less regularization
python fit_nn.py --dropout 0.2 --weight_decay 1e-6 ...

# Higher learning rate
python fit_nn.py --lr 0.005 ...
```

## Monitoring Training

Monitor training progress in real-time:
```bash
tail -f results/training_log.txt  # if logging enabled
```

Or check validation loss periodically:
```bash
# Training prints every 10 epochs
Epoch    Train Loss      Val Loss        Best Val       
----------------------------------------------------------------------
10       0.234567        0.123456        0.123456       
20       0.123456        0.098765        0.098765       
...
```

## Troubleshooting

### Out of Memory Errors

Reduce batch size:
```bash
python fit_nn.py --batch_size 16 ...
```

### Overfitting

Increase regularization:
```bash
python fit_nn.py --dropout 0.4 --weight_decay 1e-4 ...
```

### Poor Performance

- Check data quality (missing values, outliers)
- Try different architectures
- Increase training epochs
- Adjust learning rate

### NaN Loss

- Reduce learning rate
- Check for extreme values in data
- Ensure proper normalization

## Expected Performance

With 2,000 training samples (200 conditions × 10 iterations):

| Parameter | Expected R² | Expected RMSE |
|-----------|-------------|---------------|
| f11 | 0.90-0.95 | 0.01-0.02 |
| prop_h2_latent1 | 0.85-0.92 | 0.03-0.05 |
| vg1 | 0.88-0.94 | 0.02-0.04 |
| vg2 | 0.87-0.93 | 0.03-0.05 |
| f22 | 0.91-0.96 | 0.01-0.02 |
| am22 | 0.92-0.97 | 0.01-0.02 |
| rg | 0.89-0.95 | 0.02-0.03 |
| **Overall** | **0.90-0.94** | **0.02-0.03** |

*Note: Performance depends on data quality, parameter ranges, and model architecture.*

## Citation

If you use this neural network approach in your research, please cite:
- PyTorch framework
- Your simulation methods paper
- This repository/project

## Advanced Usage

### Cross-Validation

For more robust evaluation, implement k-fold cross-validation:
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    # Train model on each fold
    ...
```

### Ensemble Models

Train multiple models and average predictions:
```bash
for i in {1..5}; do
    python fit_nn.py --output results_ensemble_$i/ --seed $i
done
```

### Feature Importance

Analyze which correlations are most important using:
- Gradient-based feature importance
- Permutation importance
- SHAP values

## Support

For issues or questions:
1. Check training logs and error messages
2. Review test metrics and plots
3. Verify data format and preprocessing
4. Try different hyperparameters

## Next Steps

After successful training:
1. **Validate** on independent simulation data
2. **Apply** to real empirical data
3. **Compare** predictions with other methods
4. **Refine** model architecture based on performance
5. **Document** findings and performance benchmarks

## useful command
python ./train_realistic_pgs_only.py --data nn_training_combined400.csv --device cpu --output ./results_400con --interaction_degree 2 --epochs 3000