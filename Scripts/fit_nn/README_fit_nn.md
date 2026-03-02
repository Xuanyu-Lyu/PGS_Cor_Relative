# Neural Network Training for Parameter Prediction

Predict simulation parameters from relative correlations between kin types using PyTorch neural networks.

## Target Parameters

| Parameter | Description |
|-----------|-------------|
| `f11` | Vertical transmission (trait 1) |
| `prop_h2_latent1` | Proportion of h² that's latent (trait 1) |
| `vg1` | Genetic variance (trait 1) |
| `vg2` | Genetic variance (trait 2) |
| `f22` | Vertical transmission (trait 2) |
| `am22` | Assortative mating coefficient (trait 2) |
| `rg` | Genetic correlation between traits |

## Two Model Variants

1. **PGS-only**: Uses `cor_*_PGS1` features only
2. **PGS + Phenotypic**: Uses both `cor_*_PGS1` and `cor_*_Y1` features

## Files

| File | Purpose |
|------|---------|
| `fit_nn.py` | Shared utilities (`PARAM_NAMES`, `CorrelationDataset`, `evaluate_model`, `plot_predictions`, `plot_residuals`) |
| `train_realistic_pgs_only.py` | Train PGS-only model |
| `predict_from_observed.py` | Predict from PGS-only model |
| `train_realistic_pgs_and_pheno.py` | Train PGS + phenotypic model |
| `predict_from_observed_pgs_and_pheno.py` | Predict from PGS + phenotypic model |
| `run_prediction.sh` | Shell script to run predictions |

## Requirements

```bash
pip install torch pandas numpy matplotlib scikit-learn joblib
```

## Training

Both training scripts share the same CLI interface:

```bash
# PGS-only
python train_realistic_pgs_only.py \
    --data nn_training_combined400.csv \
    --output ./results_pgs_only \
    --interaction_degree 2 --epochs 3000 --device cpu

# PGS + Phenotypic
python train_realistic_pgs_and_pheno.py \
    --data nn_training_combined_large.csv \
    --output ./results_pgs_and_pheno \
    --interaction_degree 1 --epochs 1500 --device cpu
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | (required) | Training CSV path |
| `--output` | `results_realistic_pgs` / `results_pgs_and_pheno` | Output directory |
| `--epochs` | 1000 | Max training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--hidden_sizes` | 256 256 128 128 | Hidden layer sizes |
| `--dropout` | 0.4 | Dropout rate |
| `--interaction_degree` | 2 | Polynomial degree (1 = no interactions) |
| `--device` | auto | `cpu`, `cuda`, or `mps` |

### Output Files

```
results/
├── best_model.pt              # Model weights
├── feature_scaler.pkl         # Feature scaler
├── target_scaler.pkl          # Target scaler
├── poly_transformer.pkl       # Polynomial transformer (if degree > 1)
├── config.json                # Training config
├── test_metrics.json          # Test set metrics
├── training_history.png       # Loss curves
├── predictions_vs_true.png    # Overview plot
├── residuals.png              # Residual plots
└── prediction_{param}.png     # Per-parameter plots
```

## Prediction

### PGS-only

```bash
python predict_from_observed.py \
    --model_dir results_pgs_only \
    --correlations observed_correlations.csv
```

### PGS + Phenotypic

```bash
python predict_from_observed_pgs_and_pheno.py \
    --model_dir results_pgs_and_pheno \
    --correlations_pgs observed_correlations_PGS.csv \
    --correlations_pheno observed_correlations_pheno.csv
```

Or: `bash run_prediction.sh`

### Observed Correlation CSV Format

Both files need columns `RelType` and `Correlation`:

```csv
RelType,Correlation
S,0.559
M,0.135
MS,0.096
...
```

## Troubleshooting

- **Out of memory**: reduce `--batch_size`
- **Overfitting**: increase `--dropout` or `--weight_decay`
- **NaN loss**: reduce `--lr`
- **Poor R²**: try different `--interaction_degree` or `--hidden_sizes`

### PGS-only model
```bash
python ./train_realistic_pgs_only.py --data nn_training_combined400.csv --device cpu --output ./results_400con --interaction_degree 2 --epochs 3000
```

### PGS + Phenotypic model
```bash
# Train
python train_realistic_pgs_and_pheno.py --data nn_training_combined_large.csv --device cpu --output ./results_pgs_and_pheno --interaction_degree 1 --epochs 1500

# Train with only 3 phenotypic correlations
python train_realistic_pgs_and_pheno.py --data nn_training_combined_large.csv --device cpu --epochs 1500 --y1_features S M MS --output ./results_pgs_and_3pheno --interaction_degree 1

# Train with only 3 phenotypic correlations but using the training data with cross-trait VT
python train_realistic_pgs_and_pheno.py --data nn_training_combined_relaxed_f_250203.csv --device cpu --epochs 1000 --y1_features S M MS --output ./results_pgs_and_3pheno_f_250203 --interaction_degree 1 --param_set extended

# Predict
python predict_from_observed_pgs_and_pheno.py --model_dir results_pgs_and_3pheno_f_250203 --correlations_pgs observed_correlations_PGS.csv --correlations_pheno observed_correlations_pheno.csv

# Predict — param_names are auto-detected from config.json; works for both 7 and 9 output models
python predict_from_observed_pgs_and_pheno.py --model_dir results_pgs_and_3pheno_f_250203 --correlations_pgs observed_correlations_PGS.csv --correlations_pheno observed_correlations_pheno.csv
```