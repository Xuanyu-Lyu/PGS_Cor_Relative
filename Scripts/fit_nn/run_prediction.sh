#!/bin/bash
# Script to predict parameters from observed PGS correlations using trained neural network

# Navigate to the fit_nn directory
cd "$(dirname "$0")"

echo "======================================================================="
echo "Predicting parameters from observed PGS correlations"
echo "======================================================================="
echo ""
echo "Using model: results_realistic_pgs"
echo "Input correlations: observed_correlations.csv"
echo ""

# Run the prediction script
python predict_from_observed.py \
    --model_dir results_realistic_pgs \
    --correlations observed_correlations.csv \
    --output predictions_observed400.csv

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "✓ Prediction completed successfully!"
    echo "✓ Results saved to: results_realistic_pgs/predictions_observed400.csv"
    echo "======================================================================="
else
    echo ""
    echo "======================================================================="
    echo "✗ Prediction failed - see error messages above"
    echo "======================================================================="
    exit 1
fi
