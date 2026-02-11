#!/bin/bash
# Script to predict parameters from observed correlations using trained neural network

# Navigate to the fit_nn directory
cd "$(dirname "$0")"

# ======================================================================
# PGS-only model (previous version)
# ======================================================================
# echo "======================================================================="
# echo "Predicting parameters from observed PGS correlations (PGS-only model)"
# echo "======================================================================="
# echo ""
# echo "Using model: results_realistic_pgs"
# echo "Input correlations: observed_correlations.csv"
# echo ""
#
# python predict_from_observed.py \
#     --model_dir results_realistic_pgs \
#     --correlations observed_correlations.csv \
#     --output predictions_observed400.csv

# ======================================================================
# PGS + Phenotypic (Y1) model (new version)
# ======================================================================
echo "======================================================================="
echo "Predicting parameters from observed PGS + Phenotypic correlations"
echo "======================================================================="
echo ""
echo "Using model: results_pgs_and_pheno"
echo "Input PGS correlations: observed_correlations_PGS.csv"
echo "Input phenotypic correlations: observed_correlations_pheno.csv"
echo ""

# Run the prediction script
python predict_from_observed_pgs_and_pheno.py \
    --model_dir results_pgs_and_pheno \
    --correlations_pgs observed_correlations_PGS.csv \
    --correlations_pheno observed_correlations_pheno.csv \
    --output predictions_observed_pgs_pheno.csv

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "✓ Prediction completed successfully!"
    echo "✓ Results saved to: predictions_observed_pgs_pheno.csv"
    echo "======================================================================="
else
    echo ""
    echo "======================================================================="
    echo "✗ Prediction failed - see error messages above"
    echo "======================================================================="
    exit 1
fi
