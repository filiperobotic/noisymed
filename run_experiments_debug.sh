#!/bin/bash
# Script to run all baseline experiments
# Usage: bash run_experiments.sh [dataset]
# Example: bash run_experiments.sh breastmnist

DATASET="${1:-pneumoniamnist}"

echo "=============================================="
echo "Running Baseline Experiments for ${DATASET}"
echo "=============================================="

# Create results directory
mkdir -p results

# Noise levels to test
#NOISE_RATES=(0.0 0.2 0.4 0.6)
NOISE_RATES=(0.0 0.2)

# Run experiments with standard loss
# echo ""
# echo "Running experiments with STANDARD Cross Entropy Loss"
# echo "----------------------------------------------"

# for noise in "${NOISE_RATES[@]}"; do
#     echo ""
#     echo ">>> Running with noise_rate = $noise (Standard Loss)"
#     python train_baseline.py \
#         --dataset $DATASET \
#         --noise_rate $noise \
#         --epochs 10 \
#         --batch_size 128 \
#         --lr 0.01 \
#         --seed 42
# done

# Run experiments with weighted loss
echo ""
echo "Running experiments with WEIGHTED Cross Entropy Loss"
echo "----------------------------------------------"

for noise in "${NOISE_RATES[@]}"; do
    echo ""
    echo ">>> Running with noise_rate = $noise (Weighted Loss)"
    python train_baseline.py \
        --dataset $DATASET \
        --noise_rate $noise \
        --weighted_loss \
        --epochs 10 \
        --batch_size 128 \
        --lr 0.01 \
        --seed 42
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results saved in ./results/"
echo "=============================================="
