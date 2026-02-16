#!/bin/bash
# Script to run all baseline experiments
# Usage: bash run_experiments.sh

DATASETS=(dermamnist_bin pathmnist_bin)
NOISE_RATES=(0.0 0.2 0.4 0.6)

# Create results directory
mkdir -p results

for DATASET in "${DATASETS[@]}"; do
    echo "=============================================="
    echo "Running Baseline Experiments for ${DATASET}"
    echo "=============================================="

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
            --epochs 200 \
            --batch_size 128 \
            --lr 0.01 \
            --seed 42
    done
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results saved in ./results/"
echo "=============================================="
