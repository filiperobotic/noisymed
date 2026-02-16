#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Run CRASS experiments with lambda sweep
# ═══════════════════════════════════════════════════════════════════════════
#
# Tests CRASS across multiple lambda values to demonstrate the
# BAC vs Clinical Risk trade-off.
#
# Usage:
#   bash run_crass_experiments.sh                           # All datasets + lambdas
#   bash run_crass_experiments.sh pneumoniamnist            # Single dataset
#   bash run_crass_experiments.sh pneumoniamnist 0.2        # Single dataset + noise
#
# ═══════════════════════════════════════════════════════════════════════════

set -e

DATASETS=("pneumoniamnist" "breastmnist" "dermamnist_bin" "pathmnist_bin")
NOISE_RATES=("0.0" "0.2" "0.4" "0.6")
LAMBDAS=("1" "5" "10" "20")

EPOCHS=100
BATCH_SIZE=128
LR=0.01
SEED=42
WARMUP=10
OUTPUT_DIR="./results"
DATA_DIR="./data"

# ── Filter by CLI arguments ──────────────────────────────────────────────
FILTER_DATASET="${1:-}"
FILTER_NOISE="${2:-}"

if [ -n "$FILTER_DATASET" ]; then
    DATASETS=("$FILTER_DATASET")
    echo "Filtering to dataset: $FILTER_DATASET"
fi

if [ -n "$FILTER_NOISE" ]; then
    NOISE_RATES=("$FILTER_NOISE")
    echo "Filtering to noise rate: $FILTER_NOISE"
fi

TOTAL=$((${#DATASETS[@]} * ${#NOISE_RATES[@]} * ${#LAMBDAS[@]}))
CURRENT=0

echo "═══════════════════════════════════════════════════════════════"
echo "  CRASS Lambda Sweep: $TOTAL experiments"
echo "  Datasets:    ${DATASETS[*]}"
echo "  Noise rates: ${NOISE_RATES[*]}"
echo "  Lambdas:     ${LAMBDAS[*]}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

for DATASET in "${DATASETS[@]}"; do
    for NOISE in "${NOISE_RATES[@]}"; do
        for LAMBDA in "${LAMBDAS[@]}"; do
            CURRENT=$((CURRENT + 1))
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  [$CURRENT/$TOTAL] CRASS | $DATASET | noise=$NOISE | lambda=$LAMBDA"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            python train_crass.py \
                --dataset $DATASET \
                --noise_rate $NOISE \
                --lambda_risk $LAMBDA \
                --warmup_epochs $WARMUP \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --lr $LR \
                --seed $SEED \
                --output_dir $OUTPUT_DIR \
                --data_dir $DATA_DIR \
                --weighted_loss \
                --print_freq 20

            echo "  [$CURRENT/$TOTAL] DONE: CRASS | $DATASET | noise=$NOISE | lambda=$LAMBDA"
        done
    done
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All $TOTAL CRASS experiments completed!"
echo "  Results saved in: $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════════════"
