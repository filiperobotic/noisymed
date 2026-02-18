#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Run CRASS Adaptive and Co-teaching + CRASS experiments
# ═══════════════════════════════════════════════════════════════════════════
#
# Experiments:
#   1. CRASS Adaptive (3 strategies: v1, v2, v3)
#   2. Co-teaching + CRASS (lambda=20 and lambda adapted per noise)
#   3. Lambda sweep for CRASS at 40% noise (to find optimal fixed lambda)
#
# Default datasets: dermamnist_bin, pathmnist_bin
# Noise rates: 0.0, 0.2, 0.4, 0.6
#
# Usage:
#   bash run_new_experiments.sh                           # All experiments
#   bash run_new_experiments.sh crass_adaptive             # Only CRASS Adaptive
#   bash run_new_experiments.sh coteaching_crass            # Only Co-teaching+CRASS
#   bash run_new_experiments.sh lambda_sweep                # Only lambda sweep
#
# ═══════════════════════════════════════════════════════════════════════════

set -e

# ── Configuration ─────────────────────────────────────────────────────────
DATASETS=("dermamnist_bin" "pathmnist_bin")
NOISE_RATES=("0.0" "0.2" "0.4" "0.6")

EPOCHS=200
BATCH_SIZE=128
LR=0.01
SEED=42
OUTPUT_DIR="./results"
DATA_DIR="./data"
WARMUP=10

# CRASS Adaptive defaults
LAMBDA_MAX=20
LAMBDA_MIN=1
STRATEGIES=("v1" "v2" "v3")

# Co-teaching + CRASS defaults
CT_NUM_GRADUAL=10
CT_EXPONENT=1.0
CT_LAMBDAS=("20")

# Lambda sweep (at 40% noise)
SWEEP_LAMBDAS=("1" "5" "10" "15" "20")

# ── Filter ────────────────────────────────────────────────────────────────
FILTER="${1:-all}"

# ── Counter ───────────────────────────────────────────────────────────────
CURRENT=0

echo "═══════════════════════════════════════════════════════════════"
echo "  Running new CRASS experiments"
echo "  Filter: $FILTER"
echo "  Datasets:    ${DATASETS[*]}"
echo "  Noise rates: ${NOISE_RATES[*]}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# 1. CRASS Adaptive
# ══════════════════════════════════════════════════════════════════════════

if [ "$FILTER" = "all" ] || [ "$FILTER" = "crass_adaptive" ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  EXPERIMENT 1: CRASS Adaptive"
    echo "══════════════════════════════════════════════════════════════"

    for DATASET in "${DATASETS[@]}"; do
        for NOISE in "${NOISE_RATES[@]}"; do
            for STRATEGY in "${STRATEGIES[@]}"; do
                CURRENT=$((CURRENT + 1))
                echo ""
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo "  [$CURRENT] crass_adaptive ($STRATEGY) | $DATASET | noise=$NOISE"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

                python train_crass_adaptive.py \
                    --dataset $DATASET \
                    --noise_rate $NOISE \
                    --epochs $EPOCHS \
                    --batch_size $BATCH_SIZE \
                    --lr $LR \
                    --seed $SEED \
                    --output_dir $OUTPUT_DIR \
                    --data_dir $DATA_DIR \
                    --weighted_loss \
                    --print_freq 20 \
                    --warmup_epochs $WARMUP \
                    --lambda_max $LAMBDA_MAX \
                    --lambda_min $LAMBDA_MIN \
                    --adaptive_strategy $STRATEGY

                echo "  [$CURRENT] DONE"
            done
        done
    done
fi

# ══════════════════════════════════════════════════════════════════════════
# 2. Co-teaching + CRASS
# ══════════════════════════════════════════════════════════════════════════

if [ "$FILTER" = "all" ] || [ "$FILTER" = "coteaching_crass" ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  EXPERIMENT 2: Co-teaching + CRASS"
    echo "══════════════════════════════════════════════════════════════"

    for DATASET in "${DATASETS[@]}"; do
        for NOISE in "${NOISE_RATES[@]}"; do
            for LAMBDA in "${CT_LAMBDAS[@]}"; do
                CURRENT=$((CURRENT + 1))
                echo ""
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo "  [$CURRENT] coteaching_crass | $DATASET | noise=$NOISE | lambda=$LAMBDA"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

                python train_coteaching_crass.py \
                    --dataset $DATASET \
                    --noise_rate $NOISE \
                    --epochs $EPOCHS \
                    --batch_size $BATCH_SIZE \
                    --lr $LR \
                    --seed $SEED \
                    --output_dir $OUTPUT_DIR \
                    --data_dir $DATA_DIR \
                    --weighted_loss \
                    --print_freq 20 \
                    --forget_rate $NOISE \
                    --num_gradual $CT_NUM_GRADUAL \
                    --exponent $CT_EXPONENT \
                    --lambda_risk $LAMBDA \
                    --warmup_epochs $WARMUP

                echo "  [$CURRENT] DONE"
            done
        done
    done
fi

# ══════════════════════════════════════════════════════════════════════════
# 3. Lambda sweep for CRASS at 40% noise (find best fixed lambda)
# ══════════════════════════════════════════════════════════════════════════

if [ "$FILTER" = "all" ] || [ "$FILTER" = "lambda_sweep" ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  EXPERIMENT 3: CRASS Lambda Sweep (noise=0.4)"
    echo "══════════════════════════════════════════════════════════════"

    for DATASET in "${DATASETS[@]}"; do
        for LAMBDA in "${SWEEP_LAMBDAS[@]}"; do
            CURRENT=$((CURRENT + 1))
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  [$CURRENT] crass | $DATASET | noise=0.4 | lambda=$LAMBDA"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            python train_crass.py \
                --dataset $DATASET \
                --noise_rate 0.4 \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --lr $LR \
                --seed $SEED \
                --output_dir $OUTPUT_DIR \
                --data_dir $DATA_DIR \
                --weighted_loss \
                --print_freq 20 \
                --warmup_epochs $WARMUP \
                --lambda_risk $LAMBDA

            echo "  [$CURRENT] DONE"
        done
    done
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All experiments completed! ($CURRENT total)"
echo "  Results saved in: $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════════════"
