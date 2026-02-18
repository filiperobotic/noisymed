#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Run Cost-Sensitive Loss experiments
# ═══════════════════════════════════════════════════════════════════════════
#
# Experiments:
#   1. Baseline + CS (fixed lambda)
#   2. Baseline + CS (adaptive lambda)
#   3. CRASS + CS
#   4. Co-teaching + CS
#   5. CRASS Adaptive + CS (strategy v2)
#
# Default datasets: dermamnist_bin, pathmnist_bin
# Noise rates: 0.0, 0.2, 0.4, 0.6
#
# Usage:
#   bash run_cost_sensitive_experiments.sh                    # All experiments
#   bash run_cost_sensitive_experiments.sh baseline_cs        # Only Baseline+CS
#   bash run_cost_sensitive_experiments.sh baseline_cs_adapt  # Only Baseline+CS Adaptive
#   bash run_cost_sensitive_experiments.sh crass_cs           # Only CRASS+CS
#   bash run_cost_sensitive_experiments.sh coteaching_cs      # Only Co-teaching+CS
#   bash run_cost_sensitive_experiments.sh crass_adaptive_cs  # Only CRASS Adaptive+CS
#
# ═══════════════════════════════════════════════════════════════════════════

set -e

# ── Configuration ─────────────────────────────────────────────────────────
#DATASETS=("dermamnist_bin" "pathmnist_bin")
DATASETS=("dermamnist_bin")
#NOISE_RATES=("0.0" "0.2" "0.4" "0.6")
NOISE_RATES=("0.0" "0.2" "0.4" )

EPOCHS=200
BATCH_SIZE=128
LR=0.01
SEED=42
OUTPUT_DIR="./results"
DATA_DIR="./data"
WARMUP=10

# Cost-sensitive defaults
LAMBDA_RISK=20

# Co-teaching defaults
CT_FORGET_RATE_DEFAULT=""  # will use noise_rate
CT_NUM_GRADUAL=10
CT_EXPONENT=1.0

# CRASS Adaptive defaults
LAMBDA_MAX=20
LAMBDA_MIN=1
ADAPTIVE_STRATEGY="v2"

# ── Filter ────────────────────────────────────────────────────────────────
FILTER="${1:-all}"

# ── Counter ───────────────────────────────────────────────────────────────
CURRENT=0

echo "═══════════════════════════════════════════════════════════════"
echo "  Running Cost-Sensitive Loss experiments"
echo "  Filter: $FILTER"
echo "  Datasets:    ${DATASETS[*]}"
echo "  Noise rates: ${NOISE_RATES[*]}"
echo "  Lambda:      $LAMBDA_RISK"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# 1. Baseline + Cost-Sensitive Loss (fixed lambda)
# ══════════════════════════════════════════════════════════════════════════

if [ "$FILTER" = "all" ] || [ "$FILTER" = "baseline_cs" ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  EXPERIMENT 1: Baseline + Cost-Sensitive Loss (fixed lambda)"
    echo "══════════════════════════════════════════════════════════════"

    for DATASET in "${DATASETS[@]}"; do
        for NOISE in "${NOISE_RATES[@]}"; do
            CURRENT=$((CURRENT + 1))
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  [$CURRENT] baseline_cs | $DATASET | noise=$NOISE | lambda=$LAMBDA_RISK"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            python train_baseline_cs.py \
                --dataset $DATASET \
                --noise_rate $NOISE \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --lr $LR \
                --seed $SEED \
                --output_dir $OUTPUT_DIR \
                --data_dir $DATA_DIR \
                --lambda_risk $LAMBDA_RISK \
                --print_freq 20

            echo "  [$CURRENT] DONE"
        done
    done
fi

# ══════════════════════════════════════════════════════════════════════════
# 2. Baseline + Cost-Sensitive Loss (adaptive lambda)
# ══════════════════════════════════════════════════════════════════════════

if [ "$FILTER" = "all" ] || [ "$FILTER" = "baseline_cs_adapt" ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  EXPERIMENT 2: Baseline + CS (adaptive lambda)"
    echo "══════════════════════════════════════════════════════════════"

    for DATASET in "${DATASETS[@]}"; do
        for NOISE in "${NOISE_RATES[@]}"; do
            CURRENT=$((CURRENT + 1))
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  [$CURRENT] baseline_cs_adaptive | $DATASET | noise=$NOISE | lambda=$LAMBDA_RISK"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            python train_baseline_cs.py \
                --dataset $DATASET \
                --noise_rate $NOISE \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --lr $LR \
                --seed $SEED \
                --output_dir $OUTPUT_DIR \
                --data_dir $DATA_DIR \
                --lambda_risk $LAMBDA_RISK \
                --adaptive_lambda \
                --print_freq 20

            echo "  [$CURRENT] DONE"
        done
    done
fi

# ══════════════════════════════════════════════════════════════════════════
# 3. CRASS + Cost-Sensitive Loss
# ══════════════════════════════════════════════════════════════════════════

if [ "$FILTER" = "all" ] || [ "$FILTER" = "crass_cs" ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  EXPERIMENT 3: CRASS + Cost-Sensitive Loss"
    echo "══════════════════════════════════════════════════════════════"

    for DATASET in "${DATASETS[@]}"; do
        for NOISE in "${NOISE_RATES[@]}"; do
            CURRENT=$((CURRENT + 1))
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  [$CURRENT] crass_cs | $DATASET | noise=$NOISE | lambda=$LAMBDA_RISK"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            python train_crass_cs.py \
                --dataset $DATASET \
                --noise_rate $NOISE \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --lr $LR \
                --seed $SEED \
                --output_dir $OUTPUT_DIR \
                --data_dir $DATA_DIR \
                --lambda_risk $LAMBDA_RISK \
                --warmup_epochs $WARMUP \
                --print_freq 20

            echo "  [$CURRENT] DONE"
        done
    done
fi

# ══════════════════════════════════════════════════════════════════════════
# 4. Co-teaching + Cost-Sensitive Loss
# ══════════════════════════════════════════════════════════════════════════

if [ "$FILTER" = "all" ] || [ "$FILTER" = "coteaching_cs" ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  EXPERIMENT 4: Co-teaching + Cost-Sensitive Loss"
    echo "══════════════════════════════════════════════════════════════"

    for DATASET in "${DATASETS[@]}"; do
        for NOISE in "${NOISE_RATES[@]}"; do
            CURRENT=$((CURRENT + 1))
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  [$CURRENT] coteaching_cs | $DATASET | noise=$NOISE | lambda=$LAMBDA_RISK"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            python train_coteaching_cs.py \
                --dataset $DATASET \
                --noise_rate $NOISE \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --lr $LR \
                --seed $SEED \
                --output_dir $OUTPUT_DIR \
                --data_dir $DATA_DIR \
                --lambda_risk $LAMBDA_RISK \
                --forget_rate $NOISE \
                --num_gradual $CT_NUM_GRADUAL \
                --exponent $CT_EXPONENT \
                --print_freq 20

            echo "  [$CURRENT] DONE"
        done
    done
fi

# ══════════════════════════════════════════════════════════════════════════
# 5. CRASS Adaptive + Cost-Sensitive Loss
# ══════════════════════════════════════════════════════════════════════════

if [ "$FILTER" = "all" ] || [ "$FILTER" = "crass_adaptive_cs" ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  EXPERIMENT 5: CRASS Adaptive + Cost-Sensitive Loss"
    echo "══════════════════════════════════════════════════════════════"

    for DATASET in "${DATASETS[@]}"; do
        for NOISE in "${NOISE_RATES[@]}"; do
            CURRENT=$((CURRENT + 1))
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  [$CURRENT] crass_adaptive_cs | $DATASET | noise=$NOISE | lambda_max=$LAMBDA_MAX"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            python train_crass_adaptive_cs.py \
                --dataset $DATASET \
                --noise_rate $NOISE \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --lr $LR \
                --seed $SEED \
                --output_dir $OUTPUT_DIR \
                --data_dir $DATA_DIR \
                --lambda_max $LAMBDA_MAX \
                --lambda_min $LAMBDA_MIN \
                --adaptive_strategy $ADAPTIVE_STRATEGY \
                --warmup_epochs $WARMUP \
                --print_freq 20

            echo "  [$CURRENT] DONE"
        done
    done
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All cost-sensitive experiments completed! ($CURRENT total)"
echo "  Results saved in: $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════════════"
