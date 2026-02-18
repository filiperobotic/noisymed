#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Run Cost-Sensitive Loss experiments — DermaMNIST
# ═══════════════════════════════════════════════════════════════════════════
#
# Tests CS Loss on all base methods (excluding CRASS and derivatives).
# Configuration: 200 epochs, lr 0.01, lambda=20
#
# Methods:
#   1. Baseline + CS
#   2. GMM Filter + CS
#   3. Co-teaching + CS
#   4. DivideMix + CS
#   5. UNICON + CS
#   6. DISC + CS
#
# Usage:
#   bash run_cs_experiments_dermamnist.sh              # All methods
#   bash run_cs_experiments_dermamnist.sh baseline_cs  # Only Baseline+CS
#   bash run_cs_experiments_dermamnist.sh gmm_filter_cs
#   bash run_cs_experiments_dermamnist.sh coteaching_cs
#   bash run_cs_experiments_dermamnist.sh dividemix_cs
#   bash run_cs_experiments_dermamnist.sh unicon_cs
#   bash run_cs_experiments_dermamnist.sh disc_cs
#
# ═══════════════════════════════════════════════════════════════════════════

set -e

# ── Configuration ─────────────────────────────────────────────────────────
DATASET="dermamnist_bin"
NOISE_RATES=("0.0" "0.2" "0.4")

EPOCHS=200
BATCH_SIZE=128
LR=0.01
SEED=42
OUTPUT_DIR="./results"
DATA_DIR="./data"
WARMUP=10

# Cost-sensitive
LAMBDA_RISK=20

# Co-teaching
CT_NUM_GRADUAL=10
CT_EXPONENT=1.0

# ── Filter ────────────────────────────────────────────────────────────────
FILTER="${1:-all}"

# ── Counter ───────────────────────────────────────────────────────────────
CURRENT=0

echo "═══════════════════════════════════════════════════════════════"
echo "  CS Loss Experiments — DermaMNIST"
echo "  Filter: $FILTER"
echo "  Dataset:    $DATASET"
echo "  Noise rates: ${NOISE_RATES[*]}"
echo "  Epochs: $EPOCHS, LR: $LR, Lambda: $LAMBDA_RISK"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# 1. Baseline + CS
# ══════════════════════════════════════════════════════════════════════════

if [ "$FILTER" = "all" ] || [ "$FILTER" = "baseline_cs" ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  Baseline + Cost-Sensitive Loss"
    echo "══════════════════════════════════════════════════════════════"

    for NOISE in "${NOISE_RATES[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [$CURRENT] baseline_cs | $DATASET | noise=$NOISE | lambda=$LAMBDA_RISK"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        python train.py --method baseline_cs \
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
fi

# ══════════════════════════════════════════════════════════════════════════
# 2. GMM Filter + CS
# ══════════════════════════════════════════════════════════════════════════

if [ "$FILTER" = "all" ] || [ "$FILTER" = "gmm_filter_cs" ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  GMM Filter + Cost-Sensitive Loss"
    echo "══════════════════════════════════════════════════════════════"

    for NOISE in "${NOISE_RATES[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [$CURRENT] gmm_filter_cs | $DATASET | noise=$NOISE | lambda=$LAMBDA_RISK"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        python train.py --method gmm_filter_cs \
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
fi

# ══════════════════════════════════════════════════════════════════════════
# 3. Co-teaching + CS
# ══════════════════════════════════════════════════════════════════════════

if [ "$FILTER" = "all" ] || [ "$FILTER" = "coteaching_cs" ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  Co-teaching + Cost-Sensitive Loss"
    echo "══════════════════════════════════════════════════════════════"

    for NOISE in "${NOISE_RATES[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [$CURRENT] coteaching_cs | $DATASET | noise=$NOISE | lambda=$LAMBDA_RISK"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        python train.py --method coteaching_cs \
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
fi

# ══════════════════════════════════════════════════════════════════════════
# 4. DivideMix + CS
# ══════════════════════════════════════════════════════════════════════════

if [ "$FILTER" = "all" ] || [ "$FILTER" = "dividemix_cs" ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  DivideMix + Cost-Sensitive Loss"
    echo "══════════════════════════════════════════════════════════════"

    for NOISE in "${NOISE_RATES[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [$CURRENT] dividemix_cs | $DATASET | noise=$NOISE | lambda=$LAMBDA_RISK"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        python train.py --method dividemix_cs \
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
fi

# ══════════════════════════════════════════════════════════════════════════
# 5. UNICON + CS
# ══════════════════════════════════════════════════════════════════════════

if [ "$FILTER" = "all" ] || [ "$FILTER" = "unicon_cs" ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  UNICON + Cost-Sensitive Loss"
    echo "══════════════════════════════════════════════════════════════"

    for NOISE in "${NOISE_RATES[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [$CURRENT] unicon_cs | $DATASET | noise=$NOISE | lambda=$LAMBDA_RISK"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        python train.py --method unicon_cs \
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
fi

# ══════════════════════════════════════════════════════════════════════════
# 6. DISC + CS
# ══════════════════════════════════════════════════════════════════════════

if [ "$FILTER" = "all" ] || [ "$FILTER" = "disc_cs" ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  DISC + Cost-Sensitive Loss"
    echo "══════════════════════════════════════════════════════════════"

    for NOISE in "${NOISE_RATES[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [$CURRENT] disc_cs | $DATASET | noise=$NOISE | lambda=$LAMBDA_RISK"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        python train.py --method disc_cs \
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
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  DermaMNIST CS experiments completed! ($CURRENT total)"
echo "  Results saved in: $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════════════"
