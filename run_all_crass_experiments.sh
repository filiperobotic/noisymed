#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Run all CRASS variant experiments
# ═══════════════════════════════════════════════════════════════════════════
#
# This script runs:
#   1. Original methods (gmm_filter, dividemix, disc) as baselines
#   2. CRASS variants (gmm_filter_crass, dividemix_crass, disc_crass)
#      with multiple lambda values
#
# Datasets:  pneumoniamnist, breastmnist, dermamnist_bin, pathmnist_bin
# Noise:     0.0, 0.2, 0.4, 0.6
# Lambdas:   1, 5, 10, 20
#
# Usage:
#   bash run_all_crass_experiments.sh                         # All experiments
#   bash run_all_crass_experiments.sh gmm_filter_crass        # Single method
#   bash run_all_crass_experiments.sh gmm_filter_crass 0.2    # Method + noise
#
# ═══════════════════════════════════════════════════════════════════════════

set -e

# ── Configuration ─────────────────────────────────────────────────────────
#DATASETS=("pneumoniamnist" "breastmnist" "dermamnist_bin" "pathmnist_bin")
DATASETS=( "dermamnist_bin" "pathmnist_bin")
NOISE_RATES=("0.0" "0.2" "0.4" "0.6")
#LAMBDAS=("1" "5" "10" "20")
LAMBDAS=("20")

EPOCHS=200
BATCH_SIZE=128
LR=0.01
SEED=42
OUTPUT_DIR="./results"
DATA_DIR="./data"
WARMUP=10

# Original method defaults
GF_THRESHOLD=0.5
DM_ALPHA=4.0
DM_LAMBDA_U=25.0
DM_T=0.5
DM_P_THRESHOLD=0.5
DISC_EMA_DECAY=0.999
DISC_THRESHOLD_INIT=0.5

# ── Filter by CLI arguments ──────────────────────────────────────────────
FILTER_METHOD="${1:-}"
FILTER_NOISE="${2:-}"

# Build method list
if [ -n "$FILTER_METHOD" ]; then
    ALL_METHODS=("$FILTER_METHOD")
    echo "Filtering to method: $FILTER_METHOD"
else
    #ALL_METHODS=("gmm_filter" "dividemix" "disc" "gmm_filter_crass" "dividemix_crass" "disc_crass")
    ALL_METHODS=("dividemix_crass" "disc_crass")
fi

if [ -n "$FILTER_NOISE" ]; then
    NOISE_RATES=("$FILTER_NOISE")
    echo "Filtering to noise rate: $FILTER_NOISE"
fi

# ── Count total experiments ───────────────────────────────────────────────
TOTAL=0
for METHOD in "${ALL_METHODS[@]}"; do
    case $METHOD in
        gmm_filter|dividemix|disc)
            TOTAL=$((TOTAL + ${#DATASETS[@]} * ${#NOISE_RATES[@]}))
            ;;
        gmm_filter_crass|dividemix_crass|disc_crass)
            TOTAL=$((TOTAL + ${#DATASETS[@]} * ${#NOISE_RATES[@]} * ${#LAMBDAS[@]}))
            ;;
    esac
done

CURRENT=0

echo "═══════════════════════════════════════════════════════════════"
echo "  Running $TOTAL experiments"
echo "  Datasets:    ${DATASETS[*]}"
echo "  Noise rates: ${NOISE_RATES[*]}"
echo "  Methods:     ${ALL_METHODS[*]}"
echo "  Lambdas:     ${LAMBDAS[*]} (for CRASS variants)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Original methods (no lambda sweep) ────────────────────────────────────
for METHOD in "${ALL_METHODS[@]}"; do
    case $METHOD in
        gmm_filter)
            for DATASET in "${DATASETS[@]}"; do
                for NOISE in "${NOISE_RATES[@]}"; do
                    CURRENT=$((CURRENT + 1))
                    echo ""
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    echo "  [$CURRENT/$TOTAL] gmm_filter | $DATASET | noise=$NOISE"
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    python train_filter_loss.py \
                        --dataset $DATASET --noise_rate $NOISE \
                        --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
                        --seed $SEED --output_dir $OUTPUT_DIR --data_dir $DATA_DIR \
                        --weighted_loss --print_freq 20 \
                        --warmup_epochs $WARMUP \
                        --filter_threshold $GF_THRESHOLD
                    echo "  [$CURRENT/$TOTAL] DONE"
                done
            done
            ;;

        dividemix)
            for DATASET in "${DATASETS[@]}"; do
                for NOISE in "${NOISE_RATES[@]}"; do
                    CURRENT=$((CURRENT + 1))
                    echo ""
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    echo "  [$CURRENT/$TOTAL] dividemix | $DATASET | noise=$NOISE"
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    python train_dividemix.py \
                        --dataset $DATASET --noise_rate $NOISE \
                        --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
                        --seed $SEED --output_dir $OUTPUT_DIR --data_dir $DATA_DIR \
                        --weighted_loss --print_freq 20 \
                        --warmup_epochs $WARMUP \
                        --alpha $DM_ALPHA --lambda_u $DM_LAMBDA_U \
                        --T $DM_T --p_threshold $DM_P_THRESHOLD
                    echo "  [$CURRENT/$TOTAL] DONE"
                done
            done
            ;;

        disc)
            for DATASET in "${DATASETS[@]}"; do
                for NOISE in "${NOISE_RATES[@]}"; do
                    CURRENT=$((CURRENT + 1))
                    echo ""
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    echo "  [$CURRENT/$TOTAL] disc | $DATASET | noise=$NOISE"
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    python train_disc.py \
                        --dataset $DATASET --noise_rate $NOISE \
                        --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
                        --seed $SEED --output_dir $OUTPUT_DIR --data_dir $DATA_DIR \
                        --weighted_loss --print_freq 20 \
                        --warmup_epochs $WARMUP \
                        --ema_decay $DISC_EMA_DECAY \
                        --threshold_init $DISC_THRESHOLD_INIT
                    echo "  [$CURRENT/$TOTAL] DONE"
                done
            done
            ;;

        # ── CRASS variants (with lambda sweep) ───────────────────────────

        gmm_filter_crass)
            for DATASET in "${DATASETS[@]}"; do
                for NOISE in "${NOISE_RATES[@]}"; do
                    for LAMBDA in "${LAMBDAS[@]}"; do
                        CURRENT=$((CURRENT + 1))
                        echo ""
                        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                        echo "  [$CURRENT/$TOTAL] gmm_filter_crass | $DATASET | noise=$NOISE | lambda=$LAMBDA"
                        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                        python train_gmm_filter_crass.py \
                            --dataset $DATASET --noise_rate $NOISE \
                            --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
                            --seed $SEED --output_dir $OUTPUT_DIR --data_dir $DATA_DIR \
                            --weighted_loss --print_freq 20 \
                            --warmup_epochs $WARMUP \
                            --lambda_risk $LAMBDA
                        echo "  [$CURRENT/$TOTAL] DONE"
                    done
                done
            done
            ;;

        dividemix_crass)
            for DATASET in "${DATASETS[@]}"; do
                for NOISE in "${NOISE_RATES[@]}"; do
                    for LAMBDA in "${LAMBDAS[@]}"; do
                        CURRENT=$((CURRENT + 1))
                        echo ""
                        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                        echo "  [$CURRENT/$TOTAL] dividemix_crass | $DATASET | noise=$NOISE | lambda=$LAMBDA"
                        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                        python train_dividemix_crass.py \
                            --dataset $DATASET --noise_rate $NOISE \
                            --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
                            --seed $SEED --output_dir $OUTPUT_DIR --data_dir $DATA_DIR \
                            --weighted_loss --print_freq 20 \
                            --warmup_epochs $WARMUP \
                            --lambda_risk $LAMBDA \
                            --alpha $DM_ALPHA --lambda_u $DM_LAMBDA_U --T $DM_T
                        echo "  [$CURRENT/$TOTAL] DONE"
                    done
                done
            done
            ;;

        disc_crass)
            for DATASET in "${DATASETS[@]}"; do
                for NOISE in "${NOISE_RATES[@]}"; do
                    for LAMBDA in "${LAMBDAS[@]}"; do
                        CURRENT=$((CURRENT + 1))
                        echo ""
                        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                        echo "  [$CURRENT/$TOTAL] disc_crass | $DATASET | noise=$NOISE | lambda=$LAMBDA"
                        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                        python train_disc_crass.py \
                            --dataset $DATASET --noise_rate $NOISE \
                            --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
                            --seed $SEED --output_dir $OUTPUT_DIR --data_dir $DATA_DIR \
                            --weighted_loss --print_freq 20 \
                            --warmup_epochs $WARMUP \
                            --lambda_risk $LAMBDA \
                            --ema_decay $DISC_EMA_DECAY \
                            --threshold_init $DISC_THRESHOLD_INIT
                        echo "  [$CURRENT/$TOTAL] DONE"
                    done
                done
            done
            ;;

        *)
            echo "Unknown method: $METHOD"
            exit 1
            ;;
    esac
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All $TOTAL experiments completed!"
echo "  Results saved in: $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════════════"
