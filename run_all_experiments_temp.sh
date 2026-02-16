#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Run all noisy label experiments
# ═══════════════════════════════════════════════════════════════════════════
#
# Datasets:  pneumoniamnist, breastmnist, dermamnist_bin, pathmnist_bin
# Noise:     0.0, 0.2, 0.4, 0.6
# Methods:   baseline, coteaching, gmm_filter, crass, dividemix, unicon, disc
#
# Usage:
#   bash run_all_experiments.sh                 # Run ALL experiments
#   bash run_all_experiments.sh baseline        # Run only baseline
#   bash run_all_experiments.sh coteaching 0.2  # Run coteaching at 20% noise
#
# ═══════════════════════════════════════════════════════════════════════════

set -e

# ── Configuration ─────────────────────────────────────────────────────────
DATASETS=("dermamnist_bin" "pathmnist_bin")
NOISE_RATES=("0.0" "0.2" "0.4" "0.6")
#METHODS=("baseline" "coteaching" "gmm_filter" "crass" "dividemix" "unicon" "disc")
METHODS=( "coteaching" "dividemix" "unicon" "disc"  "crass")

EPOCHS=200
BATCH_SIZE=128
LR=0.01
SEED=42
OUTPUT_DIR="./results"
DATA_DIR="./data"

# Co-teaching defaults
CT_NUM_GRADUAL=10
CT_EXPONENT=1.0

# GMM Filter defaults
GF_WARMUP=10
GF_THRESHOLD=0.5

# DivideMix defaults
DM_WARMUP=10
DM_ALPHA=4.0
DM_LAMBDA_U=25.0
DM_T=0.5
DM_P_THRESHOLD=0.5

# UNICON defaults
UN_WARMUP=10
UN_LAMBDA_C=1.0
UN_TEMPERATURE=0.5
UN_ALPHA=4.0
UN_LAMBDA_U=25.0
UN_T=0.5
UN_THRESHOLD=0.5

# DISC defaults
DISC_WARMUP=10
DISC_EMA_DECAY=0.999
DISC_THRESHOLD_INIT=0.5

# CRASS defaults
CRASS_WARMUP=10
CRASS_LAMBDA=20

# ── Filter by CLI arguments ──────────────────────────────────────────────
FILTER_METHOD="${1:-}"
FILTER_NOISE="${2:-}"

if [ -n "$FILTER_METHOD" ]; then
    METHODS=("$FILTER_METHOD")
    echo "Filtering to method: $FILTER_METHOD"
fi

if [ -n "$FILTER_NOISE" ]; then
    NOISE_RATES=("$FILTER_NOISE")
    echo "Filtering to noise rate: $FILTER_NOISE"
fi

# ── Counter ───────────────────────────────────────────────────────────────
TOTAL=$((${#DATASETS[@]} * ${#NOISE_RATES[@]} * ${#METHODS[@]}))
CURRENT=0

echo "═══════════════════════════════════════════════════════════════"
echo "  Running $TOTAL experiments"
echo "  Datasets:    ${DATASETS[*]}"
echo "  Noise rates: ${NOISE_RATES[*]}"
echo "  Methods:     ${METHODS[*]}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

for DATASET in "${DATASETS[@]}"; do
    for NOISE in "${NOISE_RATES[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            CURRENT=$((CURRENT + 1))
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  [$CURRENT/$TOTAL] $METHOD | $DATASET | noise=$NOISE"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            # Common args
            COMMON="--dataset $DATASET --noise_rate $NOISE --epochs $EPOCHS \
                    --batch_size $BATCH_SIZE --lr $LR --seed $SEED \
                    --output_dir $OUTPUT_DIR --data_dir $DATA_DIR \
                    --weighted_loss --print_freq 20"

            case $METHOD in
                baseline)
                    python train_baseline.py $COMMON
                    ;;
                coteaching)
                    python train_coteaching.py $COMMON \
                        --forget_rate $NOISE \
                        --num_gradual $CT_NUM_GRADUAL \
                        --exponent $CT_EXPONENT
                    ;;
                gmm_filter)
                    python train_filter_loss.py $COMMON \
                        --warmup_epochs $GF_WARMUP \
                        --filter_threshold $GF_THRESHOLD
                    ;;
                dividemix)
                    python train_dividemix.py $COMMON \
                        --warmup_epochs $DM_WARMUP \
                        --alpha $DM_ALPHA \
                        --lambda_u $DM_LAMBDA_U \
                        --T $DM_T \
                        --p_threshold $DM_P_THRESHOLD
                    ;;
                unicon)
                    python train_unicon.py $COMMON \
                        --warmup_epochs $UN_WARMUP \
                        --lambda_c $UN_LAMBDA_C \
                        --temperature $UN_TEMPERATURE \
                        --alpha $UN_ALPHA \
                        --lambda_u $UN_LAMBDA_U \
                        --T $UN_T \
                        --threshold $UN_THRESHOLD
                    ;;
                crass)
                    python train_crass.py $COMMON \
                        --warmup_epochs $CRASS_WARMUP \
                        --lambda_risk $CRASS_LAMBDA
                    ;;
                disc)
                    python train_disc.py $COMMON \
                        --warmup_epochs $DISC_WARMUP \
                        --ema_decay $DISC_EMA_DECAY \
                        --threshold_init $DISC_THRESHOLD_INIT
                    ;;
                *)
                    echo "Unknown method: $METHOD"
                    exit 1
                    ;;
            esac

            echo "  [$CURRENT/$TOTAL] DONE: $METHOD | $DATASET | noise=$NOISE"
        done
    done
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All $TOTAL experiments completed!"
echo "  Results saved in: $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════════════"
