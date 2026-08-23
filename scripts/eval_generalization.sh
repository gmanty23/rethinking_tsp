#!/bin/bash

# ==============================================================================
# eval_generalization.sh
#
# Evaluate every checkpoint in the Generalization Ablation folder on multiple
# graph sizes.
# ==============================================================================

set -e

##############################################
# Paths
##############################################

MODEL_DIR="outputs/PAPER_OUTPUTS/4-Dimension3:Mechanism/Generalization_Ablation"
RESULTS_DIR="results/PAPER_RESULTS/GENERALIZATION"


OUT_DIR="results/PAPER_RESULTS/GENERALIZATION"
mkdir -p "$OUT_DIR"

##############################################
# Graph sizes to test
##############################################

GRAPH_SIZES=(
# 20
# 50
# 75
# 100
# 150
# 200
# 300
# 400
# 500
750
1000
)

##############################################
# Validation sizes
# (Reduce for larger graphs)
##############################################

VAL_SIZE=512

##############################################
# Batch sizes
##############################################

declare -A BATCH_SIZES

BATCH_SIZES[20]=256
BATCH_SIZES[50]=256
BATCH_SIZES[75]=256
BATCH_SIZES[100]=128
BATCH_SIZES[150]=64
BATCH_SIZES[200]=32
BATCH_SIZES[300]=16
BATCH_SIZES[400]=8
BATCH_SIZES[500]=4
BATCH_SIZES[750]=2
BATCH_SIZES[1000]=2

##############################################
# Collect models safely
##############################################

MODELS=$(find "$MODEL_DIR" -type f -name "epoch-*.pt" | sort -V)

MODEL_COUNT=$(echo "$MODELS" | grep -v '^$' | wc -l)

if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "ERROR: No checkpoints found."
    exit 1
fi

echo "Found $MODEL_COUNT checkpoints."

##############################################
# Loop over graph sizes
##############################################

for N in "${GRAPH_SIZES[@]}"
do

    DATASET="data/windy_tsp/windy_tsp${N}_val.pkl"

    CSV="${RESULTS_DIR}/generalization_tsp${N}.csv"

    LOG="${RESULTS_DIR}/generalization_tsp${N}.log"

    BATCH=${BATCH_SIZES[$N]}


    echo "" | tee "$LOG"
    echo "======================================================" | tee -a "$LOG"
    echo "Evaluating Graph Size: $N" | tee -a "$LOG"
    echo "Validation set : $VAL_SIZE" | tee -a "$LOG"
    echo "Batch size     : $BATCH" | tee -a "$LOG"
    echo "Beam widths    : 0 10"
    echo "======================================================" | tee -a "$LOG"


    ########################################################
    # Generate validation set if needed
    ########################################################

    if [ ! -f "$DATASET" ]; then

        echo "Generating validation dataset..." | tee -a "$LOG"

        python data/windy_tsp/generate_windy_tsp.py \
            --min_nodes $N \
            --max_nodes $N \
            --num_samples $VAL_SIZE \
            --filename "$DATASET" \
            --alpha 3.0 \
            --max_wind 1.0 \
            --seed 1234 \
            | tee -a "$LOG"

    else

        echo "Validation dataset already exists." | tee -a "$LOG"

    fi

    ########################################################
    # Evaluate
    ########################################################

    python eval_expRIVAL.py \
        "$DATASET" \
        --models $MODELS \
        --csv_out "$CSV" \
        --widths 0 10 \
        --val_size "$VAL_SIZE" \
        --batch_size "$BATCH" \
        | tee -a "$LOG"

done

echo ""
echo "======================================================"
echo "Generalization study complete."
echo "======================================================"
echo "Results saved to:"
echo "$RESULTS_DIR"