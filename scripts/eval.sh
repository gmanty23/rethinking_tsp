#!/bin/bash

# ==============================================================================
# eval_ablations.sh
#
# UPDATE: whole file
#
# Comprehensive Evaluation Pipeline for Asymmetric/Windy TSP.
# Scans a directory for all saved model checkpoints and evaluates them against 
# the specified validation dataset.
# ==============================================================================

# 1. Define Paths
VAL_DATA="data/windy_tsp/windy_tsp100_val.pkl"
OUT_CSV="results/PAPER_RESULTS/V3-ANE_AAFM_ABLATION.csv"
LOG_FILE="results/nab_ablation_neighbors.txt" 
MODEL_DIR="outputs/FINAL_PAPER/V3-ANE_AAFM_ABLATION"

# Ensure output directories exist
mkdir -p "$(dirname "$OUT_CSV")"
mkdir -p "$(dirname "$LOG_FILE")"

# 2. Collect ALL checkpoint models across all subfolders safely
# Using 'find' prevents the "Argument list too long" bash crash when loading hundreds of models
MODELS=$(find "$MODEL_DIR" -type f -name "epoch-*.pt" | sort -V)

# 3. Guard Clause: Ensure models were actually found
MODEL_COUNT=$(echo "$MODELS" | grep -v '^$' | wc -l)

if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "CRITICAL ERROR: No checkpoints found in $MODEL_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

echo "==================================================" | tee -a "$LOG_FILE"
echo "Starting Comprehensive Evaluation..." | tee -a "$LOG_FILE"
echo "Total Checkpoints found: $MODEL_COUNT" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

# 4. Run Python Evaluation Script
# Passing the safe $MODELS list to the custom evaluation pipeline
python eval_expRIVAL.py \
    "$VAL_DATA" \
    --models $MODELS \
    --csv_out "$OUT_CSV" \
    --widths 0 10 100 \
    --val_size 1280 \
    --batch_size 128 \
    | tee -a "$LOG_FILE"

echo "==================================================" | tee -a "$LOG_FILE"
echo "Done. Results saved in $OUT_CSV" | tee -a "$LOG_FILE"