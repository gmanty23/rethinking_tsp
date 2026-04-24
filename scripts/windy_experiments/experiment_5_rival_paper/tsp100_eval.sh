#!/bin/bash

# Define Paths
VAL_DATA="data/windy_tsp/windy_tsp100_val.pkl"
OUT_CSV="results/gnn100.csv"
LOG_FILE="results/gnn100.txt" 

# Collect all epoch-99 models
# MODELS=$(ls outputs/windy_tsp_100-100/mlp_100/*/epoch-99.pt)
MODELS=$(ls outputs/windy_tsp_100-100/gnn100_tsp100_forward_hybrid_ent0.05_percentage_n0.05_20260423T122319/epoch-49.pt)


echo "Starting Evaluation..." | tee -a "$LOG_FILE"
echo "Models found: $(echo "$MODELS" | wc -l)" | tee -a "$LOG_FILE"

# Run Python Script
# 1. Removed '--no_progress_bar' so you see the bar on screen.
# 2. Added '| tee -a $LOG_FILE' to save the text summary (Gaps, Times) to a file.
python eval_exp1.py \
    "$VAL_DATA" \
    --models $MODELS \
    --csv_out "$OUT_CSV" \
    --widths 0 10 100 1280 \
    --val_size 1280 \
    --batch_size 128 \
    | tee -a "$LOG_FILE"

echo "Done. Results in $OUT_CSV" | tee -a "$LOG_FILE"
