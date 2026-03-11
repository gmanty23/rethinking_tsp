#!/bin/bash

# Define Paths
VAL_DATA="data/windy_tsp/windy_tsp25_val.pkl"
OUT_CSV="results/experiment1_compknn_TSP20_neighbors_2.csv"
LOG_FILE="results/eval_run_log_TSP20_compknn.txt"  

# Collect all epoch-99 models
MODELS=$(ls outputs/knn_neighbors_2/*/epoch-99.pt)

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
