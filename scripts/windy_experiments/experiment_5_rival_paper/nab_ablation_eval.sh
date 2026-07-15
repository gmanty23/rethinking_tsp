#!/bin/bash

# Define Paths
VAL_DATA="data/windy_tsp/windy_tsp100_val.pkl"
OUT_CSV="results/PAPER_RESULTS/V3-ANE_AAFM_ABLATION.csv"
LOG_FILE="results/nab_ablation_neighbors.txt" 

# Collect ALL checkpoint models across all subfolders
# The wildcard /*/epoch-*.pt looks inside every subfolder and grabs every checkpoint
MODELS=$(ls outputs/FINAL_PAPER/V3-ANE_AAFM_ABLATION/*/epoch-*.pt)

echo "==================================================" | tee -a "$LOG_FILE"
echo "Starting Comprehensive NAB Ablation Evaluation..." | tee -a "$LOG_FILE"
# wc -w counts the number of words (paths) returned by the ls command
echo "Total Checkpoints found: $(echo "$MODELS" | wc -w)" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

# Run Python Script      # --legacy_mode \
python eval_expRIVAL.py \
    "$VAL_DATA" \
    --models $MODELS \
    --csv_out "$OUT_CSV" \
    --widths 0 10 100  \
    --val_size 1280 \
    --batch_size 128 \
    | tee -a "$LOG_FILE"

echo "==================================================" | tee -a "$LOG_FILE"
echo "Done. Results saved in $OUT_CSV" | tee -a "$LOG_FILE"