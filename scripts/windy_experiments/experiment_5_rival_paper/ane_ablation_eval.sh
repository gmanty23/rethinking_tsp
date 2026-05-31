#!/bin/bash

# Define Paths
VAL_DATA="data/windy_tsp/windy_tsp100_val.pkl"
OUT_CSV="results/ane_ablation_full_2.csv"
LOG_FILE="results/ane_ablation_full_2.txt" 

# Collect ALL checkpoint models across all subfolders
# The wildcard /*/epoch-*.pt looks inside every subfolder and grabs every checkpoint
MODELS=$(ls outputs/windy_tsp_100-100/ane_ablation_chk/*/epoch-*.pt)

echo "==================================================" | tee -a "$LOG_FILE"
echo "Starting Comprehensive GNN Layer Ablation Evaluation..." | tee -a "$LOG_FILE"
# wc -w counts the number of words (paths) returned by the ls command
echo "Total Checkpoints found: $(echo "$MODELS" | wc -w)" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

# Run Python Script
python eval_expRIVAL.py \
    "$VAL_DATA" \
    --models $MODELS \
    --csv_out "$OUT_CSV" \
    --widths 0 10 100  \
    --val_size 1280 \
    --batch_size 128 \
    # --legacy_mode \
    | tee -a "$LOG_FILE"

echo "==================================================" | tee -a "$LOG_FILE"
echo "Done. Results saved in $OUT_CSV" | tee -a "$LOG_FILE"