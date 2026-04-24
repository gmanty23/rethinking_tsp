#!/bin/bash

# ==================================================
# FINAL RESUME RUN: HYBRID MODEL (Epochs 50 to 300)
# ==================================================
BATCH_SIZE=1024        
NUM_WORKERS=20

# --- EXPERIMENT SETTINGS ---
NEW_MAX_EPOCHS=900  
PROBLEM="windy_tsp"
ENTROPY=0.05
GRAPH_SIZE=100
FEAT="hybrid"

# Standardized Sizes
VAL_SIZE=2048      
EPOCH_SIZE=128000  
ROLLOUT_SIZE=10240 

# --- EXACT HARDCODED PATH TO CHECKPOINT ---
LOAD_PATH="outputs/windy_tsp_100-100/final6_mlp100_tsp100_hybrid_ent0.05_epochs650_to_750_20260423T010955/epoch-99.pt"

# --- FINAL NAMING CONVENTIONS ---
LOG_DIR="logs_windy_tsp_mlp100_final7"
RUN_NAME="final7_mlp100_tsp100_hybrid_ent0.05_epochs650_to_750"
mkdir -p $LOG_DIR

RUN_LOG="${LOG_DIR}/${RUN_NAME}.log"
VAL_DATA="data/windy_tsp/windy_tsp${GRAPH_SIZE}_val.pkl"

echo "=================================================="
echo " LAUNCHING FINAL RESUME: $RUN_NAME"
echo "=================================================="

# Safety check: Did you copy/paste the path correctly?
if [ ! -f "$LOAD_PATH" ]; then
    echo "[ERROR] Checkpoint not found exactly at:"
    echo "$LOAD_PATH"
    exit 1
fi
echo " Successfully found checkpoint!"
echo "=================================================="
echo ">>> Launching training in the background..."

# LAUNCH TRAINING
python -u run.py \
    --problem $PROBLEM \
    --min_size $GRAPH_SIZE \
    --max_size $GRAPH_SIZE \
    --n_epochs $NEW_MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --epoch_size $EPOCH_SIZE \
    --val_datasets $VAL_DATA \
    --val_size $VAL_SIZE \
    --rollout_size $ROLLOUT_SIZE \
    --model attention \
    --encoder mlp \
    --normalization layer \
    --num_workers $NUM_WORKERS \
    --no_progress_bar \
    --entropy_coeff $ENTROPY \
    --node_feature_type $FEAT \
    --load_path "$LOAD_PATH" \
    --run_name "$RUN_NAME" > "$RUN_LOG" 2>&1 &

echo ">>> Successfully launched!"
echo ">>> Monitor progress with: tail -f $RUN_LOG"
echo "=================================================="