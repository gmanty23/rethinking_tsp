#!/bin/bash

# ==================================================
# SMART AUTO-RESUME PIPELINE (Target: 1000 Epochs)
# ==================================================
BATCH_SIZE=512         
NUM_WORKERS=6          
MAX_PARALLEL_JOBS=3    

# --- EXPERIMENT SETTINGS ---
TARGET_TOTAL_EPOCHS=1000 
PROBLEM="windy_tsp"
ENTROPY=0.05
GRAPH_SIZE=100

# Standardized Sizes
VAL_SIZE=2048      
EPOCH_SIZE=128000  
ROLLOUT_SIZE=10240 

LOG_DIR="logs_windy_tsp_smart_resume"
mkdir -p "$LOG_DIR"
VAL_DATA="data/windy_tsp/windy_tsp${GRAPH_SIZE}_val.pkl"
LOG_FILE="${LOG_DIR}/master_resume.log"

echo "==================================================" | tee -a "$LOG_FILE"
echo " LAUNCHING SMART MULTI-MODEL RESUME" | tee -a "$LOG_FILE"
echo " Target: $TARGET_TOTAL_EPOCHS Epochs" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

# Base directory where all the model folders live
BASE_DIR="outputs/windy_tsp_100-100/ane_ablation_chk"

# Loop through every subfolder in the base directory
for FOLDER in "$BASE_DIR"/*; do
    if [ -d "$FOLDER" ]; then
        MODEL_NAME=$(basename "$FOLDER")
        
        # 1. Find the highest epoch checkpoint using version sort (-v)
        HIGHEST_CKPT=$(ls -v "$FOLDER"/epoch-*.pt 2>/dev/null | tail -n 1)
        
        if [ -z "$HIGHEST_CKPT" ]; then
            echo "[SKIP] No .pt files found in $MODEL_NAME" | tee -a "$LOG_FILE"
            continue
        fi
        
        # 2. Extract the exact epoch number from the filename
        CURRENT_EPOCH=$(basename "$HIGHEST_CKPT" | sed 's/epoch-//' | sed 's/\.pt//')
        
        # Check if it is already finished
        if [ "$CURRENT_EPOCH" -ge "$((TARGET_TOTAL_EPOCHS - 1))" ]; then
            echo "[SKIP] $MODEL_NAME is already at or beyond epoch $CURRENT_EPOCH" | tee -a "$LOG_FILE"
            continue
        fi

        # --- THE MATH FIX: Calculate exactly how many epochs are remaining ---
        EPOCHS_TO_RUN=$((TARGET_TOTAL_EPOCHS - CURRENT_EPOCH - 1))

        # 3. Read the model's DNA from its args.json
        ARGS_FILE="$FOLDER/args.json"
        
        EMB_TYPE=$(python -c "import json; print(json.load(open('$ARGS_FILE')).get('node_embedding_type', 'original'))")
        FEAT_TYPE=$(python -c "import json; print(json.load(open('$ARGS_FILE')).get('node_feature_type', 'coords'))")
        NEIGHBORS=$(python -c "import json; print(json.load(open('$ARGS_FILE')).get('neighbors', 20))")
        USE_WIND=$(python -c "import json; print(json.load(open('$ARGS_FILE')).get('use_wind', False))")

        # 4. Concurrency Control: Wait if we have too many jobs running
        while [ $(jobs -r -p | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
            sleep 5
        done

        # 5. Build and launch the command
        RUN_LOG="${LOG_DIR}/resume_${MODEL_NAME}.log"
        echo " -> Resuming $MODEL_NAME from epoch $CURRENT_EPOCH (Adding $EPOCHS_TO_RUN more)..." | tee -a "$LOG_FILE"
        
        # Base command string
        CMD="python -u run.py \
            --problem $PROBLEM \
            --min_size $GRAPH_SIZE \
            --max_size $GRAPH_SIZE \
            --n_epochs $EPOCHS_TO_RUN \
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
            --neighbors $NEIGHBORS \
            --node_embedding_type $EMB_TYPE \
            --node_feature_type $FEAT_TYPE \
            --resume $HIGHEST_CKPT \
            --run_name resume_$MODEL_NAME"
            
        # Add the wind flag conditionally
        if [ "$USE_WIND" = "True" ]; then
            CMD="$CMD --use_wind"
        fi
        
        # Launch it in the background
        $CMD > "$RUN_LOG" 2>&1 &
        
        # Small sleep to prevent log collision right at startup
        sleep 2
    fi
done

wait
echo "==================================================" | tee -a "$LOG_FILE"
echo " ALL RESUME JOBS COMPLETED" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"


# tail -f logs_windy_tsp_smart_resume/*.log