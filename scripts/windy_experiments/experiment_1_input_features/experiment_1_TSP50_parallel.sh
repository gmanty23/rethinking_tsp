#!/bin/bash

# --- HARDWARE CONFIGURATION (Optimized for RTX 5080 + 24 Cores) ---
# Running 2 in parallel: We split workers to avoid CPU bottleneck
BATCH_SIZE=256
NUM_WORKERS=6  # Reduced from 10 -> 5 to allow 2 runs to share 24 cores effectively

# --- EXPERIMENT SETTINGS ---
EPOCHS=100
PROBLEM="windy_tsp"
GRAPH_SIZE=50 
ENTROPY_VALUES=(0.01 0.05 0.1 0.2 1.0)
FEATURE_TYPES=("coords" "learned" "hybrid")

# --- DATA SETTINGS (TSP-50) ---
VAL_DATA="data/windy_tsp/windy_tsp50_val.pkl"

# Adjusted sizes for TSP-50 training speed
VAL_SIZE=2560
EPOCH_SIZE=640000 
ROLLOUT_SIZE=2560

# Setup directories
LOG_DIR="logs_tsp50_final"
mkdir -p $LOG_DIR
mkdir -p data/windy_tsp
mkdir -p results/lkh_windy

# ==================================================
# 1. GENERATE VALIDATION DATA 
# ==================================================
if [ ! -f "$VAL_DATA" ]; then
    echo "=================================================="
    echo "Generating Validation Data ($VAL_SIZE samples)..."
    echo "=================================================="
    python data/windy_tsp/generate_windy_tsp.py \
        --min_nodes $GRAPH_SIZE \
        --max_nodes $GRAPH_SIZE \
        --num_samples $VAL_SIZE \
        --filename $VAL_DATA \
        --alpha 3.0 \
        --max_wind 1.0 \
        --seed 1234
else
    echo "Validation data $VAL_DATA already exists."
fi

# ==================================================
# 2. PRE-COMPUTE OPTIMAL BASELINE (LKH)
# ==================================================
LKH_TARGET="results/lkh_windy/$(basename $VAL_DATA .pkl)_lkh.pkl"
TEMP_RESULTS_DIR="results/windy_tsp50_val_fully_connected"

if [ ! -f "$LKH_TARGET" ]; then
    echo "=================================================="
    echo "Running LKH Solver (Baseline)..."
    echo "=================================================="
    
    if [ -d "$TEMP_RESULTS_DIR" ]; then rm -rf "$TEMP_RESULTS_DIR"; fi

    python eval_baseline.py lkh_windy "$VAL_DATA" -n "$VAL_SIZE" --disable_cache

    GENERATED_FILE=$(find results -name "*lkh_windy.pkl" | head -n 1)

    if [ -f "$GENERATED_FILE" ]; then
        mv "$GENERATED_FILE" "$LKH_TARGET"
        echo "LKH Solutions successfully generated and moved to $LKH_TARGET"
    else
        echo "WARNING: Could not find generated LKH file! Proceeding without baseline comparison..."
    fi
else
    echo "LKH Target already exists, skipping generation."
fi

# ==================================================
# 3. PARALLEL EXECUTION (2 JOBS)
# ==================================================
echo "=================================================="
echo "Starting Execution: Parallel Runs (Max 2 simultaneous)"
echo "=================================================="

for ENTROPY in "${ENTROPY_VALUES[@]}"; do
    for TYPE in "${FEATURE_TYPES[@]}"; do
        
        if [[ ("$TYPE" == "coords" && ( "$ENTROPY" == "0.01" || "$ENTROPY" == "0.05" || "$ENTROPY" == "0.1" )) || 
              ("$TYPE" == "hybrid" && ( "$ENTROPY" == "0.01" || "$ENTROPY" == "0.05" || "$ENTROPY" == "0.1" )) || 
              ("$TYPE" == "learned" && ( "$ENTROPY" == "0.01" || "$ENTROPY" == "0.05" || "$ENTROPY" == "0.1" )) ]]; then
            echo "Skipping $TYPE with Entropy $ENTROPY"
            continue
        fi  
        
        RUN_NAME="tsp50_fully_connected_${TYPE}_ent${ENTROPY}"
        LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

        echo " -> Launching: $TYPE | Entropy: $ENTROPY"

        #show the tail command for monitoring
        echo "    To monitor progress: tail -f $LOG_FILE"

        # Launch in background
        python -u run.py \
            --problem $PROBLEM \
            --min_size $GRAPH_SIZE \
            --max_size $GRAPH_SIZE \
            --n_epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --epoch_size $EPOCH_SIZE \
            --val_datasets $VAL_DATA \
            --val_size $VAL_SIZE \
            --rollout_size $ROLLOUT_SIZE \
            --model attention \
            --encoder gnn \
            --gated \
            --gnn_direction_mode forward \
            --normalization layer \
            --num_workers $NUM_WORKERS \
            --no_progress_bar \
            --entropy_coeff $ENTROPY \
            --node_feature_type $TYPE \
            --run_name "$RUN_NAME" \
            --neighbors 50 \
            > "$LOG_FILE" 2>&1 &
        
        # Limit to 2 parallel jobs
        if [[ $(jobs -r -p | wc -l) -ge 2 ]]; then
            wait -n
        fi
    done
done

# Wait for the last remaining background job to finish
wait
echo "All Experiments Complete."