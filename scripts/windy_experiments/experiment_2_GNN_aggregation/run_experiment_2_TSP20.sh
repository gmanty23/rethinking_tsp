#!/bin/bash

# --- HARDWARE CONFIGURATION (Safe Mode) ---
# Batch Size 1024 fits your VRAM usage based on previous runs.
BATCH_SIZE=1024
# 3 parallel runs * 6 workers = 18 cores (leaving headroom).
NUM_WORKERS=6

# --- EXPERIMENT SETTINGS ---
EPOCHS=100
PROBLEM="windy_tsp"
GRAPH_SIZE=20 

# FIXED SETTINGS (Based on Exp 1 Results)
ENTROPY=0.05
FEATURE_TYPE="hybrid"

# EXPERIMENTAL VARIABLES (The "Independent Variables")
# forward:  Messages flow A -> B (Standard)
# backward: Messages flow B -> A (Against wind/edge)
# dual:     Messages flow both ways (Bidirectional/Concatenated)
DIRECTION_MODES=("forward" "backward" "dual")

# --- DATA SETTINGS ---
VAL_DATA="data/windy_tsp/windy_tsp20_val.pkl"
VAL_SIZE=5120
EPOCH_SIZE=1280000 
ROLLOUT_SIZE=10240

# Setup directories
LOG_DIR="logs_windy_msg_passing"
mkdir -p $LOG_DIR
mkdir -p data/windy_tsp
mkdir -p results/lkh_windy

# ==================================================
# 1. GENERATE VALIDATION DATA 
# ==================================================
# We check if file exists. If you want to force regeneration, delete the file first.
if [ ! -f "$VAL_DATA" ]; then
    echo "=================================================="
    echo "Generating Validation Data ($VAL_SIZE samples)..."
    echo "=================================================="
    # Note: Ensure generate_windy_tsp.py has the new 8-stat logic!
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
LKH_TARGET="results/lkh_windy/$(basename $VAL_DATA .pkl).pkl"
TEMP_RESULTS_DIR="results/windy_tsp20_exp2_val"

# Check if LKH solution already exists
if [ ! -f "$LKH_TARGET" ]; then
    echo "=================================================="
    echo "Running LKH Solver (Baseline)..."
    echo "=================================================="
    
    # Clean up temp directory from previous runs
    if [ -d "$TEMP_RESULTS_DIR" ]; then rm -rf "$TEMP_RESULTS_DIR"; fi

    # Run the solver
    # Note: Ensure you have the LKH executable installed/compiled
    python eval_baseline.py lkh_windy "$VAL_DATA" -n "$VAL_SIZE" --disable_cache

    # Move and verify results
    # The eval_baseline usually saves to results/windy_tsp/... look for the file
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
# 3. PARALLEL EXECUTION (3 RUNS)
# ==================================================
echo "=================================================="
echo "Launching 3 parallel experiments for Message Passing..."
echo "=================================================="

PIDS=()

for MODE in "${DIRECTION_MODES[@]}"; do
        
    RUN_NAME="msg_${MODE}_${FEATURE_TYPE}_ent${ENTROPY}"
    LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
    
    echo " -> Launching: Mode=$MODE | Feat=$FEATURE_TYPE | Ent=$ENTROPY"

    # Using -u for unbuffered output
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
        --normalization layer \
        --num_workers $NUM_WORKERS \
        --no_progress_bar \
        --entropy_coeff $ENTROPY \
        --node_feature_type $FEATURE_TYPE \
        --gnn_direction_mode $MODE \
        --run_name "$RUN_NAME" \
        > "$LOG_FILE" 2>&1 &
    
    PIDS+=($!)
    sleep 5 # Slight delay to offset initial GPU memory allocation
done

echo "----------------------------------------------------------------"
echo "All processes running. PIDs: ${PIDS[*]}"
echo "Monitor with: tail -f ${LOG_DIR}/*.log"
echo "----------------------------------------------------------------"

# Wait for all processes to finish
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "Message Passing Experiment Complete."