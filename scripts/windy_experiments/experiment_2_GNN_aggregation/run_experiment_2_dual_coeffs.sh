#!/bin/bash

# --- HARDWARE CONFIGURATION ---
# Batch Size 1024 fits your VRAM usage.
BATCH_SIZE=1024
# 3 parallel runs * 6 workers = 18 cores.
NUM_WORKERS=6

# --- EXPERIMENT SETTINGS ---
EPOCHS=100
PROBLEM="windy_tsp"
GRAPH_SIZE=20 

# FIXED SETTINGS
FEATURE_TYPE="hybrid"
MODE="dual"  # We are only testing Dual mode now

# VARIABLE SETTINGS (The "Independent Variables")
# Testing higher entropy to prevent the "Dual" mode from overfitting/memorizing
ENTROPY_VALUES=(0.1 0.5 1.0)

# --- DATA SETTINGS ---
VAL_DATA="data/windy_tsp/windy_tsp20_val.pkl"
VAL_SIZE=5120
EPOCH_SIZE=1280000 
ROLLOUT_SIZE=10240

# Setup directories
LOG_DIR="logs_windy_dual_entropy"
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
LKH_TARGET="results/lkh_windy/$(basename $VAL_DATA .pkl).pkl"
TEMP_RESULTS_DIR="results/windy_tsp20_dual_val"

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
echo "Launching 3 parallel experiments for Dual Mode Entropy..."
echo "=================================================="

PIDS=()

for ENTROPY in "${ENTROPY_VALUES[@]}"; do
        
    RUN_NAME="dual_${FEATURE_TYPE}_ent${ENTROPY}"
    LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
    
    echo " -> Launching: Mode=$MODE | Feat=$FEATURE_TYPE | Ent=$ENTROPY"

    # NOTE: If you want to enable KNN sparsification, uncomment the lines below:
    # --neighbors 20 \
    # --knn_strat percentage \

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
    sleep 5 
done

echo "----------------------------------------------------------------"
echo "All processes running. PIDs: ${PIDS[*]}"
echo "Monitor with: tail -f ${LOG_DIR}/*.log"
echo "----------------------------------------------------------------"

for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "Dual Mode Entropy Experiment Complete."