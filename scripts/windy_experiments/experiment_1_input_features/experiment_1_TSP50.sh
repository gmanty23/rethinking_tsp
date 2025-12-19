#!/bin/bash

# --- CONFIGURATION for TSP50 ---

# MEMORY SAFETY: 
# TSP50 requires significantly more VRAM than TSP20.
# Since we are running 4 PARALLEL processes, we reduce Batch Size.
# 128 is a safe conservative bet. 256 might risk OOM.
BATCH_SIZE=128

EPOCHS=100
PROBLEM="windy_tsp"
GRAPH_SIZE=50 

# Validation Settings (Multiples of 128)
VAL_DATA="data/windy_tsp/windy_tsp50_val.pkl"
VAL_SIZE=2560       # 20 batches of 128

# Training Settings (Multiples of 128)
# We slightly reduce total epoch size because TSP50 is slower to process.
EPOCH_SIZE=640000   # 5000 batches
ROLLOUT_SIZE=10240  # Still divisible by 128

mkdir -p logs_exp1_tsp50
mkdir -p data/windy_tsp
mkdir -p results/lkh_windy

# ==================================================
# 1. GENERATE VALIDATION DATA (If not exists)
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
    echo "Validation data already exists at $VAL_DATA"
fi

# ==================================================
# 2. PRE-COMPUTE OPTIMAL BASELINE (LKH)
# ==================================================
LKH_TARGET="results/lkh_windy/$(basename $VAL_DATA)"
TEMP_RESULTS_DIR="results/windy_tsp${GRAPH_SIZE}_val"

# ONLY RUN LKH IF MISSING (TSP50 takes time to solve!)
if [ ! -f "$LKH_TARGET" ]; then
    echo "=================================================="
    echo "Running LKH Solver for Baseline..."
    echo "=================================================="
    
    # Clean up temp directory
    if [ -d "$TEMP_RESULTS_DIR" ]; then rm -rf "$TEMP_RESULTS_DIR"; fi

    # Run the solver
    python eval_baseline.py lkh_windy "$VAL_DATA" -n "$VAL_SIZE" --disable_cache

    # Move and verify results
    GENERATED_FILE=$(find results -name "*lkh_windy.pkl" | head -n 1)

    if [ -f "$GENERATED_FILE" ]; then
        mv "$GENERATED_FILE" "$LKH_TARGET"
        echo "LKH Solutions successfully generated and moved to $LKH_TARGET"
    else
        echo "ERROR: Could not find generated LKH file!"
        exit 1
    fi
else
    echo "LKH Baseline found ($LKH_TARGET). Skipping generation."
fi

# ==================================================
# 3. RUN EXPERIMENTS
# ==================================================

# num_workers 4 is optimal for your CPU
ARGS="--problem $PROBLEM \
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
      --num_workers 4 \
      --no_progress_bar"

echo "=================================================="
echo "Starting Parallel Execution for TSP50 on RTX 5080"
echo "Logs > logs_exp1_tsp50/"
echo "=================================================="

# 1. Baseline
echo "Launching Run A: Coords..."
python -u run.py $ARGS \
    --node_feature_type coords \
    --run_name "tsp50_coords" \
    > logs_exp1_tsp50/coords.log 2>&1 &
PID_A=$!
sleep 10  # Increased sleep for larger model initialization

# 2. Topological
echo "Launching Run B: Learned..."
python -u run.py $ARGS \
    --node_feature_type learned \
    --run_name "tsp50_learned" \
    > logs_exp1_tsp50/learned.log 2>&1 &
PID_B=$!
sleep 10

# 3. Hybrid
echo "Launching Run C: Hybrid..."
python -u run.py $ARGS \
    --node_feature_type hybrid \
    --run_name "tsp50_hybrid" \
    > logs_exp1_tsp50/hybrid.log 2>&1 &
PID_C=$!
sleep 10

# 4. Blank
echo "Launching Run D: Blank..."
python -u run.py $ARGS \
    --node_feature_type blank \
    --run_name "tsp50_blank" \
    > logs_exp1_tsp50/blank.log 2>&1 &
PID_D=$!

echo "All processes launched. PIDs: $PID_A, $PID_B, $PID_C, $PID_D"
echo "To monitor: tail -f logs_exp1_tsp50/*.log"
wait $PID_A $PID_B $PID_C $PID_D

echo "TSP50 Experiment Complete."