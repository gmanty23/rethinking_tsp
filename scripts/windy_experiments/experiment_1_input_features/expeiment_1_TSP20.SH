#!/bin/bash

# --- CONFIGURATION ---
EPOCHS=100
BATCH_SIZE=4096 
PROBLEM="windy_tsp"
GRAPH_SIZE=20 

# Validation Settings
VAL_DATA="data/windy_tsp/windy_tsp20_val.pkl"
VAL_SIZE=5120      

# Training Settings
EPOCH_SIZE=1280000 
ROLLOUT_SIZE=10240 

mkdir -p logs_exp1
mkdir -p data/windy_tsp
mkdir -p results/lkh_windy

# ==================================================
# 1. GENERATE VALIDATION DATA (ALWAYS)
# ==================================================
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

# ==================================================
# 2. PRE-COMPUTE OPTIMAL BASELINE (LKH) - ALWAYS
# ==================================================

LKH_TARGET="results/lkh_windy/$(basename $VAL_DATA)"
TEMP_RESULTS_DIR="results/windy_tsp20_val"

echo "=================================================="
echo "Generating Optimal Solutions (LKH Baseline)..."
echo "FORCING FRESH RUN: Deleting old files and caches."
echo "This WILL take time (approx 10-20 mins with 24 cores)."
echo "=================================================="

# 1. CLEANUP: Delete previous target and temp cache to force re-calculation
rm -f "$LKH_TARGET"
if [ -d "$TEMP_RESULTS_DIR" ]; then
    rm -rf "$TEMP_RESULTS_DIR"
fi

# 2. Run LKH
# --disable_cache is redundant if we deleted the folder, but kept for safety.
python eval_baseline.py lkh_windy "$VAL_DATA" -n "$VAL_SIZE" --disable_cache

# 3. Move the result
GENERATED_FILE=$(find results -name "*lkh_windy.pkl" | head -n 1)

if [ -f "$GENERATED_FILE" ]; then
    echo "Moving baseline file from $GENERATED_FILE to $LKH_TARGET"
    mv "$GENERATED_FILE" "$LKH_TARGET"
else
    echo "ERROR: Could not find generated LKH file!"
    exit 1
fi

# ==================================================
# 3. RUN EXPERIMENTS
# ==================================================

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
      --num_workers 2 \
      --no_progress_bar "

echo "=================================================="
echo "Starting Parallel Execution on RTX 5080"
echo "=================================================="

# 1. Baseline
echo "Launching Run A: Coords..."
python -u run.py $ARGS \
    --node_feature_type coords \
    --run_name "exp1_coords" \
    > logs_exp1/coords.log 2>&1 &
PID_A=$!
sleep 5  # Stagger start to stabilize memory allocation

# 2. Topological
echo "Launching Run B: Learned..."
python -u run.py $ARGS \
    --node_feature_type learned \
    --run_name "exp1_learned" \
    > logs_exp1/learned.log 2>&1 &
PID_B=$!
sleep 5

# 3. Hybrid
echo "Launching Run C: Hybrid..."
python -u run.py $ARGS \
    --node_feature_type hybrid \
    --run_name "exp1_hybrid" \
    > logs_exp1/hybrid.log 2>&1 &
PID_C=$!
sleep 5

# 4. Blank
echo "Launching Run D: Blank..."
python -u run.py $ARGS \
    --node_feature_type blank \
    --run_name "exp1_blank" \
    > logs_exp1/blank.log 2>&1 &
PID_D=$!

echo "All processes launched. PIDs: $PID_A, $PID_B, $PID_C, $PID_D"
wait $PID_A $PID_B $PID_C $PID_D

echo "Experiment 1 Complete."