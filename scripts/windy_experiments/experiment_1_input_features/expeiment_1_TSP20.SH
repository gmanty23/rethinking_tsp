#!/bin/bash

# --- CONFIGURATION ---
# Reverting to 512 to ensure 4 parallel runs fit in 16GB VRAM.
BATCH_SIZE=512

EPOCHS=100
PROBLEM="windy_tsp"
GRAPH_SIZE=20 

# Validation Settings (Multiples of 512)
VAL_DATA="data/windy_tsp/windy_tsp20_val.pkl"
VAL_SIZE=5120       # 10 batches

# Training Settings (Multiples of 512)
EPOCH_SIZE=1280000  # 2500 batches
ROLLOUT_SIZE=10240  # 20 batches

mkdir -p logs_exp1
mkdir -p data/windy_tsp
mkdir -p results/lkh_windy

# ==================================================
# 1. GENERATE VALIDATION DATA
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
# 2. PRE-COMPUTE OPTIMAL BASELINE (LKH)
# ==================================================
LKH_TARGET="results/lkh_windy/$(basename $VAL_DATA)"
TEMP_RESULTS_DIR="results/windy_tsp20_val"

if [ ! -f "$LKH_TARGET" ]; then
    echo "LKH target not found. Running solver..."
    if [ -d "$TEMP_RESULTS_DIR" ]; then rm -rf "$TEMP_RESULTS_DIR"; fi

    python eval_baseline.py lkh_windy "$VAL_DATA" -n "$VAL_SIZE" --disable_cache

    GENERATED_FILE=$(find results -name "*lkh_windy.pkl" | head -n 1)
    if [ -f "$GENERATED_FILE" ]; then
        mv "$GENERATED_FILE" "$LKH_TARGET"
    else
        echo "ERROR: Could not find generated LKH file!"
        exit 1
    fi
else
    echo "LKH Data found. Skipping solver."
fi

# ==================================================
# 3. RUN EXPERIMENTS
# ==================================================

# num_workers 4 is the sweet spot for your 24-core CPU
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
      --no_progress_bar "

echo "=================================================="
echo "Starting Parallel Execution on RTX 5080 (Batch 512)"
echo "Logs > logs_exp1/"
echo "=================================================="

# 1. Baseline
echo "Launching Run A: Coords..."
python -u run.py $ARGS \
    --node_feature_type coords \
    --run_name "exp1_coords" \
    > logs_exp1/coords.log 2>&1 &
PID_A=$!
sleep 5 

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
echo "To monitor: tail -f logs_exp1/*.log"
wait $PID_A $PID_B $PID_C $PID_D

echo "Experiment 1 Complete."