#!/bin/bash

# --- CONFIGURATION ---
EPOCHS=100
BATCH_SIZE=4096 
PROBLEM="windy_tsp"
TRAIN_DATA="data/windy_tsp/windy_tsp20_train.pkl"
VAL_DATA="data/windy_tsp/windy_tsp20_val.pkl"
GRAPH_SIZE=20 

# Adjusted sizes
VAL_SIZE=5120      
ROLLOUT_SIZE=10240 

mkdir -p logs_exp1


# --- COMMON SETTINGS ---
# OPTIMIZATION CHANGE: Reduced num_workers to 2 per run (Total 8 workers)
# This prevents CPU choking while running 4 parallel experiments.
ARGS="--problem $PROBLEM \
      --min_size $GRAPH_SIZE \
      --max_size $GRAPH_SIZE \
      --n_epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --epoch_size 1280000 \
      --train_dataset $TRAIN_DATA \
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