#!/bin/bash

# --- CONFIGURATION ---
# We use a moderate batch size to ensure 3 runs fit comfortably in 16GB VRAM
EPOCHS=100
BATCH_SIZE=512 
PROBLEM="windy_tsp"
TRAIN_DATA="data/windy_tsp/windy_tsp20_train.pkl"
VAL_DATA="data/windy_tsp/windy_tsp20_val.pkl"
GRAPH_SIZE=20 

# Adjusted sizes to be multiples of 512
VAL_SIZE=5120      
ROLLOUT_SIZE=10240 

# Create a logs directory so output doesn't clutter the screen
mkdir -p logs_exp1

# --- COMMON SETTINGS ---
# --num_workers 4 helps the CPU keep up with the GPU
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
      --num_workers 6"

echo "=================================================="
echo "Starting Parallel Execution on RTX 5080"
echo "Logs are being saved to the 'logs_exp1' folder"
echo "=================================================="

# 1. Baseline
echo "Launching Run A: Coords..."
python run.py $ARGS \
    --node_feature_type coords \
    --run_name "exp1_coords" \
    > logs_exp1/coords.log 2>&1 &
PID_A=$!

# 2. Topological
echo "Launching Run B: Learned..."
python run.py $ARGS \
    --node_feature_type learned \
    --run_name "exp1_learned" \
    > logs_exp1/learned.log 2>&1 &
PID_B=$!

# 3. Hybrid
echo "Launching Run C: Hybrid..."
python run.py $ARGS \
    --node_feature_type hybrid \
    --run_name "exp1_hybrid" \
    > logs_exp1/hybrid.log 2>&1 &
PID_C=$!

echo "All processes launched. PIDs: $PID_A, $PID_B, $PID_C"
echo "You can watch progress by running: tail -f logs_exp1/*.log"

# Wait for all background processes to finish
wait $PID_A $PID_B $PID_C

echo "Experiment 1 Complete."