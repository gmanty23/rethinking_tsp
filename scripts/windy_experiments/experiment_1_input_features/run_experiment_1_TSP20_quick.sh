#!/bin/bash

# Experiment 1: Information Study (QUICK TEST)

# --- CONFIGURATION ---
EPOCHS=5  # Reduced from 50 to 5
BATCH_SIZE=128
PROBLEM="windy_tsp"
TRAIN_DATA="data/windy_tsp/windy_tsp20_train.pkl"
VAL_DATA="data/windy_tsp/windy_tsp20_val.pkl"

GRAPH_SIZE=20 
VAL_SIZE=1280
ROLLOUT_SIZE=10240

# --- COMMON SETTINGS ---
ARGS="--problem $PROBLEM \
      --min_size $GRAPH_SIZE \
      --max_size $GRAPH_SIZE \
      --n_epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --epoch_size 12800 \
      --train_dataset $TRAIN_DATA \
      --val_datasets $VAL_DATA \
      --val_size $VAL_SIZE \
      --rollout_size $ROLLOUT_SIZE \
      --model attention \
      --encoder gnn \
      --gated \
      --gnn_direction_mode forward \
      --normalization layer \
      --no_progress_bar" # Keep this to reduce spam, but check logs

# 1. Baseline: Standard Coordinates
echo "Starting Run A: Coords (Baseline)..."
python run.py $ARGS \
    --node_feature_type coords \
    --run_name "test_coords_5ep"

# 2. Topological: Learned Stats Only
echo "Starting Run B: Learned (Topological)..."
python run.py $ARGS \
    --node_feature_type learned \
    --run_name "test_learned_5ep"

# 3. Hybrid: The Proposed Solution
echo "Starting Run C: Hybrid (Geometry + Physics)..."
python run.py $ARGS \
    --node_feature_type hybrid \
    --run_name "test_hybrid_5ep"

echo "Quick Test Complete. Check TensorBoard."