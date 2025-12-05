#!/bin/bash

# Experiment 1: Information Study
# Comparing node_feature_types: coords vs learned vs hybrid

# --- CONFIGURATION ---
EPOCHS=50
BATCH_SIZE=128
PROBLEM="windy_tsp"
TRAIN_DATA="data/windy_tsp/windy_tsp20_train.pkl"
VAL_DATA="data/windy_tsp/windy_tsp20_val.pkl"

# Explicitly set size to 20 so output folders are named correctly
GRAPH_SIZE=20 

# FIX: Set sizes that are multiples of 128 (BATCH_SIZE)
# 1280 / 128 = 10 batches
# 10240 / 128 = 80 batches
VAL_SIZE=1280
ROLLOUT_SIZE=10240

# --- COMMON SETTINGS ---
ARGS="--problem $PROBLEM \
      --min_size $GRAPH_SIZE \
      --max_size $GRAPH_SIZE \
      --n_epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --train_dataset $TRAIN_DATA \
      --val_datasets $VAL_DATA \
      --val_size $VAL_SIZE \
      --rollout_size $ROLLOUT_SIZE \
      --model attention \
      --encoder gnn \
      --gated \
      --gnn_direction_mode forward \
      --normalization layer \
      --no_progress_bar"

# 1. Baseline: Standard Coordinates (Blind to Wind)
echo "Starting Run A: Coords (Baseline)..."
python run.py $ARGS \
    --node_feature_type coords \
    --run_name "exp1_coords"

# 2. Topological: Learned Stats Only (Blind to Geometry)
echo "Starting Run B: Learned (Topological)..."
python run.py $ARGS \
    --node_feature_type learned \
    --run_name "exp1_learned"

# 3. Hybrid: The Proposed Solution
echo "Starting Run C: Hybrid (Geometry + Physics)..."
python run.py $ARGS \
    --node_feature_type hybrid \
    --run_name "exp1_hybrid"

echo "Experiment 1 Complete."