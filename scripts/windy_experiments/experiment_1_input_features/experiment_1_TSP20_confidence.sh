#!/bin/bash

# --- CONFIGURATION ---
# Reverting to 512 to ensure 4 parallel runs fit in 16GB VRAM.
BATCH_SIZE=512

EPOCHS=100
PROBLEM="windy_tsp"
GRAPH_SIZE=20 

# Define the list of Entropy Coefficients to test
# 0.01 = Light exploration
# 0.05 = Medium exploration
# 0.10 = Heavy exploration
ENTROPY_VALUES=(0.2 0.5 1.0)

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
# 1. GENERATE VALIDATION DATA (Run Once)
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
# 2. PRE-COMPUTE OPTIMAL BASELINE (LKH) (Run Once)
# ==================================================
LKH_TARGET="results/lkh_windy/$(basename $VAL_DATA)"
TEMP_RESULTS_DIR="results/windy_tsp20_val"

if [ ! -f "$LKH_TARGET" ]; then
    echo "=================================================="
    echo "Running LKH Solver..."
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
    echo "LKH Target already exists, skipping generation."
fi

# ==================================================
# 3. RUN EXPERIMENTS LOOP
# ==================================================

for ENTROPY_COEFF in "${ENTROPY_VALUES[@]}"; do

    echo ""
    echo "################################################################"
    echo "STARTING BATCH WITH ENTROPY COEFFICIENT: $ENTROPY_COEFF"
    echo "################################################################"
    echo ""

    # Define ARGS inside the loop to use the current ENTROPY_COEFF
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
          --no_progress_bar \
          --entropy_coeff $ENTROPY_COEFF" 

    echo "Logs > logs_exp1/*_ent${ENTROPY_COEFF}.log"

    # 1. Baseline
    echo "Launching Run A: Coords (Ent: $ENTROPY_COEFF)..."
    python -u run.py $ARGS \
        --node_feature_type coords \
        --run_name "exp1_coords_ent${ENTROPY_COEFF}" \
        > logs_exp1/coords_ent${ENTROPY_COEFF}.log 2>&1 &
    PID_A=$!
    sleep 5 

    # 2. Topological
    echo "Launching Run B: Learned (Ent: $ENTROPY_COEFF)..."
    python -u run.py $ARGS \
        --node_feature_type learned \
        --run_name "exp1_learned_ent${ENTROPY_COEFF}" \
        > logs_exp1/learned_ent${ENTROPY_COEFF}.log 2>&1 &
    PID_B=$!
    sleep 5

    # 3. Hybrid
    echo "Launching Run C: Hybrid (Ent: $ENTROPY_COEFF)..."
    python -u run.py $ARGS \
        --node_feature_type hybrid \
        --run_name "exp1_hybrid_ent${ENTROPY_COEFF}" \
        > logs_exp1/hybrid_ent${ENTROPY_COEFF}.log 2>&1 &
    PID_C=$!
    sleep 5

    # 4. Blank
    echo "Launching Run D: Blank (Ent: $ENTROPY_COEFF)..."
    python -u run.py $ARGS \
        --node_feature_type blank \
        --run_name "exp1_blank_ent${ENTROPY_COEFF}" \
        > logs_exp1/blank_ent${ENTROPY_COEFF}.log 2>&1 &
    PID_D=$!

    echo "All processes launched for Entropy $ENTROPY_COEFF."
    echo "PIDs: $PID_A, $PID_B, $PID_C, $PID_D"
    echo "Waiting for this batch to finish before starting next entropy value..."
    
    # CRITICAL: Wait for all 4 to finish before starting the next loop
    wait $PID_A $PID_B $PID_C $PID_D
    
    echo "Finished batch for Entropy $ENTROPY_COEFF"

done

echo "=================================================="
echo "All Experiments (All Entropies) Complete."
echo "=================================================="