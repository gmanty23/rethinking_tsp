#!/bin/bash

# --- HARDWARE CONFIGURATION ---
# Manteniendo Batch Size 1024 para asegurar que 5 ejecuciones paralelas quepan en la VRAM.
BATCH_SIZE=1024
# 4 Workers por run * 5 runs = 20 cores. Deja 4 cores libres para el SO y overhead de Python.
NUM_WORKERS=6 

# --- EXPERIMENT SETTINGS ---
EPOCHS=100
PROBLEM="windy_tsp"
GRAPH_SIZE=20 
# Solo modelo coords
FEATURE_TYPES=("coords")

# --- SPLIT GROUPS ---
# Group 1: Run these 3 first
BATCH_1=(0.01 0.05 0.1)
# Group 2: Run these 2 after Group 1 finishes
BATCH_2=(0.2 1.0)

# --- DATA SETTINGS ---
VAL_DATA="data/windy_tsp/windy_tsp20_val.pkl"
VAL_SIZE=5120
EPOCH_SIZE=1280000 
ROLLOUT_SIZE=10240

# Directorios de logs y resultados
LOG_DIR="logs_windy_coords_entropy"
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
LKH_TARGET="results/lkh_windy/$(basename $VAL_DATA .pkl)_lkh.pkl"
TEMP_RESULTS_DIR="results/windy_tsp20_val"

# Check if LKH solution already exists
if [ ! -f "$LKH_TARGET" ]; then
    echo "=================================================="
    echo "Running LKH Solver (Baseline)..."
    echo "=================================================="
    
    if [ -d "$TEMP_RESULTS_DIR" ]; then rm -rf "$TEMP_RESULTS_DIR"; fi

    # Run the solver
    python eval_baseline.py lkh_windy "$VAL_DATA" -n "$VAL_SIZE" --disable_cache

    # Move and verify results
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
# 2. DEFINE RUN FUNCTION
# ==================================================
run_batch() {
    # Accepts an array of entropies as arguments
    local ENTROPIES=("$@")
    local PIDS=()

    echo "------------------------------------------------"
    echo "Starting Batch: ${ENTROPIES[*]}"
    echo "------------------------------------------------"

    for ENTROPY in "${ENTROPIES[@]}"; do
        for TYPE in "${FEATURE_TYPES[@]}"; do
            
            RUN_NAME="exp1_${TYPE}_ent${ENTROPY}"
            LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
            
            echo " -> Launching: $TYPE | Entropy: $ENTROPY"

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
                --gnn_direction_mode forward \
                --normalization layer \
                --num_workers $NUM_WORKERS \
                --no_progress_bar \
                --entropy_coeff $ENTROPY \
                --node_feature_type $TYPE \
                --run_name "$RUN_NAME" \
                > "$LOG_FILE" 2>&1 &
            
            PIDS+=($!)
            sleep 5 # Stagger start to ease VRAM allocation
        done
    done

    echo "Waiting for PIDs: ${PIDS[*]}..."
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    echo "Batch Complete."
}

# ==================================================
# 3. EXECUTE BATCHES
# ==================================================

# --- RUN BATCH 1 (3 Experiments) ---
run_batch "${BATCH_1[@]}"

# --- RUN BATCH 2 (2 Experiments) ---
# We slightly increase workers for the smaller batch to use idle CPU
NUM_WORKERS=8 
run_batch "${BATCH_2[@]}"

echo "=================================================="
echo "All Experiments Complete."
echo "=================================================="