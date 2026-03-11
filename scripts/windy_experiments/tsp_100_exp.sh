#!/bin/bash

# ==================================================
# HARDWARE & CONFIGURATION
# ==================================================
# Batch Size 1024 for faster sequential processing.
# If you get OOM (Out of Memory), reduce to 512.
BATCH_SIZE=128
NUM_WORKERS=6

# --- EXPERIMENT SETTINGS ---
EPOCHS=100
PROBLEM="windy_tsp"
GRAPH_SIZE=100

# FIX: Use Percentage Strategy (0.2 * 100 = 20 neighbors)
NEIGHBORS=0.2
KNN_STRAT="random_percentage"

# FIX: Sizes must be multiples of BATCH_SIZE (1024)
VAL_SIZE=2048     
EPOCH_SIZE=128000
ROLLOUT_SIZE=10240

# --- PATHS ---
VAL_DATA="data/windy_tsp/windy_tsp100_val.pkl"
LOG_DIR="logs_windy_tsp100_seq_random"
mkdir -p $LOG_DIR
mkdir -p data/windy_tsp
mkdir -p results/lkh_windy

# *** NEW: SINGLE LOG FILE FOR EVERYTHING ***
LOG_FILE="${LOG_DIR}/tsp100_sequential_combined_random.log"

# Initialize Log File
echo "==================================================" > "$LOG_FILE"
echo "STARTING SEQUENTIAL EXPERIMENTS TSP-100" >> "$LOG_FILE"
echo "Date: $(date)" >> "$LOG_FILE"
echo "==================================================" >> "$LOG_FILE"

echo "Logging all output to: $LOG_FILE"

# ==================================================
# 1. GENERATE VALIDATION DATA
# ==================================================
if [ ! -f "$VAL_DATA" ]; then
    echo ">>> Generating Validation Data..." | tee -a "$LOG_FILE"
    python data/windy_tsp/generate_windy_tsp.py \
        --min_nodes $GRAPH_SIZE \
        --max_nodes $GRAPH_SIZE \
        --num_samples $VAL_SIZE \
        --filename $VAL_DATA \
        --alpha 3.0 \
        --max_wind 1.0 \
        --seed 1234 >> "$LOG_FILE" 2>&1
else
    echo ">>> Validation data exists." | tee -a "$LOG_FILE"
fi

# ==================================================
# 2. PRE-COMPUTE LKH BASELINE
# ==================================================
LKH_TARGET="results/lkh_windy/$(basename $VAL_DATA .pkl).pkl"

if [ ! -f "$LKH_TARGET" ]; then
    echo ">>> Running LKH Solver (Baseline)..." | tee -a "$LOG_FILE"
    python eval_baseline.py lkh_windy "$VAL_DATA" -n "$VAL_SIZE" --disable_cache >> "$LOG_FILE" 2>&1
    
    GENERATED_FILE=$(find results -name "*lkh_windy.pkl" | head -n 1)
    if [ -f "$GENERATED_FILE" ]; then
        mv "$GENERATED_FILE" "$LKH_TARGET"
        echo ">>> LKH Baseline Ready." | tee -a "$LOG_FILE"
    fi
else
    echo ">>> LKH Target exists." | tee -a "$LOG_FILE"
fi


# ==================================================
# 3. PART A: FEATURE TYPE x ENTROPY
# ==================================================
echo "==================================================" | tee -a "$LOG_FILE"
echo "STARTING PART A: Node Features & Entropy Grid" | tee -a "$LOG_FILE"
echo "Fixed Mode: forward" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

FEATURE_TYPES=("coords" "learned" "hybrid")
ENTROPY_VALUES=(0.01 0.05 0.1 0.5)
FIXED_MODE="forward"

for FEAT in "${FEATURE_TYPES[@]}"; do
    for ENTROPY in "${ENTROPY_VALUES[@]}"; do
        
        RUN_NAME="tsp100_${FIXED_MODE}_${FEAT}_ent${ENTROPY}"
        
        echo " -> Running: Feat=$FEAT | Ent=$ENTROPY ..." | tee -a "$LOG_FILE"
        
        # Add separator to log file
        echo "" >> "$LOG_FILE"
        echo "----------------------------------------------------------" >> "$LOG_FILE"
        echo "STARTING RUN: $RUN_NAME" >> "$LOG_FILE"
        echo "----------------------------------------------------------" >> "$LOG_FILE"
        
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
            --node_feature_type $FEAT \
            --gnn_direction_mode $FIXED_MODE \
            --neighbors $NEIGHBORS \
            --knn_strat $KNN_STRAT \
            --run_name "$RUN_NAME" \
            >> "$LOG_FILE" 2>&1
            
        echo "    [Done]" | tee -a "$LOG_FILE"
    done
done


# ==================================================
# 4. PART B: MESSAGE PASSING MODES
# ==================================================
echo "==================================================" | tee -a "$LOG_FILE"
echo "STARTING PART B: Message Passing Modes" | tee -a "$LOG_FILE"
echo "Fixed Feature: hybrid | Fixed Entropy: 0.01" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

MODES=("forward" "backward" "dual")
FIXED_FEAT="hybrid"
FIXED_ENTROPY=0.01

for MODE in "${MODES[@]}"; do
    
    RUN_NAME="tsp100_${MODE}_${FIXED_FEAT}_ent${FIXED_ENTROPY}"
    
    echo " -> Running: Mode=$MODE ..." | tee -a "$LOG_FILE"
    
    echo "" >> "$LOG_FILE"
    echo "----------------------------------------------------------" >> "$LOG_FILE"
    echo "STARTING RUN: $RUN_NAME" >> "$LOG_FILE"
    echo "----------------------------------------------------------" >> "$LOG_FILE"
    
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
        --entropy_coeff $FIXED_ENTROPY \
        --node_feature_type $FIXED_FEAT \
        --gnn_direction_mode $MODE \
        --neighbors $NEIGHBORS \
        --knn_strat $KNN_STRAT \
        --run_name "$RUN_NAME" \
        >> "$LOG_FILE" 2>&1
    
    echo "    [Done]" | tee -a "$LOG_FILE"
    
done

echo "==================================================" | tee -a "$LOG_FILE"
echo "STARTING PART B: Message Passing Modes" | tee -a "$LOG_FILE"
echo "Fixed Feature: hybrid | Fixed Entropy: 0.01" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

MODES=("forward" "backward" "dual")
FIXED_FEAT="hybrid"
FIXED_ENTROPY=0.05

for MODE in "${MODES[@]}"; do
    
    RUN_NAME="tsp100_${MODE}_${FIXED_FEAT}_ent${FIXED_ENTROPY}"
    
    echo " -> Running: Mode=$MODE ..." | tee -a "$LOG_FILE"
    
    echo "" >> "$LOG_FILE"
    echo "----------------------------------------------------------" >> "$LOG_FILE"
    echo "STARTING RUN: $RUN_NAME" >> "$LOG_FILE"
    echo "----------------------------------------------------------" >> "$LOG_FILE"
    
    # to tail the log file for monitoring
    echo "    To monitor progress: tail -f $LOG_FILE" | tee -a "$LOG_FILE"

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
        --entropy_coeff $FIXED_ENTROPY \
        --node_feature_type $FIXED_FEAT \
        --gnn_direction_mode $MODE \
        --neighbors $NEIGHBORS \
        --knn_strat $KNN_STRAT \
        --run_name "$RUN_NAME" \
        >> "$LOG_FILE" 2>&1
    
    echo "    [Done]" | tee -a "$LOG_FILE"
    
done
echo "==================================================" | tee -a "$LOG_FILE"
echo "ALL EXPERIMENTS COMPLETE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"