#!/bin/bash

# ==================================================
# HARDWARE & CONFIGURATION (OOM Safety Applied)
# ==================================================
# Reduced BATCH_SIZE to 64 to prevent CUDA Out of Memory errors on TSP-100
BATCH_SIZE=64
NUM_WORKERS=4

# --- EXPERIMENT SETTINGS ---
EPOCHS=100
PROBLEM="windy_tsp"
GRAPH_SIZE=100

# Standardized Sizes
VAL_SIZE=2048     
EPOCH_SIZE=128000
ROLLOUT_SIZE=10240

# --- PATHS ---
VAL_DATA="data/windy_tsp/windy_tsp100_val.pkl"
LOG_DIR="logs_windy_tsp100_knn_study"
mkdir -p $LOG_DIR
mkdir -p data/windy_tsp
mkdir -p results/lkh_windy

LOG_FILE="${LOG_DIR}/tsp100_knn_ablation_combined.log"

# Initialize Log File
echo "==================================================" > "$LOG_FILE"
echo "STARTING KNN ABLATION EXPERIMENTS TSP-100" >> "$LOG_FILE"
echo "Date: $(date)" >> "$LOG_FILE"
echo "==================================================" >> "$LOG_FILE"

# ==================================================
# 1. GENERATE VALIDATION DATA (Unchanged)
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
fi

# ==================================================
# 2. PRE-COMPUTE LKH BASELINE (Unchanged)
# ==================================================
LKH_TARGET="results/lkh_windy/$(basename $VAL_DATA .pkl).pkl"
if [ ! -f "$LKH_TARGET" ]; then
    echo ">>> Running LKH Solver (Baseline)..." | tee -a "$LOG_FILE"
    python eval_baseline.py lkh_windy "$VAL_DATA" -n "$VAL_SIZE" --disable_cache >> "$LOG_FILE" 2>&1
    GENERATED_FILE=$(find results -name "*lkh_windy.pkl" | head -n 1)
    if [ -f "$GENERATED_FILE" ]; then
        mv "$GENERATED_FILE" "$LKH_TARGET"
    fi
fi

# ==================================================
# 3. KNN ABLATION STUDY
# ==================================================
echo "==================================================" | tee -a "$LOG_FILE"
echo "STARTING KNN STUDY: Fixed Mode=dual, Feat=hybrid" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

# 0.2 is computed first as the anchor. Then we explore lower (0.1, 0.15) 
# and higher (0.25, 0.3) connectivity.
KNN_VALUES=(0.2)
KNN_STRAT="percentage"
FIXED_MODE="dual"
FIXED_FEAT="hybrid"
FIXED_ENTROPY=0.01

for KNN in "${KNN_VALUES[@]}"; do
    
    RUN_NAME="tsp100_knn${KNN}_${FIXED_MODE}_${FIXED_FEAT}"
    
    echo " -> Running: KNN=$KNN | Mode=$FIXED_MODE ..." | tee -a "$LOG_FILE"
    #Tell the user how to monitor the log file for this run
    echo "    To monitor progress: tail -f $LOG_FILE" | tee -a "$LOG_FILE"
    
    echo "" >> "$LOG_FILE"
    echo "----------------------------------------------------------" >> "$LOG_FILE"
    echo "STARTING RUN: $RUN_NAME" >> "$LOG_FILE"
    echo "----------------------------------------------------------" >> "$LOG_FILE"

    #skip if log file already exists for this run (assumes run completed)
    if grep -q "STARTING RUN: $RUN_NAME" "$LOG_FILE"; then
        echo "    [SKIP] Run $RUN_NAME already logged. Assuming completed." | tee -a "$LOG_FILE"
        continue
    fi  
    
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
        --gnn_direction_mode $FIXED_MODE \
        --neighbors $KNN \
        --knn_strat $KNN_STRAT \
        --run_name "$RUN_NAME" \
        >> "$LOG_FILE" 2>&1
        
    echo "    [Done: $RUN_NAME]" | tee -a "$LOG_FILE"
done

echo "==================================================" | tee -a "$LOG_FILE"
echo "ALL EXPERIMENTS COMPLETE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"