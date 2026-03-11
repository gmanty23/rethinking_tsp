#!/bin/bash

# ==================================================
# HARDWARE & CONFIGURATION
# ==================================================
# Batch Size 128 is very safe for TSP20/50 parallel runs.
BATCH_SIZE=512
NUM_WORKERS=6

# --- FIXED EXPERIMENT SETTINGS ---
EPOCHS=100
PROBLEM="windy_tsp"

# Fixed Feature & KNN Mode
FIXED_FEAT="hybrid"
KNN_STRAT="percentage"
NEIGHBORS=0.2

# Standardized Sizes (Multiples of 128)
VAL_SIZE=2048      # 16 * 128
EPOCH_SIZE=128000  # 1000 * 128
ROLLOUT_SIZE=10240 # 80 * 128

# --- VARIABLES TO TEST ---
SIZES=(20)
MODES=("forward" "backward" "dual")
ENTROPY_VALUES=(0.01 0.05 0.1 0.2)

# --- PATHS ---
LOG_DIR="logs_windy_tsp_knn_percentage_study"
mkdir -p $LOG_DIR
mkdir -p data/windy_tsp
mkdir -p results/lkh_windy

# Master Log File
LOG_FILE="${LOG_DIR}/tsp50_20_parallel_entropy_percentage_knn.log"

# Initialize Log
echo "==================================================" > "$LOG_FILE"
echo "STARTING PARALLEL PERCENTAGE KNN ENTROPY STUDY (TSP-50 & TSP-20)" >> "$LOG_FILE"
echo "Batch: $BATCH_SIZE | Parallel: 4 Entropies per Mode" >> "$LOG_FILE"
echo "Date: $(date)" >> "$LOG_FILE"
echo "==================================================" >> "$LOG_FILE"

echo "Logging all output to: $LOG_FILE"

# ==================================================
# MAIN LOOP: GRAPH SIZES
# ==================================================
for GRAPH_SIZE in "${SIZES[@]}"; do

    echo "" | tee -a "$LOG_FILE"
    echo "##################################################" | tee -a "$LOG_FILE"
    echo "   SETTING UP FOR GRAPH SIZE: $GRAPH_SIZE" | tee -a "$LOG_FILE"
    echo "##################################################" | tee -a "$LOG_FILE"

    VAL_DATA="data/windy_tsp/windy_tsp${GRAPH_SIZE}_val.pkl"

    # 1. GENERATE VALIDATION DATA
    if [ ! -f "$VAL_DATA" ]; then
        echo ">>> Generating Validation Data (N=$GRAPH_SIZE)..." | tee -a "$LOG_FILE"
        python data/windy_tsp/generate_windy_tsp.py \
            --min_nodes $GRAPH_SIZE \
            --max_nodes $GRAPH_SIZE \
            --num_samples $VAL_SIZE \
            --filename $VAL_DATA \
            --alpha 3.0 \
            --max_wind 1.0 \
            --seed 1234 >> "$LOG_FILE" 2>&1
    else
        echo ">>> Validation data exists for N=$GRAPH_SIZE." | tee -a "$LOG_FILE"
    fi

    # 2. PRE-COMPUTE LKH BASELINE
    LKH_TARGET="results/lkh_windy/$(basename $VAL_DATA .pkl).pkl"

    if [ ! -f "$LKH_TARGET" ]; then
        echo ">>> Running LKH Solver (Baseline) for N=$GRAPH_SIZE..." | tee -a "$LOG_FILE"
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
    # NESTED LOOPS: MODE (Seq) -> ENTROPY (Parallel)
    # ==================================================
    for MODE in "${MODES[@]}"; do
        
        echo "" >> "$LOG_FILE"
        echo "----------------------------------------------------------" >> "$LOG_FILE"
        echo "STARTING BATCH: N=$GRAPH_SIZE | Mode=$MODE | All Entropies" >> "$LOG_FILE"
        echo "----------------------------------------------------------" >> "$LOG_FILE"
        echo " -> Launching 4 parallel runs for Mode=$MODE..." | tee -a "$LOG_FILE"

        # LAUNCH 4 PARALLEL RUNS (One for each entropy)
        for ENTROPY in "${ENTROPY_VALUES[@]}"; do
            
            RUN_NAME="tsp${GRAPH_SIZE}_${MODE}_${FIXED_FEAT}_ent${ENTROPY}_knn20"
            
            # Note: We redirect individual run output to its own file to avoid mixing logs,
            # but we echo start status to the main log.
            RUN_LOG="${LOG_DIR}/${RUN_NAME}.log"
            
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
                --node_feature_type $FIXED_FEAT \
                --gnn_direction_mode $MODE \
                --neighbors $NEIGHBORS \
                --knn_strat $KNN_STRAT \
                --run_name "$RUN_NAME" \
                > "$RUN_LOG" 2>&1 &  # <--- BACKGROUND EXECUTION
                
        done
        
        # CRITICAL: Wait for all 4 entropies to finish before changing Mode or Size
        wait
        echo "    [Batch Complete] All entropies for $MODE finished." | tee -a "$LOG_FILE"
        
    done

done

echo "" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo "ALL EXPERIMENTS COMPLETE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"