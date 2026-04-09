#!/bin/bash

# ==================================================
# HARDWARE & CONFIGURATION
# ==================================================
BATCH_SIZE=512
NUM_WORKERS=3
MAX_PARALLEL_JOBS=6

# --- FIXED EXPERIMENT SETTINGS ---
EPOCHS=100
PROBLEM="windy_tsp"
NEIGHBOR=0.2

# Standardized Sizes (Multiples of 128)
VAL_SIZE=2048      # 16 * 128
EPOCH_SIZE=128000  # 1000 * 128
ROLLOUT_SIZE=10240 # 80 * 128

# --- VARIABLES TO TEST ---
GRAPH_SIZES=(20)
FIXED_FEATS=("coords" "learned")
KNN_STRATS=("percentage" "random_percentage")
MODES=("forward" "backward" "dual")
ENTRPY=(0.01 0.1 0.05 0.5)

# --- PATHS ---
LOG_DIR="logs_windy_tsp_knn_neighbor_study"
OUTPUT_DIR="outputs/windy_tsp_20-20" 
mkdir -p $LOG_DIR
mkdir -p data/windy_tsp
mkdir -p results/lkh_windy
mkdir -p $OUTPUT_DIR

# Master Log File
LOG_FILE="${LOG_DIR}/windy_tsp_ablation_study.log"

# Initialize Log
echo "==================================================" > "$LOG_FILE"
echo "STARTING CONTINUOUS PARALLEL FEATURE & GRAPH SIZE STUDY" >> "$LOG_FILE"
echo "Batch: $BATCH_SIZE | Max Parallel Jobs: $MAX_PARALLEL_JOBS" >> "$LOG_FILE"
echo "Date: $(date)" >> "$LOG_FILE"
echo "==================================================" >> "$LOG_FILE"

echo "Logging all output to: $LOG_FILE"

# ==================================================
# PREPARATION: VALIDATION DATA & LKH BASELINE
# ==================================================
for GRAPH_SIZE in "${GRAPH_SIZES[@]}"; do
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
done

# ==================================================
# MAIN EXPERIMENT LOOP (Asynchronous Worker Pool)
# ==================================================
for GRAPH_SIZE in "${GRAPH_SIZES[@]}"; do
    VAL_DATA="data/windy_tsp/windy_tsp${GRAPH_SIZE}_val.pkl"
    
    for STRAT in "${KNN_STRATS[@]}"; do
        for FIXED_FEAT in "${FIXED_FEATS[@]}"; do
            for MODE in "${MODES[@]}"; do
                for ENTROPY in "${ENTRPY[@]}"; do
                
                    RUN_NAME="tsp${GRAPH_SIZE}_${MODE}_${FIXED_FEAT}_ent${ENTROPY}_${STRAT}_n${NEIGHBOR}"
                    RUN_LOG="${LOG_DIR}/${RUN_NAME}.log"
                    
                    # 1. SKIP CHECK: Does a folder with this configuration already exist?
                    EXISTING_DIR=$(find "$OUTPUT_DIR" -type d -name "${RUN_NAME}_*" | head -n 1)
                    
                    if [ -n "$EXISTING_DIR" ]; then
                        echo "    [Skip] Folder already exists for: $RUN_NAME" | tee -a "$LOG_FILE"
                        continue
                    fi
                    
                    # 2. CONCURRENCY CONTROL: Wait if we have reached MAX_PARALLEL_JOBS
                    while [ $(jobs -r -p | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
                        sleep 5
                    done
                    
                    echo " -> Launching: Size=${GRAPH_SIZE} | Feat=${FIXED_FEAT} | Strat=${STRAT} | Mode=${MODE} | Entropy=${ENTROPY}" | tee -a "$LOG_FILE"

                    # 3. LAUNCH TRAINING IN BACKGROUND
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
                        --neighbors $NEIGHBOR \
                        --knn_strat $STRAT \
                        --run_name "$RUN_NAME" \
                        > "$RUN_LOG" 2>&1 &
                done        
            done
        done
    done
done

# Wait for the final batch of background jobs to complete
wait
echo "" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo "ALL EXPERIMENTS COMPLETE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"