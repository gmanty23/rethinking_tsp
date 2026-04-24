#!/bin/bash

# ==================================================
# HARDWARE & CONFIGURATION
# ==================================================
BATCH_SIZE=128        # Lowered to 128 to prevent OOM on N=100 GNNs
NUM_WORKERS=18
MAX_PARALLEL_JOBS=2   # Running sequentially for safety

# --- FIXED EXPERIMENT SETTINGS ---
EPOCHS=49             
PROBLEM="windy_tsp" 
ENTROPY=0.05
N_LAYERS=1

# Standardized Sizes
VAL_SIZE=2048      
EPOCH_SIZE=128000  
ROLLOUT_SIZE=10240 

# --- VARIABLES TO TEST ---
GRAPH_SIZES=(100)
FEATURES=("hybrid")  #meter luego coords y learned

# New GNN-Specific Variables
KNN_STRATS=("cost_weighted_percentage" "percentage" "random_percentage") 
NEIGHBORS=(0.05 0.2)
MODES=("forward" "backward" "dual")

# --- PATHS ---
LOG_DIR="logs_windy_tsp_gnn100_ablation"
OUTPUT_DIR="outputs/gnn100_ablation" 
mkdir -p $LOG_DIR
mkdir -p data/windy_tsp
mkdir -p results/lkh_windy
mkdir -p $OUTPUT_DIR

# Master Log File
LOG_FILE="${LOG_DIR}/upgrade_ablation_tsp_gnn100.log"

# Initialize Log
echo "==================================================" > "$LOG_FILE"
echo "STARTING CONTINUOUS PARALLEL GNN100 ABLATION STUDY" >> "$LOG_FILE"
echo "Batch: $BATCH_SIZE | Max Parallel Jobs: $MAX_PARALLEL_JOBS" >> "$LOG_FILE"
echo "Date: $(date)" >> "$LOG_FILE"
echo "==================================================" >> "$LOG_FILE"

echo "Logging all output to: $LOG_FILE"
echo ""
echo "=================================================="
echo "To monitor all running logs, open a new terminal and run:"
echo "  tail -f ${LOG_DIR}/*.log"
echo "=================================================="
echo ""

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
# MAIN EXPERIMENT LOOP
# ==================================================
for GRAPH_SIZE in "${GRAPH_SIZES[@]}"; do
    VAL_DATA="data/windy_tsp/windy_tsp${GRAPH_SIZE}_val.pkl"

    for STRAT in "${KNN_STRATS[@]}"; do 
        for FEAT in "${FEATURES[@]}"; do
            for MODE in "${MODES[@]}"; do 
                for NEIGHBOR in "${NEIGHBORS[@]}"; do
                
                    RUN_NAME="upgrade_ablation_gnn100_tsp${GRAPH_SIZE}_${MODE}_${FEAT}_ent${ENTROPY}_${STRAT}_n${NEIGHBOR}"
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
                    
                    echo " -> Launching: Size=${GRAPH_SIZE} | Feat=${FEAT} | Strat=${STRAT} | Mode=${MODE} | Neighbors=${NEIGHBOR}" | tee -a "$LOG_FILE"

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
                        --n_encode_layers $N_LAYERS \
                        --gated \
                        --normalization layer \
                        --num_workers $NUM_WORKERS \
                        --no_progress_bar \
                        --entropy_coeff $ENTROPY \
                        --node_feature_type $FEAT \
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