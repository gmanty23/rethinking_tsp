#!/bin/bash

# ==============================================================================
# resume_ablations.sh
# UPDATE: whole file

# Smart Continuation Pipeline for Asymmetric/Windy TSP.
# Automatically finds the highest epoch checkpoint for a given configuration 
# and resumes training until TARGET_TOTAL_EPOCHS is reached.
# ==============================================================================


# --------------------------------------------------
# 1. HARDWARE & CONCURRENCY CONFIGURATION
# --------------------------------------------------
BATCH_SIZE=512
NUM_WORKERS=10
MAX_PARALLEL_JOBS=1


# --------------------------------------------------
# 2. EXPERIMENT SETTINGS
# --------------------------------------------------
# Set this to the absolute TOTAL epochs you want the models to reach
# (e.g., if it ran for 100 and you want 500 more, set this to 600)
TARGET_TOTAL_EPOCHS=1000 
PROBLEM="windy_tsp"
ENTROPY=0.05

# Standardized Sizes (Multiples of 128)
VAL_SIZE=2048
EPOCH_SIZE=128000
ROLLOUT_SIZE=10240


# --------------------------------------------------
# 3. ABLATION GRID (Must match the architecture you are trying to resume)
# --------------------------------------------------
GRAPH_SIZES=(50 20)
NEIGHBORS=(100)     # CRUCUAL: Must be between 0 and 1

ENCODERS=(
    #"gnn:deep"         # GNN with Deep NAB Injection
    #"gnn:standard"    # GNN with Layer 0 Initialization
    #"gat:standard"
    #"edge_gat:standard"
    "aafm:standard"
    #"mlp:standard"
)

ABLATIONS=(
    #"original:hybrid"       
    #"ane_pure:coords"      
    #"ane_hybrid:coords"     
    #"ane_no_gate:coords"   
    "ane_3way_gate:coords"  
    #"ane_stats_only:coords" 
)

NAB_MODES=(
    #"none"         
    "both"          
    #"decoder"       
    #"encoder"      
)

KNN_STRATS=(
    #"random_percentage"            
    #"percentage"                   
    "cost_weighted_percentage"      
)

GNN_DIRECTIONS=(
    "dual"       
    #"forward"   
    #"backward"  
)

N_LAYERS=(
    #1
    #2
    3
    #4
    #5
)


# --------------------------------------------------
# 4. PATHS & LOGGING SETUP
# --------------------------------------------------
EXPERIMENT_NAME="NAB-CLIPPED-V3_FINAL_ablation"
LOG_DIR="logs_windy_tsp_${EXPERIMENT_NAME}_resume"
BASE_OUTPUT_DIR="outputs/PAPER_OUTPUTS/1-Benchmarking/best_models/${EXPERIMENT_NAME}"
NEW_OUTPUT_DIR="outputs/PAPER_OUTPUTS/1-Benchmarking/best_models/${EXPERIMENT_NAME}_resumed"

mkdir -p "$LOG_DIR"
mkdir -p "$NEW_OUTPUT_DIR"
LOG_FILE="${LOG_DIR}/master_resume.log"

echo "==================================================" > "$LOG_FILE"
echo " STARTING SMART RESUME FOR TARGET: $TARGET_TOTAL_EPOCHS EPOCHS" >> "$LOG_FILE"
echo " Date: $(date)" >> "$LOG_FILE"
echo "==================================================" >> "$LOG_FILE"

echo "Logging all output to: $LOG_FILE"
echo "To monitor all running logs, open a new terminal and run:"
echo "  tail -f ${LOG_DIR}/*.log"
echo "=================================================="
echo ""

# --------------------------------------------------
# 5. MAIN RESUME LOOP
# --------------------------------------------------
for GRAPH_SIZE in "${GRAPH_SIZES[@]}"; do
    VAL_DATA="data/windy_tsp/windy_tsp${GRAPH_SIZE}_val.pkl"
    PROBLEM_DIR="${PROBLEM}_${GRAPH_SIZE}-${GRAPH_SIZE}"

    for ENC_CONF in "${ENCODERS[@]}"; do
        for ABLATION in "${ABLATIONS[@]}"; do
            for NAB_MODE in "${NAB_MODES[@]}"; do   
                for NEIGHBOR in "${NEIGHBORS[@]}"; do
                    for STRAT in "${KNN_STRATS[@]}"; do
                        for DIR in "${GNN_DIRECTIONS[@]}"; do
                            for N_LAY in "${N_LAYERS[@]}"; do
                    
                                # 1. Parse Configurations
                                ENC_TYPE="${ENC_CONF%%:*}"
                                ENC_MODE="${ENC_CONF##*:}"
                                EMB_TYPE="${ABLATION%%:*}"
                                FEAT_TYPE="${ABLATION##*:}"

                                # ==========================================
                                # SANITY CHECKS (Must match train.sh logic)
                                # ==========================================
                                CURRENT_NAB_MODE="$NAB_MODE"
                                if [ "$ENC_TYPE" == "aafm" ] && [ "$CURRENT_NAB_MODE" == "decoder" ]; then
                                    echo "    [Skip] AAFM with decoder NAB is not supported. Skipping." | tee -a "$LOG_FILE"
                                    continue
                                fi

                                                                # Check 2.5: ONLY FOR TSP SIZES ABLATION: make mlp force the nab mode to 'decoder'
                                if [ "$ENC_TYPE" == "mlp" ]; then
                                    if [ "$NAB_MODE" == "encoder" ] || [ "$NAB_MODE" == "both" ]; then
                                        echo "    [Override] MLP ignores Encoder NAB. Forcing nab_mode=decoder for $ENC_TYPE." | tee -a "$LOG_FILE"
                                        # Brute-force change the variable for this specific run
                                        NAB_MODE="decoder" 
                                    fi
                                fi

                                if [ "$ENC_TYPE" == "aafm" ]; then
                                    if [ "$CURRENT_NAB_MODE" == "none" ]; then
                                        echo "    [Skip] AAFM requires Encoder NAB. Skipping." | tee -a "$LOG_FILE"
                                        continue
                                    fi
                                fi
                                
                                if [ "$ENC_TYPE" == "mlp" ]; then
                                    if [ "$CURRENT_NAB_MODE" == "encoder" ] || [ "$CURRENT_NAB_MODE" == "both" ]; then
                                        echo "    [Skip] MLP ignores Encoder NAB. Skipping." | tee -a "$LOG_FILE"
                                        continue
                                    fi
                                fi

                                if [ "$ENC_TYPE" == "gnn" ] && [ "$ENC_MODE" == "deep" ]; then
                                    if [ "$CURRENT_NAB_MODE" == "none" ] || [ "$CURRENT_NAB_MODE" == "decoder" ]; then
                                        echo "    [Skip] GNN:Deep identical to GNN:Standard here. Skipping." | tee -a "$LOG_FILE"
                                        continue
                                    fi
                                fi

                                # if [ "$ENC_TYPE" != "gnn" ] && [ "$DIR" != "forward" ]; then
                                #     continue
                                # fi
                                # ==========================================

                                # 2. Reconstruct the EXACT Run Name from the Training Phase
                                RUN_NAME="${EXPERIMENT_NAME}_${ENC_TYPE}-${ENC_MODE}_${EMB_TYPE}_${FEAT_TYPE}_nab-${CURRENT_NAB_MODE}_tsp${GRAPH_SIZE}_ent${ENTROPY}_neighbors${NEIGHBOR}_strat-${STRAT}_dir-${DIR}_layers-${N_LAY}"
                                
                                # 3. Find the original output directory
                                MODEL_DIR=$(find "$BASE_OUTPUT_DIR/$PROBLEM_DIR" -maxdepth 1 -type d -name "${RUN_NAME}_*" | sort -r | head -n 1)
                                
                                if [ -z "$MODEL_DIR" ]; then
                                    echo "    [Error] Original training directory not found for: $RUN_NAME" | tee -a "$LOG_FILE"
                                    continue
                                fi

                                # 4. SMART RESUME MATH
                                HIGHEST_CKPT=$(ls -v "$MODEL_DIR"/epoch-*.pt 2>/dev/null | tail -n 1)
                                
                                if [ -z "$HIGHEST_CKPT" ]; then
                                    echo "    [Error] No .pt checkpoints found in $MODEL_DIR" | tee -a "$LOG_FILE"
                                    continue
                                fi
                                
                                CURRENT_EPOCH=$(basename "$HIGHEST_CKPT" | sed 's/epoch-//' | sed 's/\.pt//')
                                
                                if [ "$CURRENT_EPOCH" -ge "$((TARGET_TOTAL_EPOCHS - 1))" ]; then
                                    echo "    [Skip] $RUN_NAME is already at or beyond target epoch $TARGET_TOTAL_EPOCHS (Current: $CURRENT_EPOCH)" | tee -a "$LOG_FILE"
                                    continue
                                fi

                                EPOCHS_TO_RUN=$((TARGET_TOTAL_EPOCHS - CURRENT_EPOCH - 1))

                                # 5. Dynamic Flag Injection for GNNs
                                DEEP_BIAS_FLAG=""
                                GNN_DIR_FLAG=""
                                if [ "$ENC_TYPE" == "gnn" ]; then
                                    GNN_DIR_FLAG="--gnn_direction_mode $DIR --gated"
                                    if [ "$ENC_MODE" == "deep" ]; then
                                        DEEP_BIAS_FLAG="--gnn_deep_bias"
                                    fi
                                fi

                                # 6. GPU Assignment Logic (Round-Robin)
                                NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
                                CURRENT_JOBS=$(jobs -r -p | wc -l)
                                TARGET_GPU=$(( CURRENT_JOBS % NUM_GPUS ))

                                # Concurrency Control
                                while [ $(jobs -r -p | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
                                    sleep 5
                                done
                                    
                                RESUME_RUN_NAME="resume_${RUN_NAME}"
                                RUN_LOG="${LOG_DIR}/${RESUME_RUN_NAME}.log"
                                
                                echo " -> Resuming on GPU ${TARGET_GPU}: ${ENC_TYPE}(${ENC_MODE}) | Layers=${N_LAY} | From Epoch $CURRENT_EPOCH (Running $EPOCHS_TO_RUN more)" | tee -a "$LOG_FILE"
                                
                                # 7. LAUNCH RESUME COMMAND
                                python -u run.py \
                                    --problem $PROBLEM \
                                    --min_size $GRAPH_SIZE \
                                    --max_size $GRAPH_SIZE \
                                    --n_epochs $EPOCHS_TO_RUN \
                                    --batch_size $BATCH_SIZE \
                                    --epoch_size $EPOCH_SIZE \
                                    --val_datasets $VAL_DATA \
                                    --val_size $VAL_SIZE \
                                    --rollout_size $ROLLOUT_SIZE \
                                    --model attention \
                                    --encoder $ENC_TYPE \
                                    --n_encode_layers $N_LAY \
                                    $DEEP_BIAS_FLAG \
                                    $GNN_DIR_FLAG \
                                    --normalization layer \
                                    --num_workers $NUM_WORKERS \
                                    --no_progress_bar \
                                    --entropy_coeff $ENTROPY \
                                    --neighbors $NEIGHBOR \
                                    --knn_strat $STRAT \
                                    --node_embedding_type $EMB_TYPE \
                                    --node_feature_type $FEAT_TYPE \
                                    --nab_mode $CURRENT_NAB_MODE \
                                    --resume "$HIGHEST_CKPT" \
                                    --run_name "$RESUME_RUN_NAME" \
                                    --output_dir "$NEW_OUTPUT_DIR" \
                                    > "$RUN_LOG" 2>&1 &
                            done
                        done
                    done
                done
            done
        done
    done
done

# Wait for the final batch of background jobs to complete
wait
echo "" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo "ALL RESUME EXPERIMENTS COMPLETE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"