#!/bin/bash

# ==================================================
# SMART CONTINUATION PIPELINE (NAB Ablation)
# ==================================================
BATCH_SIZE=128
NUM_WORKERS=6
MAX_PARALLEL_JOBS=1

# --- EXPERIMENT SETTINGS ---
# Set this to the absolute TOTAL epochs you want the models to reach
# (e.g., if it ran for 100 and you want 500 more, set this to 600)
TARGET_TOTAL_EPOCHS=200 
PROBLEM="windy_tsp"
ENTROPY=0.05
NEIGHBORS=(100) # SAME NUMBER AS RRNCO
GRAPH_SIZES=(100)

# Standardized Sizes (Multiples of 128)
VAL_SIZE=2048
EPOCH_SIZE=128000
ROLLOUT_SIZE=10240

# ==================================================
# UNCOMMENT THE COMBINATIONS YOU WANT TO RESUME
# ==================================================

# 1. Encoders to resume
ENCODERS=(
     #"gat:standard"
     #"edge_gat:standard"
     "aafm:standard"
     #"mlp:standard"
     "gnn:standard"
     #"gnn:deep"
)

# 2. Feature Types to resume
ABLATIONS=(
    "original:hybrid"       
    # "ane_pure:coords"     
    # "ane_hybrid:coords"   
    # "ane_no_gate:coords"  
    # "ane_3way_gate:coords"
    # "ane_stats_only:coords" 
)

# 3. NAB Placement Ablations to resume
NAB_MODES=(
    # "none"      
     "both"   
     #"decoder"   
    # "encoder"      
)

# ==================================================

# --- PATHS ---
LOG_DIR="logs_windy_tsp_NAB-CLIPPED-V3_ablation_NOWIND_resume"
BASE_OUTPUT_DIR="outputs/NAB-CLIPPED-V3_ablation_NOWIND_resume"
NEW_OUTPUT_DIR="outputs/NAB-CLIPPED-V3_ablation_NOWIND_resume"

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

# ==================================================
# MAIN RESUME LOOP
# ==================================================
for GRAPH_SIZE in "${GRAPH_SIZES[@]}"; do
    VAL_DATA="data/windy_tsp/windy_tsp${GRAPH_SIZE}_val.pkl"
    # PyTorch/Run.py automatically creates this subfolder structure:
    PROBLEM_DIR="${PROBLEM}_${GRAPH_SIZE}-${GRAPH_SIZE}"

    for ENC_CONF in "${ENCODERS[@]}"; do
        for ABLATION in "${ABLATIONS[@]}"; do
            for NAB_MODE in "${NAB_MODES[@]}"; do   
                for NEIGHBOR in "${NEIGHBORS[@]}"; do
                    
                    # 1. Parse Configurations
                    ENC_TYPE="${ENC_CONF%%:*}"
                    ENC_MODE="${ENC_CONF##*:}"
                    EMB_TYPE="${ABLATION%%:*}"
                    FEAT_TYPE="${ABLATION##*:}"

                    # ==========================================
                    # THE AAFM SWAP LOGIC
                    # ==========================================
                    #New change, as the aafm with just encoder diverges, we will skip it for now. If we want to run it, we can just set the nab_mode to encoder and it will work.
                    CURRENT_NAB_MODE="$NAB_MODE"
                    if [ "$ENC_TYPE" == "aafm" ] && [ "$CURRENT_NAB_MODE" == "decoder" ]; then
                        # Automatically swap decoder -> encoder for AAFM
                        # CURRENT_NAB_MODE="encoder"
                        echo "    [Skip] AAFM with decoder NAB is not supported. Skipping." | tee -a "$LOG_FILE"
                        continue
                    fi

                    # ==========================================
                    # SANITY CHECKS (Using the swapped CURRENT_NAB_MODE)
                    # ==========================================
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
                    # ==========================================

                    # 2. Reconstruct the Exact Run Name from the Training Phase
                    RUN_NAME="NAB-CLIPPED-V3_${ENC_TYPE}-${ENC_MODE}_${EMB_TYPE}_${FEAT_TYPE}_nab-${CURRENT_NAB_MODE}_tsp${GRAPH_SIZE}_ent${ENTROPY}_neighbors${NEIGHBOR}"
                    
                    # 3. Find the original output directory using the updated path structure
                    MODEL_DIR=$(find "$BASE_OUTPUT_DIR/$PROBLEM_DIR" -maxdepth 1 -type d -name "resume_${RUN_NAME}_*" | sort -r | head -n 1)
                    
                    if [ -z "$MODEL_DIR" ]; then
                        echo "    [Error] Original training directory not found for: $RUN_NAME" | tee -a "$LOG_FILE"
                        echo "            (Looked in: $BASE_OUTPUT_DIR/$PROBLEM_DIR/)" | tee -a "$LOG_FILE"
                        continue
                    fi

                    # 4. SMART RESUME MATH (Find exact remaining epochs)
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
                        GNN_DIR_FLAG="--gnn_direction_mode dual --gated"
                        if [ "$ENC_MODE" == "deep" ]; then
                            DEEP_BIAS_FLAG="--gnn_deep_bias"
                        fi
                    fi

                    # 6. Concurrency Control
                    while [ $(jobs -r -p | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
                        sleep 5
                    done
                        
                    RESUME_RUN_NAME="resume_${RUN_NAME}"
                    RUN_LOG="${LOG_DIR}/${RESUME_RUN_NAME}.log"
                    
                    echo " -> Resuming: ${ENC_TYPE}(${ENC_MODE}) | NAB: ${CURRENT_NAB_MODE} | From Epoch $CURRENT_EPOCH (Running $EPOCHS_TO_RUN more)" | tee -a "$LOG_FILE"
                    
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
                        $DEEP_BIAS_FLAG \
                        $GNN_DIR_FLAG \
                        --normalization layer \
                        --num_workers $NUM_WORKERS \
                        --no_progress_bar \
                        --entropy_coeff $ENTROPY \
                        --neighbors $NEIGHBOR \
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

# Wait for the final batch of background jobs to complete
wait
echo "" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo "ALL RESUME EXPERIMENTS COMPLETE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"