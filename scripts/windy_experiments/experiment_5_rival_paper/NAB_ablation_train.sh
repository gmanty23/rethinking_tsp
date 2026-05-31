#!/bin/bash

# ==================================================
# HARDWARE & CONFIGURATION
# ==================================================
BATCH_SIZE=128
NUM_WORKERS=6
MAX_PARALLEL_JOBS=1

# --- FIXED EXPERIMENT SETTINGS ---
EPOCHS=100
PROBLEM="windy_tsp"
ENTROPY=0.05
NEIGHBORS=100 # SAME NUMBER AS RRNCO

# Standardized Sizes (Multiples of 128)
VAL_SIZE=2048      # 16 * 128
EPOCH_SIZE=128000  # 1000 * 128
ROLLOUT_SIZE=10240 # 80 * 128

# --- VARIABLES TO TEST ---
GRAPH_SIZES=(100)

# 1. Encoders to test (Format: "ENCODER_TYPE:MODE")
ENCODERS=(
    "gat:standard"
    "edge_gat:standard"
    "aafm:standard"
    "mlp:standard"
    "gnn:standard"     # Method 1: Layer 0 Initialization
    "gnn:deep"         # Method 2: Deep Gate Injection
)

# 2. Feature Types (Array format: "NODE_EMBEDDING_TYPE:NODE_FEATURE_TYPE")
ABLATIONS=(
    "original:hybrid"       # (coords+stats)concatenated -> Currently the only one active
    # "ane_pure:coords"     # (coords+local distances)gated
    # "ane_hybrid:coords"   # ((coords+local distances)gated+global stats)concatenated
    # "ane_no_gate:coords"  # (coords+local distances+global stats)concatenated
    # "ane_3way_gate:coords"# (coords+local distances+global stats) gated
    # "ane_stats_only:coords" # (coords+stats)gated
)

# 3. NAB Placement Ablations
NAB_MODES=(
    "none"      # Baseline (Standard Encoder + Standard Decoder)
    "both"   # NAB injected into Encoder only
    "decoder"   # NAB injected into Decoder only
    "encoder"      # NAB injected into both Encoder and Decoder
)

# --- PATHS ---
LOG_DIR="logs_windy_tsp_NAB-CLIPPED-FIXED_ablation"
OUTPUT_DIR="outputs/NAB-CLIPPED-FIXED_ablation"
mkdir -p $LOG_DIR
mkdir -p data/windy_tsp
mkdir -p results/lkh_windy
mkdir -p $OUTPUT_DIR

# Master Log File
LOG_FILE="${LOG_DIR}/tsp_NAB-CLIPPED-FIXED_ablation_study.log"

# Initialize Log
echo "==================================================" > "$LOG_FILE"
echo "STARTING CONTINUOUS PARALLEL FULL ABLATION STUDY" >> "$LOG_FILE"
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
# MAIN EXPERIMENT LOOP (Asynchronous Worker Pool)
# ==================================================
for GRAPH_SIZE in "${GRAPH_SIZES[@]}"; do
    VAL_DATA="data/windy_tsp/windy_tsp${GRAPH_SIZE}_val.pkl"

    for ENC_CONF in "${ENCODERS[@]}"; do
        for ABLATION in "${ABLATIONS[@]}"; do
            for NAB_MODE in "${NAB_MODES[@]}"; do   
                for NEIGHBOR in "${NEIGHBORS[@]}"; do
                    
                    # 1. Parse Encoder Configuration
                    ENC_TYPE="${ENC_CONF%%:*}"
                    ENC_MODE="${ENC_CONF##*:}"
                    
                    # 2. Parse Feature Configuration
                    EMB_TYPE="${ABLATION%%:*}"
                    FEAT_TYPE="${ABLATION##*:}"

                    # ==========================================
                    # SANITY CHECKS: PREVENT REDUNDANT/CRASHING RUNS
                    # ==========================================
                    
                    # Check 1: AAFM mathematically requires Encoder NAB
                    if [ "$ENC_TYPE" == "aafm" ]; then
                        if [ "$NAB_MODE" == "none" ] || [ "$NAB_MODE" == "decoder" ]; then
                            echo "    [Skip] AAFM requires Encoder NAB. Skipping $ENC_TYPE with nab_mode=$NAB_MODE." | tee -a "$LOG_FILE"
                            continue
                        fi
                    fi
                    
                    # # Check 2: Edge GAT ignores Encoder NAB (prevents duplicate runs)
                    # if [ "$ENC_TYPE" == "edge_gat" ]; then
                    #     if [ "$NAB_MODE" == "encoder" ] || [ "$NAB_MODE" == "both" ]; then
                    #         echo "    [Skip] Edge GAT ignores Encoder NAB. Skipping $ENC_TYPE with nab_mode=$NAB_MODE." | tee -a "$LOG_FILE"
                    #         continue
                    #     fi
                    # fi

                    # Check 3: MLP ignores Encoder NAB (prevents duplicate runs)
                    if [ "$ENC_TYPE" == "mlp" ]; then
                        if [ "$NAB_MODE" == "encoder" ] || [ "$NAB_MODE" == "both" ]; then
                            echo "    [Skip] MLP ignores Encoder NAB. Skipping $ENC_TYPE with nab_mode=$NAB_MODE." | tee -a "$LOG_FILE"
                            continue
                        fi
                    fi
                    # ==========================================

                    # 3. Dynamic Flag Injection for GNNs
                    DEEP_BIAS_FLAG=""
                    GNN_DIR_FLAG=""
                    
                    if [ "$ENC_TYPE" == "gnn" ]; then
                        GNN_DIR_FLAG="--gnn_direction_mode dual --gated"
                        if [ "$ENC_MODE" == "deep" ]; then
                            DEEP_BIAS_FLAG="--gnn_deep_bias"
                        fi
                    fi
                    
                    # Construct a highly specific run name for the combination
                    RUN_NAME="NAB-CLIPPED-FIXED_${ENC_TYPE}-${ENC_MODE}_${EMB_TYPE}_${FEAT_TYPE}_nab-${NAB_MODE}_tsp${GRAPH_SIZE}_WIND_ent${ENTROPY}_neighbors${NEIGHBOR}"
                    RUN_LOG="${LOG_DIR}/${RUN_NAME}.log"
                        
                    # SKIP CHECK: Does a folder with this configuration already exist?
                    EXISTING_DIR=$(find "$OUTPUT_DIR" -type d -name "${RUN_NAME}_*" | head -n 1)
                        
                    if [ -n "$EXISTING_DIR" ]; then
                        echo "    [Skip] Folder already exists for: $RUN_NAME" | tee -a "$LOG_FILE"
                        continue
                    fi
                        
                    # CONCURRENCY CONTROL: Wait if we have reached MAX_PARALLEL_JOBS
                    while [ $(jobs -r -p | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
                        sleep 5
                    done
                        
                    echo " -> Launching: Enc=${ENC_TYPE}(${ENC_MODE}) | Feat=${EMB_TYPE} | NAB=${NAB_MODE}" | tee -a "$LOG_FILE"
                    
                    # LAUNCH TRAINING IN BACKGROUND
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
                        --nab_mode $NAB_MODE \
                        --use_wind \
                        --run_name "$RUN_NAME" \
                        --output_dir "$OUTPUT_DIR" \
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