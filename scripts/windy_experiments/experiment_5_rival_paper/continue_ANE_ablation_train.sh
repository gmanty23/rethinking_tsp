#!/bin/bash

# ==================================================
# ANE ABLATION RESUME RUN (Epochs 50 to 900)
# ==================================================
BATCH_SIZE=512         # Adjusted to allow 3 parallel runs safely
NUM_WORKERS=4          # Lowered to prevent CPU bottlenecking across 3 jobs
MAX_PARALLEL_JOBS=3

# --- EXPERIMENT SETTINGS ---
EPOCHS_TO_ADD=900      # Starts at 50 + 850 = 900 Total Epochs
PROBLEM="windy_tsp"
ENTROPY=0.05
GRAPH_SIZE=100
NEIGHBORS=0.25

# Standardized Sizes
VAL_SIZE=2048      
EPOCH_SIZE=128000  
ROLLOUT_SIZE=10240 

# --- THE THREE CHECKPOINTS ---
# Format: "PATH_TO_CHECKPOINT | EMBEDDING_TYPE | FEATURE_TYPE"
RESUME_TASKS=(
    "outputs/windy_tsp_100-100/mlp_ANEABLATION_tsp100_ane_hybrid_ent0.05_20260430T162852/epoch-49.pt|ane_hybrid|coords"
    "outputs/windy_tsp_100-100/mlp_ANEABLATION_tsp100_ane_pure_ent0.05_20260430T154340/epoch-49.pt|ane_pure|coords"
    "outputs/windy_tsp_100-100/mlp_ANEABLATION_tsp100_original_ent0.05_20260430T140212/epoch-49.pt|original|hybrid"
)

# --- DIRECTORIES ---
LOG_DIR="logs_windy_tsp_ane_resume"
mkdir -p $LOG_DIR
VAL_DATA="data/windy_tsp/windy_tsp${GRAPH_SIZE}_val.pkl"
LOG_FILE="${LOG_DIR}/master_resume.log"

echo "==================================================" | tee -a "$LOG_FILE"
echo " LAUNCHING MULTI-MODEL RESUME (Target: 900 Epochs)" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"


echo "use the command below to monitor all logs in real-time:"
echo "  tail -f ${LOG_DIR}/*.log"
echo "=================================================="

# ==================================================
# MAIN LAUNCH LOOP
# ==================================================
for TASK in "${RESUME_TASKS[@]}"; do
    # Split the string by the pipe character '|'
    IFS='|' read -r LOAD_PATH EMB_TYPE FEAT_TYPE <<< "$TASK"
    
    RUN_NAME="resume_mlp_tsp${GRAPH_SIZE}_${EMB_TYPE}_ent${ENTROPY}"
    RUN_LOG="${LOG_DIR}/${RUN_NAME}.log"

    # Safety check: Does the checkpoint exist?
    if [ ! -f "$LOAD_PATH" ]; then
        echo "[ERROR] Checkpoint not found: $LOAD_PATH" | tee -a "$LOG_FILE"
        continue
    fi

    # Concurrency control
    while [ $(jobs -r -p | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
        sleep 5
    done

    echo " -> Resuming: $EMB_TYPE (Base Features: $FEAT_TYPE)" | tee -a "$LOG_FILE"

    # LAUNCH TRAINING
    python -u run.py \
        --problem $PROBLEM \
        --min_size $GRAPH_SIZE \
        --max_size $GRAPH_SIZE \
        --n_epochs $EPOCHS_TO_ADD \
        --batch_size $BATCH_SIZE \
        --epoch_size $EPOCH_SIZE \
        --val_datasets $VAL_DATA \
        --val_size $VAL_SIZE \
        --rollout_size $ROLLOUT_SIZE \
        --model attention \
        --encoder mlp \
        --normalization layer \
        --num_workers $NUM_WORKERS \
        --no_progress_bar \
        --entropy_coeff $ENTROPY \
        --neighbors $NEIGHBORS \
        --node_embedding_type $EMB_TYPE \
        --node_feature_type $FEAT_TYPE \
        --resume "$LOAD_PATH" \
        --run_name "$RUN_NAME" > "$RUN_LOG" 2>&1 &

done

# Wait for all background jobs to finish
wait
echo "" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo " ALL RESUME JOBS COMPLETED" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"