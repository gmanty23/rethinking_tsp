#!/bin/bash

# =====================================================================
# CONFIGURATION
# =====================================================================
# Main dataset for heuristics and metaheuristics
DATASET_MAIN="data/windy_tsp/windy_tsp20_val.pkl"
SAMPLES_MAIN=2048

# Small dataset exclusively for the exact mathematical solver
DATASET_SMALL="data/windy_tsp/windy_tsp20_val.pkl" 
SAMPLES_EXACT=1

OUT_DIR="results/eval_pycombinatorial"

# =====================================================================
# ALGORITHM GROUPS
# =====================================================================
EXACT=("bellman_held_karp")
CONSTRUCTIVE=("nearest_neighbour" "nearest_insertion" "farthest_insertion" "cheapest_insertion" "random_insertion" "clarke_wright")
ATSP_SPECIAL=("karp_steele" "greedy_karp_steele")
METAHEURISTICS=("aco" "simulated_annealing" "tabu_search" "grasp" "vns" "genetic_algorithm")
ADVANCED=("large_neighborhood_search" "brkga" "q_learning")

mkdir -p $OUT_DIR

echo "===================================================="
echo "🚀 STARTING FULL WINDY TSP BASELINE EVALUATION"
echo "Main Target : $DATASET_MAIN ($SAMPLES_MAIN samples)"
echo "Exact Target: $DATASET_SMALL ($SAMPLES_EXACT sample)"
echo "Output Dir  : $OUT_DIR"
echo "===================================================="

# Helper function to run the python script and log the output
run_eval() {
    local method=$1
    local dataset=$2
    local samples=$3
    
    echo -e "\n---> Running: \033[1;34m$method\033[0m"
    python eval_pycombinatorial.py \
        --method "$method" \
        --dataset_path "$dataset" \
        --num_samples "$samples" \
        --output_dir "$OUT_DIR"
}

# --- 1. EXACT METHOD (1 Sample, N=20) ---
echo -e "\n\033[1;33m[ PHASE 1: EXACT MATHEMATICAL SOLVER ]\033[0m"
for m in "${EXACT[@]}"; do
    run_eval "$m" "$DATASET_SMALL" "$SAMPLES_EXACT"
done

# --- 2. CONSTRUCTIVE HEURISTICS ---
echo -e "\n\033[1;33m[ PHASE 2: CONSTRUCTIVE HEURISTICS ]\033[0m"
for m in "${CONSTRUCTIVE[@]}"; do
    run_eval "$m" "$DATASET_MAIN" "$SAMPLES_MAIN"
done

# --- 3. ATSP-SPECIFIC ---
echo -e "\n\033[1;33m[ PHASE 3: ATSP-SPECIFIC METHODS ]\033[0m"
for m in "${ATSP_SPECIAL[@]}"; do
    run_eval "$m" "$DATASET_MAIN" "$SAMPLES_MAIN"
done

# --- 4. METAHEURISTICS ---
echo -e "\n\033[1;33m[ PHASE 4: TRAJECTORY & POPULATION METAHEURISTICS ]\033[0m"
for m in "${METAHEURISTICS[@]}"; do
    run_eval "$m" "$DATASET_MAIN" "$SAMPLES_MAIN"
done

# --- 5. ADVANCED / RL ---
echo -e "\n\033[1;33m[ PHASE 5: ADVANCED OR & REINFORCEMENT LEARNING ]\033[0m"
for m in "${ADVANCED[@]}"; do
    run_eval "$m" "$DATASET_MAIN" "$SAMPLES_MAIN"
done

echo -e "\n===================================================="
echo "✅ ALL EVALUATIONS COMPLETED. "
echo "Results are safely stored in $OUT_DIR"
echo "===================================================="