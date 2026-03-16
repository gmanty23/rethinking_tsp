#!/bin/bash

# --- CONFIGURATION ---
DATASET="data/windy_tsp/windy_tsp20_val.pkl"
SAMPLES=5
OUT_DIR="results/smoke_test"

# Define algorithm categories
CONSTRUCTIVE=("nearest_neighbour" "nearest_insertion" "farthest_insertion" "cheapest_insertion" "random_insertion" "clarke_wright")
ATSP_SPECIAL=("karp_steele" "greedy_karp_steele")
METAHEURISTICS=("aco" "simulated_annealing" "tabu_search" "grasp" "vns" "genetic_algorithm")
EXACT_METHODS=("bellman_held_karp") 
ADVANCED=("large_neighborhood_search" "brkga" "q_learning")

# Create output directory
mkdir -p $OUT_DIR

echo "===================================================="
echo "🛠️  VERBOSE SMOKE TEST: pyCombinatorial Baselines"
echo "===================================================="

# Helper function to execute and show output
run_verbose_test() {
    local method=$1
    echo -e "\n\033[1;34m>>> TESTING METHOD: $method <<<\033[0m"
    
    # Run the python script directly. 
    # Output (including prints and errors) will show in terminal.
    python eval_pycombinatorial.py \
        --method "$method" \
        --dataset_path "$DATASET" \
        --num_samples "$SAMPLES" \
        --output_dir "$OUT_DIR"
        
    local status=$?
    if [ $status -eq 0 ]; then
        echo -e "\033[0;32m[MATCH] $method completed successfully.\033[0m"
    else
        echo -e "\033[0;31m[ERROR] $method failed with exit code $status.\033[0m"
        # Optional: wait for user input if a failure occurs to read the traceback
        # read -p "Press enter to continue to next method..."
    fi
    echo "----------------------------------------------------"
    sleep 1 # Brief pause to allow reading output
}

# --- Execute All Groups ---

for m in "${CONSTRUCTIVE[@]}"; do run_verbose_test "$m"; done
for m in "${ATSP_SPECIAL[@]}"; do run_verbose_test "$m"; done
for m in "${METAHEURISTICS[@]}"; do run_verbose_test "$m"; done
for m in "${EXACT_METHODS[@]}"; do run_verbose_test "$m"; done
for m in "${ADVANCED[@]}"; do run_verbose_test "$m"; done 


echo -e "\n✅ All tests attempted. If you saw Tracebacks above, please check the logic for those specific methods."