import os
import json
import time
import numpy as np
import pickle
from tqdm import tqdm
import concurrent.futures
import argparse

# Ensure pyCombinatorial is installed
from pyCombinatorial import algorithm

# Your existing Windy TSP data generator functions
from data.windy_tsp.generate_windy_tsp import calculate_cost_matrix

def load_windy_dataset(filepath, num_samples=None):
    """Loads the exact validation dataset used by the GNN."""
    print(f"Loading dataset from {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    if num_samples is not None:
        data = data[:num_samples]
    return data

def evaluate_single_instance(args):
    """
    Worker function to evaluate a single TSP instance.
    Packaged to work smoothly with concurrent.futures.
    """
    instance_idx, coords, wind_vector, method = args
    
    # Extract alpha if it exists in your data format, otherwise default
    alpha = 3.0 # Default based on Rethinking TSP / your generator logic
    
    # 1. Compute Asymmetric Matrix
    cost_matrix = calculate_cost_matrix(coords, wind_vector, alpha)
    
    start_time = time.time()
    
    # =====================================================================
    # 2. Run selected PyCombinatorial Baseline Directly
    # =====================================================================
    
    # --- SUBSECTION A: CONSTRUCTIVE HEURISTICS ---
    # Nature: Deterministic/Greedy. 
    # Sensitivity: LOW. These are mostly parameter-free or only require the matrix.
    # Generally O(N^2) or O(N^3). Very fast and stable for baseline comparisons.
    if method == 'nearest_neighbour':
        tour, cost = algorithm.nearest_neighbour(distance_matrix=cost_matrix, local_search=False, verbose=False)

    elif method == 'nearest_insertion':
        tour, cost = algorithm.nearest_insertion(distance_matrix=cost_matrix, verbose=False)

    elif method == 'farthest_insertion':
        tour, cost = algorithm.farthest_insertion(distance_matrix=cost_matrix, local_search=False, verbose=False)

    elif method == 'cheapest_insertion':
        tour, cost = algorithm.cheapest_insertion(distance_matrix=cost_matrix, local_search=False, verbose=False)

    elif method == 'random_insertion':
        tour, cost = algorithm.random_insertion(distance_matrix=cost_matrix, verbose=False)

    elif method == 'multifragment':
        tour, cost = algorithm.multifragment_heuristic(distance_matrix=cost_matrix, local_search=False, verbose=False)

    elif method == 'clarke_wright':
        tour, cost = algorithm.clarke_wright_savings(distance_matrix=cost_matrix, local_search=False, verbose=False)

    # --- SUBSECTION B: ATSP-SPECIFIC & MATH-DRIVEN ---
    # Nature: Solves Assignment Problem (LAP) then patches cycles.
    # Sensitivity: LOW.
    # Designed specifically for asymmetric matrices / cycle patching.
    elif method == 'karp_steele':
        tour, cost = algorithm.karp_steele_patching(distance_matrix=cost_matrix, verbose=False)

    elif method == 'greedy_karp_steele':
        tour, cost = algorithm.greedy_karp_steele_patching(distance_matrix=cost_matrix, verbose=False)

    # --- SUBSECTION C: METAHEURISTICS (HYPERPARAMETER SENSITIVE) ---
    # Note: These often require a 'city_tour' (initial seed). 
    # We use a simple Nearest Neighbour tour to seed them for better convergence.
    elif method in ['aco', 'simulated_annealing', 'tabu_search', 'grasp', 'vns', 'genetic_algorithm']:
        # Generate seed tour using Nearest Neighbour
        seed_tour_np, seed_cost = algorithm.nearest_neighbour(distance_matrix=cost_matrix, local_search=False, verbose=False)
        # CONVERT TOUR TO LIST (Sequences usually need list methods like .index or .pop)
        seed_tour = seed_tour_np.tolist() if hasattr(seed_tour_np, 'tolist') else list(seed_tour_np)
        # "Solution Object" for Tabu and VNS
        seed_solution = [seed_tour, seed_cost]

        if method == 'aco':
            # HIGHLY SENSITIVE: decay and alpha/beta balance impact stagnation vs exploration.
            tour, cost = algorithm.ant_colony_optimization(
                distance_matrix=cost_matrix, ants=cost_matrix.shape[0], iterations=100, 
                alpha=1, beta=2, decay=0.05, local_search=False, verbose=False
            )

        elif method == 'simulated_annealing':
            # SENSITIVE: initial_temperature and alpha (cooling rate) determine convergence depth.
            tour, cost = algorithm.simulated_annealing_tsp(
                distance_matrix=cost_matrix, initial_temperature=1.0, alpha=0.9, 
                temperature_iterations=10, verbose=False
            )

        elif method == 'tabu_search':
                # Pass seed_solution [tour, cost] instead of just seed_tour
                tour, cost = algorithm.tabu_search(
                    distance_matrix=cost_matrix, city_tour=seed_solution, 
                    iterations=150, tabu_tenure=max(1, int(cost_matrix.shape[0]/3)), verbose=False
                )

        elif method == 'grasp':
            # SENSITIVE: rcl (candidate list) and greediness_value control the randomness.
            # Using the full name from your list:
            tour, cost = algorithm.greedy_randomized_adaptive_search_procedure(
                distance_matrix=cost_matrix, city_tour=seed_tour, 
                iterations=50, rcl=5, greediness_value=0.5, verbose=False
            )

        elif method == 'vns':
                # Pass seed_solution [tour, cost] instead of just seed_tour
                tour, cost = algorithm.variable_neighborhood_search(
                    distance_matrix=cost_matrix, city_tour=seed_solution, 
                    max_attempts=20, neighbourhood_size=5, iterations=50, verbose=False
                )
            
        elif method == 'genetic_algorithm':
            # SENSITIVE: population_size and mutation_rate.
            tour, cost = algorithm.genetic_algorithm(
                distance_matrix=cost_matrix, population_size=20, 
                generations=100, mutation_rate=0.1, verbose=False
            )

    # --- SUBSECTION D: ADVANCED / REINFORCEMENT LEARNING ---
    # Nature: Learning-based. May involve exploration and non-determinism.
    # Sensitivity: HIGH. Hyperparameters can drastically affect performance and convergence.  
        
    elif method == 'large_neighborhood_search':
        tour, cost = algorithm.large_neighborhood_search(
            distance_matrix=cost_matrix, 
            iterations=150, 
            neighborhood_size=4, 
            local_search=False, 
            verbose=False
        )

    elif method == 'brkga':
        tour, cost = algorithm.biased_random_key_genetic_algorithm(
            distance_matrix=cost_matrix, 
            population_size=20, 
            generations=100, 
            verbose=False
        )
            
    elif method == 'q_learning':
            result = algorithm.q_learning(
                distance_matrix=cost_matrix, 
                learning_rate=0.1,       # How much it overrides old info
                discount_factor=0.95,    # How much it cares about future rewards
                epsilon=0.15,            # Exploration rate (15% random moves)
                episodes=5000,           # CRITICAL: Restore to default 5000
                local_search=False, 
                verbose=False
            )
            tour, cost = result[0], result[1]

    # --- SUBSECTION E: EXACT METHODS ---
    # Nature: Dynamic Programming / Branch & Bound.
    # Sensitivity: NONE (Always finds optimal). 
    # WARNING: O(2^N). DO NOT RUN ON N > 25.
    
    elif method == 'bellman_held_karp':
        tour, cost = algorithm.bellman_held_karp_exact_algorithm(distance_matrix=cost_matrix, verbose=False)

    # elif method == 'branch_and_bound':
    #     tour, cost = algorithm.branch_and_bound(distance_matrix=cost_matrix)

    else:
        raise ValueError(f"Method '{method}' not found or incompatible with ATSP Windy TSP.")

    duration = time.time() - start_time
    
    return instance_idx, cost, duration, tour

def main(opts):
    print(f"--- Starting pyCombinatorial Parallel Evaluation ---")
    print(f"Algorithm: {opts.method.upper()}")
    
    # 1. Ensure Output Directory Exists
    os.makedirs(opts.output_dir, exist_ok=True)
    
    # 2. Load Data
    dataset = load_windy_dataset(opts.dataset_path, num_samples=opts.num_samples)
    N_nodes = len(dataset[0]['loc']) if isinstance(dataset[0], dict) else len(dataset[0][0])
    print(f"Loaded {len(dataset)} instances (N={N_nodes}).")

    # 3. Prepare arguments for parallel processing
    tasks = []
    for i, instance in enumerate(dataset):
        if isinstance(instance, dict):
            coords = instance['loc']
            wind_vector = instance['wind']
        else:
            coords, wind_vector = instance[0], instance[1]
        tasks.append((i, coords, wind_vector, opts.method))

    results_cost = []
    results_time = []
    all_tours = {}

    # 4. Execute using Multiprocessing (CPU parallelization instead of GPU batching)
    # If max_workers is None, it defaults to the number of processors on the machine.
    with concurrent.futures.ProcessPoolExecutor(max_workers=opts.max_workers) as executor:
        # Use tqdm to track progress as futures complete
        futures = {executor.submit(evaluate_single_instance, task): task for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc=f"Evaluating {opts.method}"):
            try:
                idx, cost, duration, tour = future.result()
                results_cost.append(cost)
                results_time.append(duration)
                all_tours[idx] = tour
            except Exception as exc:
                print(f"Instance generated an exception: {exc}")

    # 5. Aggregate metrics
    avg_cost = np.mean(results_cost)
    avg_time = np.mean(results_time)
    
    print("\n--- Final Results ---")
    print(f"Method:                  {opts.method.upper()}")
    print(f"Average Tour Cost:       {avg_cost:.4f}")
    print(f"Average Execution Time:  {avg_time:.4f} seconds/instance")

    # 6. Save results to JSON
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    output_filename = os.path.join(opts.output_dir, f"{opts.method}_N{N_nodes}_samples{len(dataset)}_{timestamp}.json")
    results_payload = {
        "method": opts.method,
        "graph_size": N_nodes,
        "num_samples": len(dataset),
        "average_cost": avg_cost,
        "average_time_per_instance": avg_time,
        "costs": results_cost,        # Saving individual costs for statistical testing later
        "times": results_time
    }
    
    with open(output_filename, 'w') as f:
        json.dump(results_payload, f, indent=4)
        
    print(f"Results successfully saved to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="data/windy_tsp/windy_tsp50_val.pkl")
    parser.add_argument("--num_samples", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="results/eval_pycombinatorial")
    parser.add_argument("--max_workers", type=int, default=None)
    main(parser.parse_args())