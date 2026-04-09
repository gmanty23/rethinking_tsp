import os
import csv
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

# =====================================================================
# LIST OF ALL BASELINES
# =====================================================================
# ALL_METHODS = [
#     'nearest_neighbour', 'farthest_insertion', 
#     'cheapest_insertion', 'random_insertion', 'multifragment', 
#     'clarke_wright', 'karp_steele', 'greedy_karp_steele',
#     'aco', 'simulated_annealing', 'tabu_search', 'grasp', 'vns', 
#     'genetic_algorithm', 'large_neighborhood_search', 'brkga', 'q_learning'
#     # 'nearest_insertion', 'grasp', 'bellman_held_karp' # WARNING: O(2^N). Uncomment only for N < 20
# ]
ALL_METHODS = [
    'nearest_insertion'
]

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
    """
    instance_idx, coords, wind_vector, method = args
    alpha = 3.0 # Default based on Rethinking TSP / your generator logic
    cost_matrix = calculate_cost_matrix(coords, wind_vector, alpha)
    
    start_time = time.time()
    
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
    elif method == 'karp_steele':
        tour, cost = algorithm.karp_steele_patching(distance_matrix=cost_matrix, verbose=False)
    elif method == 'greedy_karp_steele':
        tour, cost = algorithm.greedy_karp_steele_patching(distance_matrix=cost_matrix, verbose=False)
    elif method in ['aco', 'simulated_annealing', 'tabu_search', 'grasp', 'vns', 'genetic_algorithm']:
        seed_tour_np, seed_cost = algorithm.nearest_neighbour(distance_matrix=cost_matrix, local_search=False, verbose=False)
        seed_tour = seed_tour_np.tolist() if hasattr(seed_tour_np, 'tolist') else list(seed_tour_np)
        seed_solution = [seed_tour, seed_cost]

        if method == 'aco':
            tour, cost = algorithm.ant_colony_optimization(
                distance_matrix=cost_matrix, ants=cost_matrix.shape[0], iterations=100, 
                alpha=1, beta=2, decay=0.05, local_search=False, verbose=False
            )
        elif method == 'simulated_annealing':
            tour, cost = algorithm.simulated_annealing_tsp(
                distance_matrix=cost_matrix, initial_temperature=1.0, alpha=0.9, 
                temperature_iterations=10, verbose=False
            )
        elif method == 'tabu_search':
            tour, cost = algorithm.tabu_search(
                distance_matrix=cost_matrix, city_tour=seed_solution, 
                iterations=150, tabu_tenure=max(1, int(cost_matrix.shape[0]/3)), verbose=False
            )
        elif method == 'grasp':
                    # Catch the full result instead of forcing a 2-variable unpack
                    result = algorithm.greedy_randomized_adaptive_search_procedure(
                        distance_matrix=cost_matrix, city_tour=seed_tour, 
                        iterations=50, rcl=5, greediness_value=0.5, verbose=False
                    )
                    # Safely extract just the first two items
                    tour, cost = result[0], result[1]
        elif method == 'vns':
            tour, cost = algorithm.variable_neighborhood_search(
                distance_matrix=cost_matrix, city_tour=seed_solution, 
                max_attempts=20, neighbourhood_size=5, iterations=50, verbose=False
            )
        elif method == 'genetic_algorithm':
            tour, cost = algorithm.genetic_algorithm(
                distance_matrix=cost_matrix, population_size=20, 
                generations=100, mutation_rate=0.1, verbose=False
            )
    elif method == 'large_neighborhood_search':
        tour, cost = algorithm.large_neighborhood_search(
            distance_matrix=cost_matrix, iterations=150, neighborhood_size=4, 
            local_search=False, verbose=False
        )
    elif method == 'brkga':
        tour, cost = algorithm.biased_random_key_genetic_algorithm(
            distance_matrix=cost_matrix, population_size=20, generations=100, verbose=False
        )
    elif method == 'q_learning':
        result = algorithm.q_learning(
            distance_matrix=cost_matrix, learning_rate=0.1, discount_factor=0.95, 
            epsilon=0.15, episodes=5000, local_search=False, verbose=False
        )
        tour, cost = result[0], result[1]
    elif method == 'bellman_held_karp':
        tour, cost = algorithm.bellman_held_karp_exact_algorithm(distance_matrix=cost_matrix, verbose=False)
    else:
        raise ValueError(f"Method '{method}' not found or incompatible with ATSP Windy TSP.")

    duration = time.time() - start_time
    
    return instance_idx, cost, duration, tour

def run_evaluation(method, dataset, lkh_costs, opts, N_nodes, csv_filename):
    """Executes the evaluation for a single method and appends to CSV."""
    print(f"\n[{method.upper()}] Preparing tasks...")
    
    tasks = []
    for i, instance in enumerate(dataset):
        if isinstance(instance, dict):
            coords, wind_vector = instance['loc'], instance['wind']
        else:
            coords, wind_vector = instance[0], instance[1]
        tasks.append((i, coords, wind_vector, method))

    results_cost = [0] * len(tasks)
    results_time = [0] * len(tasks)

    with concurrent.futures.ProcessPoolExecutor(max_workers=opts.max_workers) as executor:
        futures = {executor.submit(evaluate_single_instance, task): task for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc=f"Evaluating {method}"):
            try:
                idx, cost, duration, tour = future.result()
                results_cost[idx] = cost
                results_time[idx] = duration
            except Exception as exc:
                print(f"Instance generated an exception: {exc}")

    avg_cost = np.mean(results_cost)
    avg_time = np.mean(results_time)
    
    # Gap MoR
    gaps = [(c - l) / l for c, l in zip(results_cost, lkh_costs)]
    gap_mor = np.mean(gaps) * 100.0
    
    print(f"[{method.upper()}] Avg Cost: {avg_cost:.4f} | Gap MoR: {gap_mor:.2f}% | Avg Time: {avg_time:.4f}s")

    # Append to CSV
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Baseline', 'Avg_Cost', 'Gap_MoR_pct', 'Time_per_Solution_sec'])
        writer.writerow([method, avg_cost, gap_mor, avg_time])

def main(opts):
    os.makedirs(opts.output_dir, exist_ok=True)
    
    # 1. Load Evaluation Data (Once)
    dataset = load_windy_dataset(opts.dataset_path, num_samples=opts.num_samples)
    N_nodes = len(dataset[0]['loc']) if isinstance(dataset[0], dict) else len(dataset[0][0])
    print(f"Loaded {len(dataset)} instances (N={N_nodes}).")

    # 2. Load LKH Baseline Data (Once)
    print(f"Loading LKH baseline from {opts.lkh_path}...")
    with open(opts.lkh_path, 'rb') as f:
        lkh_data = pickle.load(f)
    
    # If it's a zip iterator, exhaust it into a list
    if type(lkh_data).__name__ == 'zip':
        lkh_data = list(lkh_data)
        print(f"Unpacked zip object. First item looks like this:\n{lkh_data[0]}")
    
    lkh_costs = []
    for item in lkh_data:
        # If the item is a tuple or list (e.g., from a zip object)
        if isinstance(item, (tuple, list)):
            # We assume cost is a float or int. 
            # If your tuple is (tour, cost) or (cost, tour), this finds the number.
            # It explicitly avoids lists/arrays (which are the tours).
            numeric_values = [x for x in item if isinstance(x, (float, int)) and not isinstance(x, bool)]
            
            if len(numeric_values) == 1:
                lkh_costs.append(numeric_values[0])
            elif len(numeric_values) >= 2:
                # If there are multiple numbers (e.g., cost and time), 
                # cost is usually much larger than time, or you can hardcode the index here if needed.
                # We will just grab the first numeric value for now. 
                lkh_costs.append(numeric_values[0]) 
                
        # Fallbacks for other common structures
        elif isinstance(item, dict):
            lkh_costs.append(item.get('cost', item.get('obj')))
        elif isinstance(item, (float, int)):
            lkh_costs.append(item)
    
    lkh_costs = lkh_costs[:opts.num_samples]
    print(f"Successfully extracted {len(lkh_costs)} LKH costs. Average LKH cost: {np.mean(lkh_costs):.4f}")

    # 3. Determine Methods to Run
    methods_to_run = ALL_METHODS if opts.method.lower() == 'all' else [opts.method]
    
    csv_filename = os.path.join(opts.output_dir, f"baselines_summary_N{N_nodes}.csv")
    print(f"\nOutput will be appended to: {csv_filename}")
    print(f"Methods scheduled to run: {len(methods_to_run)}\n" + "-"*50)

    # 4. Iterate and Evaluate
    for method in methods_to_run:
        run_evaluation(method, dataset, lkh_costs, opts, N_nodes, csv_filename)
        
    print("\n--- ALL EVALUATIONS COMPLETE ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Now accepts "all" to run the entire suite automatically
    parser.add_argument("--method", type=str, default="all", help="Specific method name or 'all' to run every baseline")
    parser.add_argument("--dataset_path", type=str, default="data/windy_tsp/windy_tsp50_val.pkl")
    parser.add_argument("--lkh_path", type=str, default="results/lkh_windy/windy_tsp50_val.pkl", help="Path to LKH baseline results")
    parser.add_argument("--num_samples", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="results/eval_pycombinatorial")
    parser.add_argument("--max_workers", type=int, default=20)
    main(parser.parse_args())