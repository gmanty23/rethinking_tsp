import os
import time
import numpy as np
import argparse
from tqdm import tqdm

# Import pyCombinatorial modules 
from pyCombinatorial import algorithm

# Import the existing Windy TSP data generator
# Assuming calculate_cost_matrix is available in this path based on your structure
from data.windy_tsp.generate_windy_tsp import generate_windy_tsp_data, calculate_cost_matrix

def run_pycombinatorial_baseline(distance_matrix, method='aco', **kwargs):
    """
    Adapter function to evaluate pyCombinatorial algorithms on an asymmetric graph.
    """
    start_time = time.time()
    
    # 
    if method == 'aco':
        # Ant Colony Optimization
        # Parameters can be tuned in kwargs
        solver = algorithm.ant_colony_optimization(
            distance_matrix=distance_matrix,
            colony_size=kwargs.get('colony_size', 15),
            elite=kwargs.get('elite', 1),
            alpha=kwargs.get('alpha', 1.0),
            beta=kwargs.get('beta', 3.0),
            evaporation=kwargs.get('evaporation', 0.5),
            generations=kwargs.get('generations', 100),
            verbose=False
        )
    elif method == 'farthest_insertion':
        # Farthest Insertion Constructive Heuristic
        solver = algorithm.farthest_insertion(
            distance_matrix=distance_matrix,
            verbose=False
        )
    elif method == 'karp_steele':
        # Karp-Steele Patching (ATSP specific)
        solver = algorithm.karp_steele_patching(
            distance_matrix=distance_matrix,
            verbose=False
        )
    else:
        raise ValueError(f"Method {method} not supported or not ATSP-compatible.")

    # Execute solver - pyCombinatorial usually returns the tour sequence and cost
    tour, cost = solver.run()
    duration = time.time() - start_time
    
    return tour, cost, duration

def main(opts):
    print(f"--- Starting pyCombinatorial Evaluation on Windy TSP ---")
    print(f"Graph Size (N): {opts.graph_size}")
    print(f"Dataset Size: {opts.dataset_size}")
    print(f"Algorithm: {opts.method.upper()}")
    
    # 1. Generate or Load Evaluation Data
    dataset = generate_windy_tsp_data(opts.dataset_size, opts.graph_size)
    
    results_cost = []
    results_time = []
    
    # 2. Iterate through instances
    for i, instance in enumerate(tqdm(dataset, desc=f"Evaluating {opts.method}")):
        coords = instance['loc']
        wind_vector = instance['wind']
        
        # Explicitly calculate the ASYMMETRIC distance matrix
        cost_matrix = calculate_cost_matrix(coords, wind_vector)
        
        # Pass the pre-computed matrix to bypass symmetric Euclidean assumptions
        tour, cost, duration = run_pycombinatorial_baseline(
            distance_matrix=cost_matrix, 
            method=opts.method
        )
        
        results_cost.append(cost)
        results_time.append(duration)

    # 3. Aggregate and Report Results
    avg_cost = np.mean(results_cost)
    avg_time = np.mean(results_time)
    
    print("\n--- Final Results ---")
    print(f"Method: {opts.method.upper()}")
    print(f"Average Tour Cost: {avg_cost:.4f}")
    print(f"Average Execution Time: {avg_time:.4f} seconds/instance")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pyCombinatorial baselines on Windy TSP")
    parser.add_argument("--graph_size", type=int, default=20, help="Number of nodes (N)")
    parser.add_argument("--dataset_size", type=int, default=100, help="Number of instances to evaluate")
    parser.add_argument("--method", type=str, default="aco", choices=['aco', 'farthest_insertion', 'karp_steele'], 
                        help="Baseline algorithm to run")
    
    opts = parser.parse_args()
    main(opts)