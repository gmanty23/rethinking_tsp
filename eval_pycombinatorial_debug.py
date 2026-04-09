# import time
# import traceback
# import pickle
# import numpy as np

# # Import library and your matrix generator
# from pyCombinatorial import algorithm
# from data.windy_tsp.generate_windy_tsp import calculate_cost_matrix

# def load_single_instance(filepath="data/windy_tsp/windy_tsp20_val.pkl"):
#     """Loads just the very first instance of the dataset."""
#     with open(filepath, 'rb') as f:
#         data = pickle.load(f)
#     return data[0]

# def run_advanced_debug():
#     print("====================================================")
#     print("🔬 DEBUGGING ADVANCED METHODS (N=20, 1 Sample)")
#     print("====================================================\n")
    
#     # 1. Prepare Data
#     instance = load_single_instance()
#     coords, wind_vector = (instance['loc'], instance['wind']) if isinstance(instance, dict) else (instance[0], instance[1])
#     cost_matrix = calculate_cost_matrix(coords, wind_vector, alpha=3.0)
    
#     # 2. Define the tests with specific parameters to keep them fast
#     tests = [
#         ("Q-Learning", 'q_learning', {
#             'episodes': 100,         # Massively reduced for quick debugging
#             'local_search': False,   # Disable 2-opt for asymmetric matrix
#             'verbose': False
#         }),
#         ("Large Neighborhood Search (LNS)", 'large_neighborhood_search', {
#             'iterations': 50,
#             'neighborhood_size': 4,
#             'local_search': False,   # Disable 2-opt
#             'verbose': False
#         }),
#         ("BRKGA", 'biased_random_key_genetic_algorithm', {
#             'population_size': 20,
#             'generations': 50,       # Massively reduced (default is 50,000)
#             # BRKGA does not have a local_search argument in its signature
#             'verbose': False
#         })
#     ]
    
#     for name, method_name, params in tests:
#         print(f"🚀 Testing: {name} ({method_name})")
#         start_time = time.time()
#         try:
#             # Fetch the function dynamically from the algorithm module
#             func = getattr(algorithm, method_name)
            
#             # Run the function passing the NumPy matrix
#             tour, cost = func(distance_matrix=cost_matrix, **params)
            
#             duration = time.time() - start_time
#             print(f"   ✅ SUCCESS!")
#             print(f"   -> Cost: {cost:.4f}")
#             print(f"   -> Time: {duration:.4f} seconds\n")
            
#         except Exception as e:
#             print(f"   ❌ FAILED: {e}")
#             print("   --- Traceback ---")
#             traceback.print_exc()
#             print("   -----------------\n")

# if __name__ == "__main__":
#     run_advanced_debug()


import os
import pickle
import numpy as np

# Hide GPU just in case
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pyCombinatorial import algorithm
from data.windy_tsp.generate_windy_tsp import calculate_cost_matrix

def load_windy_dataset(filepath, num_samples=5):
    """Loads just a few samples for quick debugging."""
    print(f"Loading {num_samples} samples from {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data[:num_samples]

def evaluate_true_cost(tour, cost_matrix):
    """
    Manually calculates the exact cost of a given tour using the asymmetric matrix.
    Handles whether the library implicitly closed the loop or not, and fixes 1-based indexing.
    """
    # Convert to list if it's a numpy array
    if hasattr(tour, 'tolist'):
        tour = tour.tolist()
        
    # FIX: Check if pyCombinatorial returned 1-based node IDs (1 to 20)
    # If so, shift everything down by 1 so numpy can read it (0 to 19)
    if min(tour) == 1:
        tour = [int(node) - 1 for node in tour]
    else:
        tour = [int(node) for node in tour]
        
    actual_cost = 0.0
    for i in range(len(tour) - 1):
        actual_cost += cost_matrix[tour[i], tour[i+1]]
        
    # If the tour array doesn't explicitly return to the start node, add the return trip
    if tour[0] != tour[-1]:
        actual_cost += cost_matrix[tour[-1], tour[0]]
        
    return actual_cost

def run_sanity_check():
    dataset_path = "data/windy_tsp/windy_tsp20_val.pkl"
    alpha = 3.0
    
    dataset = load_windy_dataset(dataset_path, num_samples=5)
    
    methods_to_test = [
        'nearest_neighbour',
        'nearest_insertion',
        'farthest_insertion',
        'cheapest_insertion'
    ]
    
    print("\n" + "="*70)
    print(f"{'Method':<20} | {'Instance':<8} | {'Lib Cost':<10} | {'True Cost':<10} | {'Match?'}")
    print("="*70)
    
    for method in methods_to_test:
        for i, instance in enumerate(dataset):
            # Extract data
            if isinstance(instance, dict):
                coords, wind_vector = instance['loc'], instance['wind']
            else:
                coords, wind_vector = instance[0], instance[1]
                
            # Generate EXACT asymmetric matrix
            cost_matrix = calculate_cost_matrix(coords, wind_vector, alpha)
            
            # Run the library method
            if method == 'nearest_neighbour':
                tour, lib_cost = algorithm.nearest_neighbour(distance_matrix=cost_matrix, local_search=False, verbose=False)
            elif method == 'nearest_insertion':
                tour, lib_cost = algorithm.nearest_insertion(distance_matrix=cost_matrix, verbose=False)
            elif method == 'farthest_insertion':
                tour, lib_cost = algorithm.farthest_insertion(distance_matrix=cost_matrix, local_search=False, verbose=False)
            elif method == 'cheapest_insertion':
                tour, lib_cost = algorithm.cheapest_insertion(distance_matrix=cost_matrix, local_search=False, verbose=False)
            
            # Calculate the REAL cost of the returned tour
            true_cost = evaluate_true_cost(tour, cost_matrix)
            
            # Check if they match (allowing for tiny floating point differences)
            match = "YES" if abs(lib_cost - true_cost) < 1e-4 else "NO ❌"
            
            print(f"{method:<20} | {i:<8} | {lib_cost:<10.4f} | {true_cost:<10.4f} | {match}")
        print("-" * 70)

if __name__ == "__main__":
    run_sanity_check()