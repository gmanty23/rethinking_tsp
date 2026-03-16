import time
import traceback
import pickle
import numpy as np

# Import library and your matrix generator
from pyCombinatorial import algorithm
from data.windy_tsp.generate_windy_tsp import calculate_cost_matrix

def load_single_instance(filepath="data/windy_tsp/windy_tsp20_val.pkl"):
    """Loads just the very first instance of the dataset."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data[0]

def run_advanced_debug():
    print("====================================================")
    print("🔬 DEBUGGING ADVANCED METHODS (N=20, 1 Sample)")
    print("====================================================\n")
    
    # 1. Prepare Data
    instance = load_single_instance()
    coords, wind_vector = (instance['loc'], instance['wind']) if isinstance(instance, dict) else (instance[0], instance[1])
    cost_matrix = calculate_cost_matrix(coords, wind_vector, alpha=3.0)
    
    # 2. Define the tests with specific parameters to keep them fast
    tests = [
        ("Q-Learning", 'q_learning', {
            'episodes': 100,         # Massively reduced for quick debugging
            'local_search': False,   # Disable 2-opt for asymmetric matrix
            'verbose': False
        }),
        ("Large Neighborhood Search (LNS)", 'large_neighborhood_search', {
            'iterations': 50,
            'neighborhood_size': 4,
            'local_search': False,   # Disable 2-opt
            'verbose': False
        }),
        ("BRKGA", 'biased_random_key_genetic_algorithm', {
            'population_size': 20,
            'generations': 50,       # Massively reduced (default is 50,000)
            # BRKGA does not have a local_search argument in its signature
            'verbose': False
        })
    ]
    
    for name, method_name, params in tests:
        print(f"🚀 Testing: {name} ({method_name})")
        start_time = time.time()
        try:
            # Fetch the function dynamically from the algorithm module
            func = getattr(algorithm, method_name)
            
            # Run the function passing the NumPy matrix
            tour, cost = func(distance_matrix=cost_matrix, **params)
            
            duration = time.time() - start_time
            print(f"   ✅ SUCCESS!")
            print(f"   -> Cost: {cost:.4f}")
            print(f"   -> Time: {duration:.4f} seconds\n")
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            print("   --- Traceback ---")
            traceback.print_exc()
            print("   -----------------\n")

if __name__ == "__main__":
    run_advanced_debug()