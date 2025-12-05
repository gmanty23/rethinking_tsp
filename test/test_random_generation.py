import torch
import numpy as np
from tqdm import tqdm
from data.windy_tsp.generate_windy_tsp import generate_windy_instance

def check_generator():
    print("Checking 10,000 generated instances for NaNs...")
    num_samples = 10000
    min_size = 20
    max_size = 20
    
    nans_found = 0
    
    for _ in tqdm(range(num_samples)):
        # Simulate the dataset generation logic
        num_nodes = np.random.randint(low=min_size, high=max_size+1)
        data = generate_windy_instance(num_nodes, alpha=2.0, max_wind=0.5)
        
        loc = data['loc']
        
        # Check coordinates (used in 'coords' mode)
        if np.isnan(loc).any() or np.isinf(loc).any():
            print(f"FAIL: Found NaNs/Infs in 'loc' (Coordinates)!")
            print(f"Sample:\n{loc}")
            nans_found += 1
            break
            
        # Check derived physics (used in 'learned'/'hybrid' mode)
        wind = data['wind']
        if np.isnan(wind).any():
            print("FAIL: Found NaNs in 'wind'!")
            nans_found += 1
            break

    if nans_found == 0:
        print("SUCCESS: No NaNs found in 10,000 samples. The issue might be elsewhere.")
    else:
        print("FAILURE: Generator is producing bad data.")

if __name__ == "__main__":
    check_generator()