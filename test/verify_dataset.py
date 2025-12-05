import pickle
import numpy as np

val_path = 'data/windy_tsp/windy_tsp20_val.pkl'

print(f"Checking {val_path}...")
with open(val_path, 'rb') as f:
    data = pickle.load(f)
    if not isinstance(data, list):
        print("Data is not a list!")
    # Check if data is empty
    if len(data) == 0:
        print("Data list is empty!")

for i, item in enumerate(data):
    loc = item['loc']
    # Check for NaNs in raw coordinates
    if np.isnan(loc).any():
        print(f"Found NaN in sample {i} locations!")
    
    # Check for potential Division by Zero (Duplicate points)
    diff = loc[None, :, :] - loc[:, None, :]
    dists = np.linalg.norm(diff, axis=-1)
    # Diagonal is always 0, but if off-diagonal is 0, it causes 0/0 in 'u' calculation
    np.fill_diagonal(dists, 1.0) 
    if (dists == 0).any():
        print(f"Found duplicate points (dist=0) in sample {i}! This will cause NaNs.")