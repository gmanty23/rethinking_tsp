import pickle
import numpy as np
import torch

# Load your generated Windy TSP validation set
filepath = "data/windy_tsp/windy_tsp50_val.pkl"
print(f"Loading {filepath}...\n")

with open(filepath, 'rb') as f:
    data = pickle.load(f)

print(f"Dataset type: {type(data)}")

# NCO datasets are usually lists of graph instances
if isinstance(data, list):
    print(f"Number of instances: {len(data)}")
    
    # Grab the very first graph in the dataset
    first_instance = data[0]
    print(f"\nStructure of the FIRST instance (type: {type(first_instance)}):")
    
    if isinstance(first_instance, dict):
        # Print the keys and the shapes of their contents
        for key, value in first_instance.items():
            if hasattr(value, 'shape'):
                print(f"  - '{key}': shape {value.shape} | type {type(value).__name__}")
            elif isinstance(value, list) or isinstance(value, tuple):
                print(f"  - '{key}': length {len(value)} | type {type(value).__name__}")
            else:
                print(f"  - '{key}': value {value} | type {type(value).__name__}")
    else:
        print("  - Instance is not a dictionary. Raw data:")
        if hasattr(first_instance, 'shape'):
            print(f"    Shape: {first_instance.shape}")
        else:
            print(f"    {first_instance}")
else:
    print("Dataset is not a list. Inspecting raw object...")
    print(data)