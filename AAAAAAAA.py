# import pickle
# import numpy as np

# def analyze_dataset(path):
#     with open(path, 'rb') as f:
#         data = pickle.load(f)
    
#     alphas = []
#     magnitudes = []
    
#     for i, instance in enumerate(data):
#         # Depending on how your data is structured (dict or list/tuple)
#         # Assuming the structure from your previous output:
#         if isinstance(instance, dict):
#             alphas.append(instance.get('alpha', 0))
#             wind_vec = np.array(instance.get('wind', [0, 0]))
#             magnitudes.append(np.linalg.norm(wind_vec))
#         else:
#             # If it's the standard Rethinking TSP format, it might be nested
#             # Adjust this part if the logic below fails
#             pass

#     print(f"--- Dataset Analysis: {path} ---")
#     print(f"Total Instances: {len(data)}")
#     print(f"Alpha (Mean):    {np.mean(alphas):.2f} (Std: {np.std(alphas):.4f})")
#     print(f"Wind Magnitude:")
#     print(f"  - Mean:        {np.mean(magnitudes):.4f}")
#     print(f"  - Max:         {np.max(magnitudes):.4f}")
#     print(f"  - Min:         {np.min(magnitudes):.4f}")
#     print(f"  - Variance:    {np.var(magnitudes):.4f}")

# if __name__ == "__main__":
#     DATASET_PATH = "data/windy_tsp/windy_tsp20_val.pkl"
#     analyze_dataset(DATASET_PATH)


import pickle

def check_pkl_length(filepath):
    print(f"[*] Checking: {filepath}")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        # Ensure it's measurable
        if not isinstance(data, list):
            data = list(data)
            
        print(f"    -> Total instances: {len(data)}\n")
        
    except FileNotFoundError:
        print("    -> Error: File not found. Please check the path.\n")
    except Exception as e:
        print(f"    -> Error loading file: {e}\n")

if __name__ == "__main__":
    # Your file paths
    DATASET = "data/windy_tsp/windy_tsp20_val.pkl"
    LKH_BASELINE = "results/windy_tsp20_val/windy_tsp20_valn1280-lkh_windy.pkl"
    
    check_pkl_length(DATASET)
    check_pkl_length(LKH_BASELINE)