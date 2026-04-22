import pickle
import numpy as np

def convert_pkl_to_npz(pkl_path, npz_path):
    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    all_locs = []
    all_dist_matrices = []
    
    for instance in data:
        # 1. Extract the exact keys from your dictionary
        locs = instance['loc'].astype(np.float32)      # Shape: (N, 2)
        wind = instance['wind'].astype(np.float32)      # Shape: (2,)
        alpha = float(instance['alpha'])                # Value: 3.0
        
        # 2. Calculate Euclidean Distance (N x N grid)
        # diff[i, j] is the vector FROM node i TO node j
        diff = locs[np.newaxis, :, :] - locs[:, np.newaxis, :]  # Shape: (N, N, 2)
        euclidean_dist = np.linalg.norm(diff, axis=-1)          # Shape: (N, N)
        
        # 3. Normalized Direction Unit Vectors
        dist_clamped = np.maximum(euclidean_dist, 1e-8)
        u_hat = diff / dist_clamped[..., np.newaxis]            # Shape: (N, N, 2)
        
        # 4. Project Wind onto Direction
        wind_proj = np.sum(u_hat * wind, axis=-1)               # Shape: (N, N)
        
        # 5. Apply Exponential Physics Formula
        # Cost = Dist * exp( -alpha * (wind . direction) )
        exponent = -1.0 * alpha * wind_proj
        multiplier = np.exp(exponent)
        
        cost_matrix = euclidean_dist * multiplier               # Shape: (N, N)
        
        # Ensure the diagonal (cost to stay at the same node) is strictly 0
        np.fill_diagonal(cost_matrix, 0.0)

        all_locs.append(locs)
        all_dist_matrices.append(cost_matrix.astype(np.float32))

    # Stack into RRNCO's expected shape: (Batch, N, 2) and (Batch, N, N)
    locs_tensor = np.stack(all_locs)
    dist_tensor = np.stack(all_dist_matrices)

    print("\n--- Final RRNCO Format ---")
    print(f"locs: shape {locs_tensor.shape} | type {locs_tensor.dtype}")
    print(f"distance_matrix: shape {dist_tensor.shape} | type {dist_tensor.dtype}")

    # Save to .npz
    np.savez(npz_path, locs=locs_tensor, distance_matrix=dist_tensor)
    print(f"\nSuccessfully saved to {npz_path}")

if __name__ == "__main__":
    # Convert your Validation dataset
    convert_pkl_to_npz(
        pkl_path="/home/pfc/gms/code/rethinking_tsp/data/windy_tsp/windy_tsp100_train.pkl", 
        npz_path="windy_tsp100_train.npz"
    )