import argparse
import numpy as np
import os
import pickle
import time
import pprint as pp

def generate_windy_instance(num_nodes, alpha=1.0, max_wind=0.5):
    """
    Generates a single Windy TSP instance using an Exponential Cost formulation.
    
    Args:
        num_nodes: Number of cities.
        alpha: Coefficient for wind effect.
        max_wind: Maximum magnitude of the wind vector.
        
    Returns:
        dict: {
            'loc': np.array (N x 2), coordinates
            'wind': np.array (2,), global wind vector
            'alpha': float
        }
    """
    # 1. Generate Locations: Uniformly in [0, 1] square
    loc = np.random.rand(num_nodes, 2)
    
    # 2. Generate Global Wind Vector
    # We generate a random direction and a random magnitude
    wind_angle = np.random.uniform(0, 2 * np.pi)
    wind_mag = np.random.uniform(0, max_wind)
    wind = np.array([wind_mag * np.cos(wind_angle), wind_mag * np.sin(wind_angle)])
    
    # NOTE: With Exponential formulation, cost is always positive.

    return {
        'loc': loc,
        'wind': wind,
        'alpha': alpha
    }

def calculate_cost_matrix(loc, wind, alpha):
    """
    Calculates the Asymmetric Cost Matrix using the robust Exponential formulation.
    
    Formula: C_ij = ||p_j - p_i|| * exp( -alpha * (wind . u_ij) )
    
    Interpretation (assuming alpha > 0):
    - Moving WITH wind (wind . u_ij > 0): 
        Exponent is negative -> Factor < 1 -> Cost DECREASES (Easier).
    - Moving AGAINST wind (wind . u_ij < 0): 
        Exponent is positive -> Factor > 1 -> Cost INCREASES (Harder).
    
    Args:
        loc: (N, 2) coordinates
        wind: (2,) wind vector
        alpha: float coefficient
    """
    n = len(loc)
    coords = loc
    
    # Differences: (N, N, 2)
    diff = coords[None, :, :] - coords[:, None, :] # j - i
    
    # Euclidean Distances: (N, N)
    dists = np.linalg.norm(diff, axis=-1)
    
    # Avoid division by zero for diagonal
    dists[np.arange(n), np.arange(n)] = 1.0 
    
    # Unit vectors u_ij: (N, N, 2)
    u = diff / dists[:, :, None]
    u[np.arange(n), np.arange(n)] = 0 # Fix diagonal
    
    # Project wind onto unit vectors: (N, N)
    # wind is (2,), u is (N, N, 2) -> dot product over last axis
    wind_proj = np.dot(u, wind)
    
    # --- ROBUST FORMULATION CHANGE ---
    # We use a negative sign so that tailwind reduces cost.
    costs = dists * np.exp(-1.0 * alpha * wind_proj)
    
    # Restore diagonal to 0
    costs[np.arange(n), np.arange(n)] = 0
    
    return costs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_nodes", type=int, default=20)
    parser.add_argument("--max_nodes", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    
    # Windy TSP specific args
    parser.add_argument("--alpha", type=float, default=2.0, help="Wind influence coefficient")
    parser.add_argument("--max_wind", type=float, default=0.5, help="Maximum wind magnitude")
    
    opts = parser.parse_args()
    
    np.random.seed(opts.seed)
    
    if opts.filename is None:
        opts.filename = f"data/windy_tsp/windy_data/windy_tsp{opts.min_nodes}-{opts.max_nodes}_len{opts.num_samples}.pkl"
    
    os.makedirs(os.path.dirname(opts.filename) if os.path.dirname(opts.filename) else '.', exist_ok=True)
    
    pp.pprint(vars(opts))
    
    dataset = []
    
    start_time = time.time()
    for _ in range(opts.num_samples):
        # Sample graph size
        num_nodes = np.random.randint(low=opts.min_nodes, high=opts.max_nodes + 1)
        
        # Generate Instance
        instance = generate_windy_instance(num_nodes, opts.alpha, opts.max_wind)
        dataset.append(instance)
        
    end_time = time.time() - start_time
    
    # Verification Step: Check one instance for asymmetry
    sample = dataset[0]
    C = calculate_cost_matrix(sample['loc'], sample['wind'], sample['alpha'])
    is_symmetric = np.allclose(C, C.T)
    
    print(f"\n[Verification] Sample 0 Asymmetry Check (Exponential Formulation):")
    print(f"  Wind Vector: {sample['wind']}")
    print(f"  Alpha: {sample['alpha']}")
    
    # Check direction logic
    # Find two nodes with significant wind alignment
    n = len(sample['loc'])
    # Recalculate diffs and projs manually for verification log
    coords = sample['loc']
    diff = coords[None, :, :] - coords[:, None, :]
    dists = np.linalg.norm(diff, axis=-1)
    u = np.zeros_like(diff)
    non_diag = dists > 1e-9
    u[non_diag] = diff[non_diag] / dists[non_diag][:, None]
    projs = np.dot(u, sample['wind'])
    
    # Find max tailwind (proj > 0)
    # flatten
    flat_idxs = np.argmax(projs)
    i, j = np.unravel_index(flat_idxs, projs.shape)
    
    print(f"  Max Tailwind Edge ({i}->{j}):")
    print(f"    Distance: {dists[i,j]:.4f}")
    print(f"    Wind Proj: {projs[i,j]:.4f} (Should be > 0)")
    print(f"    Cost ({i}->{j}): {C[i,j]:.4f} (Should be < Distance)")
    print(f"    Cost ({j}->{i}): {C[j,i]:.4f} (Should be > Distance)")
    
    if C[i,j] < dists[i,j] and C[j,i] > dists[j,i]:
        print("  >> Success: Wind aids movement (Lower Cost) and hinders return (Higher Cost).")
    else:
        print("  >> FAILURE: Cost logic mismatch.")
    
    if not is_symmetric:
        print("  >> Success: Asymmetry confirmed.")
    
    # Save as Pickle
    with open(opts.filename, "wb") as f:
        pickle.dump(dataset, f)
        
    print(f"\nCompleted generation of {opts.num_samples} samples.")
    print(f"Saved to {opts.filename}")
    print(f"Total time: {end_time:.1f}s")