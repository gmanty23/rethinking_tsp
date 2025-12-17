#PROBLEM: COORDINATES FEATURES IS DOING TOO GOOD AND THE OTHERS TOO BAD
#HYPOTHESES:
# 1) THE WIND PARAMETER 'ALPHA' IS TOO LOW, MAKING THE PROBLEM TOO CLOSE TO EUCLIDEAN TSP
# 2) THE MODEL ARCHITECTURE IS NOT SUITED TO LEARN THE WIND EFFECT (E.G., NOT ENOUGH CAPACITY)
#3) THE TRAINING PROCEDURE (E.G., LR, BASELINE) IS NOT SUITED TO LEARN THE WIND EFFECT  
# THIS CODE CHECKS THE DATA STATS TO DIAGNOSE HYPOTHESIS 1

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from tqdm import tqdm

# ==========================================
# 1. GENERATOR LOGIC (Mimics codebase)
# ==========================================
def generate_windy_instance(num_nodes, alpha=5.0, max_wind=0.5):
    """
    Replicates the logic found in test/windy_tsp_visualization.py 
    and used by problems/tsp/problem_tsp.py[cite: 440].
    """
    # Random locations [0, 1] x [0, 1]
    loc = np.random.rand(num_nodes, 2)
    
    # Random wind vector
    wind_angle = np.random.uniform(0, 2 * np.pi)
    wind_mag = np.random.uniform(0, max_wind)
    wind = np.array([wind_mag * np.cos(wind_angle), wind_mag * np.sin(wind_angle)])
    
    return {'loc': loc, 'wind': wind, 'alpha': alpha}

# ==========================================
# 2. ANALYSIS LOGIC
# ==========================================
def check_dominance_on_the_fly():
    # --- CONFIGURATION (Change these to test your hypothesis) ---
    NUM_SAMPLES = 1000
    MIN_SIZE = 20
    MAX_SIZE = 20  # Keep fixed size for consistent analysis
    
    # Current Default Parameters in your code
    ALPHA = 3.0     
    MAX_WIND = 1.0 
    
    print(f"Generating {NUM_SAMPLES} samples on the fly (Nodes: {MIN_SIZE}-{MAX_SIZE})...")
    print(f"Parameters: Alpha={ALPHA}, Max_Wind={MAX_WIND}")
    
    correlations = []
    asymmetry_ratios = []
    wind_multipliers = []
    
    # Mimic the loop in TSPDataset [cite: 439]
    for _ in tqdm(range(NUM_SAMPLES)):
        # 1. Random size generation (as done in problem_tsp.py)
        num_nodes = np.random.randint(low=MIN_SIZE, high=MAX_SIZE+1)
        
        # 2. Generate Instance
        sample = generate_windy_instance(num_nodes, alpha=ALPHA, max_wind=MAX_WIND)
        
        loc = sample['loc']
        wind = sample['wind']
        alpha = sample['alpha']
        
        # 3. Calculate Physics (Same as previous script)
        # Euclidean Distances
        diff = loc[None, :, :] - loc[:, None, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, 1.0) # Prevent div/0
        
        # Unit Vectors
        u = diff / dists[:, :, None]
        
        # Costs
        wind_proj = np.dot(u, wind)
        multiplier = np.exp(-1.0 * alpha * wind_proj)
        costs = dists * multiplier
        
        # Clean diagonals
        np.fill_diagonal(dists, 0)
        np.fill_diagonal(costs, 0)
        np.fill_diagonal(multiplier, 1)

        # --- METRICS ---
        mask = ~np.eye(num_nodes, dtype=bool)
        flat_dist = dists[mask]
        flat_cost = costs[mask]
        
        # Metric A: Correlation
        corr, _ = scipy.stats.pearsonr(flat_dist, flat_cost)
        correlations.append(corr)
        
        # Metric B: Asymmetry Ratio
        c_ij = costs
        c_ji = costs.T
        abs_diff = np.abs(c_ij - c_ji)
        mean_c = (c_ij + c_ji) / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            asym = abs_diff / mean_c
        asymmetry_ratios.append(np.mean(asym[mask]))
        
        # Metric C: Multiplier
        wind_multipliers.extend(multiplier[mask])

    # === REPORTING ===
    avg_corr = np.mean(correlations)
    avg_asym = np.mean(asymmetry_ratios)
    
    print("\n" + "="*40)
    print(f" DIAGNOSTIC RESULTS (Alpha={ALPHA}, Wind={MAX_WIND})")
    print("="*40)
    print(f"1. Euclidean Dominance (Correlation): {avg_corr:.4f}")
    print(f"   (> 0.90 means Distance is the only thing that matters)")
    
    print(f"2. Asymmetry Strength: {avg_asym:.4f}")
    print(f"   (< 0.10 means the problem is basically symmetric)")

    # Plotting
    plt.figure(figsize=(14, 4))
    
    # Plot 1: Cost vs Dist
    plt.subplot(1, 3, 1)
    plt.scatter(flat_dist[:2000], flat_cost[:2000], alpha=0.1, s=5, c='blue')
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Windy Cost")
    plt.title(f"Cost vs Dist (Corr: {avg_corr:.3f})")
    
    # Plot 2: Asymmetry Histogram
    plt.subplot(1, 3, 2)
    plt.hist(asymmetry_ratios, bins=30, color='orange', alpha=0.7)
    plt.xlabel("Avg Asymmetry Ratio")
    plt.title("Asymmetry Distribution")
    
    # Plot 3: Multiplier Histogram
    plt.subplot(1, 3, 3)
    plt.hist(wind_multipliers, bins=50, color='green', alpha=0.7)
    plt.xlabel("Cost Multiplier (Base=1.0)")
    plt.title(f"Wind Effect (Alpha={ALPHA})")
    plt.axvline(1.0, color='k', linestyle='--')
    
    plt.tight_layout()
    plot_filename = f"dominance_check_a{ALPHA}_w{MAX_WIND}.png"
    plt.savefig(plot_filename)
    print(f"\nSaved plots to '{plot_filename}'")

if __name__ == "__main__":
    check_dominance_on_the_fly()