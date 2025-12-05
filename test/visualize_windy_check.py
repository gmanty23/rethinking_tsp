import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import os
import pickle
from problems.tsp.problem_tsp import TSPDataset, WindyTSP

def get_full_cost_matrix(dataset, num_nodes):
    item = dataset[0] # Single item
    features = item['nodes'] # (N, 7)
    
    loc = features[:, 0:2].numpy()
    wind = features[0, 2:4].numpy()
    alpha = features[0, 4].item()
    
    print(f"Visualization Config:")
    print(f"  Wind Vector: [{wind[0]:.2f}, {wind[1]:.2f}]")
    print(f"  Alpha: {alpha}")
    
    n = len(loc)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            diff = loc[j] - loc[i]
            dist = np.linalg.norm(diff)
            u = diff / dist
            proj = np.dot(u, wind)
            cost = dist * np.exp(-1.0 * alpha * proj)
            matrix[i, j] = cost
            
    return loc, wind, matrix

def plot_windy_graph(loc, wind, matrix):
    fig, ax = plt.subplots(figsize=(10, 8))
    n = len(loc)
    
    # 1. Plot Nodes
    ax.scatter(loc[:, 0], loc[:, 1], c='black', zorder=10, s=100)
    for i in range(n):
        ax.text(loc[i, 0]+0.02, loc[i, 1]+0.02, f"Node {i}", fontsize=12, weight='bold')

    # 2. Plot Wind Vector
    center = np.mean(loc, axis=0)
    wind_mag = np.linalg.norm(wind)
    # Scale arrow for visibility in plot, roughly length 0.4
    arrow_scale = 0.4 / (wind_mag if wind_mag > 1e-6 else 1.0) 
    
    ax.arrow(center[0]-0.2, center[1], wind[0]*arrow_scale, wind[1]*arrow_scale, 
             head_width=0.05, head_length=0.1, fc='lightblue', ec='lightblue', width=0.02, zorder=0)
    
    ax.text(center[0], center[1]-0.15, f"WIND\nMag: {wind_mag:.2f}", color='lightblue', fontsize=12, weight='bold', ha='center')

    # 3. Plot Edges
    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            dist = np.linalg.norm(loc[j] - loc[i])
            cost = matrix[i, j]
            
            if cost < dist:
                color = 'green'
                weight = 2
            else:
                color = 'red'
                weight = 1
                
            # Curve Left to match text position
            arrow = patches.FancyArrowPatch(
                (loc[i, 0], loc[i, 1]), 
                (loc[j, 0], loc[j, 1]),
                connectionstyle="arc3,rad=-0.2", 
                arrowstyle="->",
                color=color,
                linewidth=weight,
                mutation_scale=20,
                zorder=5
            )
            ax.add_patch(arrow)
            
            # Label the cost on the curve
            mid = (loc[i] + loc[j]) / 2
            perp = np.array([-(loc[j,1]-loc[i,1]), loc[j,0]-loc[i,0]])
            perp = perp / np.linalg.norm(perp)
            text_pos = mid + perp * 0.08
            
            label = f"{cost:.2f}"
            ax.text(text_pos[0], text_pos[1], label, color=color, fontsize=9, ha='center', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.set_title(f"Visualizing Asymmetry in Windy TSP\nGreen=Cheaper than Dist, Red=More Expensive", fontsize=14)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    output_file = "windy_tsp_visualization.png"
    plt.savefig(output_file)
    print(f"\n[Graphic] Plot saved to {output_file}")

def run_visual_test():
    filename = "debug_windy_viz.pkl"
    
    # Randomize Wind
    # We use None to ensure it pulls from OS entropy, effectively ignoring any fixed seed set elsewhere
    rng = np.random.default_rng() 
    angle = rng.uniform(0, 2 * np.pi)
    mag = rng.uniform(0.5, 1.5) # Random magnitude between 0.5 and 1.5
    wind = np.array([mag * np.cos(angle), mag * np.sin(angle)])
    
    print(f"Generated Random Wind: Angle={np.degrees(angle):.1f} deg, Magnitude={mag:.2f}")

    mock_instance = {
        'loc': np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 
        'wind': wind, 
        'alpha': 1.0
    }
    with open(filename, 'wb') as f:
        pickle.dump([mock_instance], f)

    dataset = TSPDataset(filename=filename, node_feature_type='hybrid', batch_size=1, num_samples=1)
    loc, wind, matrix = get_full_cost_matrix(dataset, 3)
    
    plot_windy_graph(loc, wind, matrix)
    
    if os.path.exists(filename):
        os.remove(filename)

if __name__ == "__main__":
    run_visual_test()