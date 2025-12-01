import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import os
import pickle
from problems.tsp.problem_tsp import TSPDataset, WindyTSP

def get_full_cost_matrix(dataset, num_nodes):
    """
    Helper to extract the NxN cost matrix from the WindyTSP logic.
    We simulate a batch of 'all possible tours' to extract edge costs.
    """
    # 1. Create a dummy tour just to use get_costs structure, 
    # but effectively we want to compute costs for all pairs.
    # Actually, we can just reimplement the cost logic simply here for visualization 
    # using the exact same formula to ensure 1:1 match.
    
    # Unpack
    item = dataset[0] # Single item
    features = item['nodes'] # (N, 7)
    
    loc = features[:, 0:2].numpy()
    wind = features[0, 2:4].numpy() # Global wind
    alpha = features[0, 4].item()
    
    print(f"Visualization Config:")
    print(f"  Wind: {wind}, Alpha: {alpha}")
    
    n = len(loc)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            # Vector i -> j
            diff = loc[j] - loc[i]
            dist = np.linalg.norm(diff)
            u = diff / dist
            
            # Proj
            proj = np.dot(u, wind)
            
            # Cost Formula
            # Cost = Dist * exp(-alpha * proj)
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

    # 2. Plot Wind Vector (Big Blue Arrow in background)
    # Center of map
    center = np.mean(loc, axis=0)
    ax.arrow(center[0]-0.2, center[1], wind[0]*0.4, wind[1]*0.4, 
             head_width=0.05, head_length=0.1, fc='lightblue', ec='lightblue', width=0.02, zorder=0)
    ax.text(center[0], center[1]-0.1, "WIND DIRECTION", color='lightblue', fontsize=14, weight='bold', ha='center')

    # 3. Plot Edges
    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            # Determine Euclidean distance for comparison
            dist = np.linalg.norm(loc[j] - loc[i])
            cost = matrix[i, j]
            
            # Color Logic: Green if Easy (Tailwind), Red if Hard (Headwind)
            if cost < dist:
                color = 'green'
                style = 'solid'
                weight = 2
            else:
                color = 'red'
                style = 'dashed'
                weight = 1
                
            # Draw Curved Arrow (Bezier) to separate i->j from j->i
            # rad=0.2 gives a nice curve
            arrow = patches.FancyArrowPatch(
                (loc[i, 0], loc[i, 1]), 
                (loc[j, 0], loc[j, 1]),
                connectionstyle="arc3,rad=0.15", 
                arrowstyle="->",
                color=color,
                linewidth=weight,
                mutation_scale=20,
                zorder=5
            )
            ax.add_patch(arrow)
            
            # Label the cost on the curve
            # Midpoint calculation for curve
            mid = (loc[i] + loc[j]) / 2
            # Offset mid slightly perpendicular to line to match curve
            perp = np.array([-(loc[j,1]-loc[i,1]), loc[j,0]-loc[i,0]])
            perp = perp / np.linalg.norm(perp)
            # 0.15 is the rad, so we shift text roughly that amount
            text_pos = mid + perp * 0.08
            
            label = f"{cost:.2f}"
            ax.text(text_pos[0], text_pos[1], label, color=color, fontsize=9, ha='center', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.set_title("Visualizing Asymmetry in Windy TSP\nGreen=Tailwind (Cheap), Red=Headwind (Expensive)", fontsize=14)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    output_file = "windy_tsp_visualization.png"
    plt.savefig(output_file)
    print(f"\n[Graphic] Plot saved to {output_file}")
    print("Check the image to see the Green vs Red arrows!")

def run_visual_test():
    # 1. Create Data (Same as before)
    filename = "debug_windy_viz.pkl"
    mock_instance = {
        'loc': np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 
        'wind': np.array([1.0, 0.0]), 
        'alpha': 1.0
    }
    with open(filename, 'wb') as f:
        pickle.dump([mock_instance], f)

    # 2. Load
    dataset = TSPDataset(filename=filename, node_feature_type='hybrid', batch_size=1, num_samples=1)
    
    # 3. Calculate Matrix
    loc, wind, matrix = get_full_cost_matrix(dataset, 3)
    
    # 4. Print Matrix for sanity
    print("\nCost Matrix (Row=From, Col=To):")
    print(np.array_str(matrix, precision=2, suppress_small=True))
    
    # 5. Plot
    plot_windy_graph(loc, wind, matrix)
    
    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)

if __name__ == "__main__":
    run_visual_test()