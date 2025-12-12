import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Add root directory to path so we can import from 'utils' and 'problems'
sys.path.append(os.getcwd())

from utils import load_model, move_to
from problems.tsp.problem_tsp import nearest_neighbor_graph
from data.windy_tsp.generate_windy_tsp import generate_windy_instance

def get_edge_color(loc_from, loc_to, wind, alpha):
    """
    Determines edge color based on wind advantage.
    Green = Tailwind (Cost < Dist)
    Red = Headwind (Cost > Dist)
    Blue = Neutral
    """
    diff = loc_to - loc_from
    dist = np.linalg.norm(diff)
    if dist < 1e-6: return 'black', 0
    
    u = diff / dist
    proj = np.dot(u, wind)
    cost = dist * np.exp(-1.0 * alpha * proj)
    
    ratio = cost / dist
    if ratio < 0.95: return 'green', cost # Tailwind
    if ratio > 1.05: return 'red', cost   # Headwind
    return 'blue', cost                   # Neutral

def make_gif(model_path, save_name="agent_tour.gif"):
    print(f"Loading model from {model_path}...")
    model, _ = load_model(model_path)
    model.eval()
    
    # 1. Generate a Scenario
    print("Generating a random Windy TSP-20 instance...")
    N = 20
    data = generate_windy_instance(N, alpha=2.0, max_wind=1.0) # Strong wind for visibility
    loc = data['loc']
    wind = data['wind']
    alpha = data['alpha']
    
    # 2. Prepare Data for Model (Super-Packing to 7D)
    # We construct the 7D tensor manually to ensure compatibility with Hybrid models
    loc_T = torch.FloatTensor(loc)
    wind_T = torch.FloatTensor(wind).repeat(N, 1)
    alpha_T = torch.FloatTensor([alpha]).repeat(N, 1)
    
    # Calculate stats features
    diff = loc[None, :, :] - loc[:, None, :]
    dists = np.linalg.norm(diff, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = diff / dists[:, :, None]
    u[np.isnan(u)] = 0
    proj = np.dot(u, wind)
    costs = dists * np.exp(-1.0 * alpha * proj)
    np.fill_diagonal(costs, 0)
    stat_out = np.sum(costs, axis=1, keepdims=True) / (N - 1)
    stat_in = np.sum(costs, axis=0, keepdims=True).T / (N - 1)
    
    stat_out_T = torch.FloatTensor(stat_out)
    stat_in_T = torch.FloatTensor(stat_in)
    
    # [Batch, N, 7]
    input_data = torch.cat([loc_T, wind_T, alpha_T, stat_out_T, stat_in_T], dim=-1).unsqueeze(0)
    
    # Graph for GNN (using dummy KNN)
    graph = torch.ByteTensor(nearest_neighbor_graph(loc, 20, None)).unsqueeze(0)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = input_data.to(device)
    graph = graph.to(device)
    
    # 3. Run Model
    print("Running model inference...")
    with torch.no_grad():
        # We need the full tour. 'return_pi=True' gives us the indices
        # We assume Greedy decoding for visualization
        model.set_decode_type("greedy")
        cost, _, pi = model(input_data, graph, return_pi=True)
        tour = pi.cpu().numpy()[0]
    
    # 4. Create Animation
    print("Creating GIF...")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title("Agent Route Construction")
    
    # Draw Wind Arrow
    center = np.mean(loc, axis=0)
    ax.arrow(center[0]-0.1, center[1], wind[0]*0.2, wind[1]*0.2, 
             head_width=0.05, color='cyan', alpha=0.3, width=0.015)
    ax.text(center[0], center[1]-0.15, "WIND", color='cyan', ha='center', alpha=0.5, weight='bold')

    # Draw Nodes
    ax.scatter(loc[:, 0], loc[:, 1], c='black', zorder=10)
    # Start Node
    ax.scatter(loc[tour[0], 0], loc[tour[0], 1], c='gold', s=150, marker='*', zorder=11, label='Start')
    
    lines = []
    
    def update(frame):
        if frame == 0: return []
        
        # Current edge
        idx_from = tour[frame-1]
        idx_to = tour[frame % N] # Wrap for last edge
        
        p1 = loc[idx_from]
        p2 = loc[idx_to]
        
        color, _ = get_edge_color(p1, p2, wind, alpha)
        
        # Plot line
        line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2, alpha=0.8)
        lines.append(line)
        return lines

    ani = FuncAnimation(fig, update, frames=N+1, interval=500, blit=False)
    
    ani.save(save_name, writer=PillowWriter(fps=2))
    print(f"Saved visualization to {save_name}")

if __name__ == "__main__":
    
    # --- CONFIGURATION VARIABLES ---
    # Set your model path and output filename here
    
    MODEL_PATH = "/home/pfc/gms/code/rethinking_tsp/outputs/exp1_coords_20251210T133907/epoch-49.pt" 
    # Example: "outputs/windy_tsp_20-20/exp1_hybrid_20251210T130509/epoch-0.pt"
    
    OUTPUT_FILE = "agent_tour.gif"
    
    # --------------------------------
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please check the path or wait for training to complete an epoch.")
    else:
        make_gif(MODEL_PATH, OUTPUT_FILE)