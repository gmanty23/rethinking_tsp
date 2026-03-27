#!/usr/bin/env python

import os
import json
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import subprocess
import uuid
import argparse

# --- Project Imports ---
# Adjust these imports if your exact file structure differs slightly
from nets.attention_model import AttentionModel
from nets.encoders.gnn_encoder import GNNEncoder
from nets.encoders.gat_encoder import GraphAttentionEncoder
from nets.encoders.mlp_encoder import MLPEncoder
from problems.tsp.problem_tsp import TSP # Or WindyTSP, depending on your class name
from utils import torch_load_cpu

def set_seed(seed):
    """
    Ensures deterministic generation across all libraries for rigorous ablation.
    This guarantees that the exact same graph (nodes + wind vector) is generated
    for different models, allowing for an apples-to-apples comparison.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_windy_model(folder_path):
    """
    Parses args.json and loads the model weights from epoch-99.pt.
    Dynamically reconstructs the architecture based on the saved hyperparameters.
    """
    args_path = os.path.join(folder_path, 'args.json')
    weights_path = os.path.join(folder_path, 'epoch-99.pt')
    
    with open(args_path, 'r') as f:
        args = json.load(f)
        
    print(f"[*] Loaded configuration for model: {args.get('run_name', 'Unknown')}")
    
    # Initialize the problem environment
    # Note: Replace 'TSP()' with 'WindyTSP()' if you separated the classes
    problem = TSP() 
    
    # Map string arguments to actual Encoder classes
    encoder_classes = {'gnn': GNNEncoder, 'gat': GraphAttentionEncoder, 'mlp': MLPEncoder}
    encoder_class = encoder_classes.get(args.get('encoder', 'gnn'))
    
    # Reconstruct the model architecture
    model = AttentionModel(
        problem=problem,
        embedding_dim=args['embedding_dim'],
        encoder_class=encoder_class,
        n_encode_layers=args['n_encode_layers'],
        aggregation=args.get('aggregation', 'max'),
        normalization=args.get('normalization', 'layer'),
        
        # --- ADD THESE NEW LINES ---
        learn_norm=args.get('learn_norm', True),
        track_norm=args.get('track_norm', True),
        # ---------------------------
        
        node_feature_type=args.get('node_feature_type', 'coords'),
        gnn_direction_mode=args.get('gnn_direction_mode', 'forward')
    )
    
    # Load weights, handling potential DistributedDataParallel wrapping ('module.')
    load_data = torch_load_cpu(weights_path)
    state_dict = load_data.get('model', load_data)
    unwrapped_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(unwrapped_state_dict)
    
    # Set to evaluation mode and deterministic greedy decoding
    model.eval()
    model.set_decode_type("greedy")
    return model, args

def solve_lkh(cost_matrix):
    """
    Wrapper to run the LKH-3 heuristic solver on the asymmetric cost matrix.
    LKH requires integer weights, so we scale the float matrix up.
    """
    LKH_PATH = "./LKH" # Ensure the compiled LKH executable is in this path
    if not os.path.exists(LKH_PATH):
        print("[!] LKH executable not found at './LKH'. Returning a sequential dummy tour.")
        return np.arange(len(cost_matrix))
        
    num_nodes = len(cost_matrix)
    uid = str(uuid.uuid4())[:8] # Unique ID to prevent file collisions if run in parallel
    filename = f"t_{uid}.atsp"
    par_filename = f"t_{uid}.par"
    tour_filename = f"t_{uid}.tour"
    
    # Scale matrix to integers to preserve precision for LKH
    int_matrix = (cost_matrix * 100000).astype(int)
    
    # Write ATSP (Asymmetric TSP) file
    with open(filename, 'w') as f:
        f.write(f"NAME: temp\nTYPE: ATSP\nDIMENSION: {num_nodes}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
        for row in int_matrix:
            f.write(" ".join(map(str, row)) + "\n")
        f.write("EOF\n")
        
    # Write LKH Parameter file
    with open(par_filename, 'w') as f:
        f.write(f"PROBLEM_FILE = {filename}\nTOUR_FILE = {tour_filename}\nRUNS = 1\nTRACE_LEVEL = 0\n")
        
    tour = []
    try:
        # Execute LKH silently
        subprocess.run([LKH_PATH, par_filename], check=True, stdout=subprocess.DEVNULL)
        
        # Parse the resulting tour file
        if os.path.exists(tour_filename):
            with open(tour_filename, 'r') as f:
                lines = f.readlines()
                reading = False
                for line in lines:
                    if "TOUR_SECTION" in line: 
                        reading = True
                        continue
                    if "EOF" in line or "-1" in line: 
                        break
                    if reading: 
                        # LKH uses 1-based indexing; we subtract 1 for 0-based Python indexing
                        tour.append(int(line.strip()) - 1)
    finally:
        # Clean up temporary files
        for f in [filename, par_filename, tour_filename]:
            if os.path.exists(f): 
                os.remove(f)
            
    return np.array(tour) if tour else np.arange(num_nodes)

from matplotlib.colors import Normalize

from matplotlib.colors import Normalize

def draw_tour(ax, loc, tour, cost_matrix, wind_vector, title, cost, alpha_wind=3.0):
    """
    Draws the nodes, global wind vector (centered, bright arrow), and heatmap-colored edges.
    Green edges = Tailwind. Red edges = Headwind.
    """
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Scatter plot for cities/nodes (on top)
    ax.scatter(loc[:, 0], loc[:, 1], c='black', s=50, zorder=5)
    
    # Normalize wind vector for consistent plotting length
    wind_norm = np.linalg.norm(wind_vector)
    w_hat = wind_vector / (wind_norm + 1e-9)
    
    # --- CENTERED WIND REPRESENTATION ---
    # Decreased scale makes the arrow longer. Increased alpha and vibrant color make it brighter.
    # pivot='mid' ensures the center of the arrow is exactly at (0.5, 0.5).
    ax.quiver(0.5, 0.5, w_hat[0], w_hat[1], 
              color='deepskyblue', scale=1.5, width=0.035, 
              transform=ax.transAxes, zorder=1, alpha=0.45, pivot='mid')
    
    # --- BRIGHTER RED-TO-GREEN COLORMAP ---
    norm = Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap('RdYlGn') # Red (-1) to Green (+1)
    
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        
        # Calculate geometric direction vector of the chosen edge
        dir_vec = loc[v] - loc[u]
        dist = np.linalg.norm(dir_vec)
        u_ij = dir_vec / (dist + 1e-9)
        
        # Calculate alignment with wind (Range: -1.0 to 1.0)
        alignment = np.dot(u_ij, w_hat)
        
        # Map alignment directly to vibrant color
        color = cmap(norm(alignment))
        
        # Draw directed edge
        ax.annotate("", xy=loc[v], xytext=loc[u],
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.5, 
                                    shrinkA=5, shrinkB=5, connectionstyle="arc3,rad=0.1"),
                    zorder=4)
    
    ax.set_xlabel(f"Tour Length: {cost:.4f}", fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('equal')

def main(model_folder, seed):
    # 1. Enforce strict reproducibility
    set_seed(seed)
    print(f"[*] Random seed set to: {seed}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Model and Arguments
    model, args = load_windy_model(model_folder)
    model.to(device)
    

    # 3. Generate a Test Graph
    # We use the problem's dataset generator to create exactly one graph
    dataset = model.problem.make_dataset(
        num_samples=1, 
        batch_size=1, # <--- ADD THIS LINE
        min_size=args.get('min_size', 20), 
        max_size=args.get('max_size', 20), 
        neighbors=args.get('neighbors', 0.2), 
        knn_strat=args.get('knn_strat', 'percentage'), 
        node_feature_type=args.get('node_feature_type', 'coords')
    )
    
    batch = dataset[0]
    
    # Ensure correct tensor dimensions for a batch size of 1
    nodes = batch['nodes'].unsqueeze(0).to(device)
    graph = batch.get('graph', torch.empty(0)).unsqueeze(0).to(device) if 'graph' in batch else None
    cost_matrix = batch['cost_matrix'].numpy()
    
    # Extract coordinates and the global wind vector for plotting
    loc = nodes[0, :, :2].cpu().numpy()
    wind_vector = nodes[0, 0, 2:4].cpu().numpy() # Assumes wind is concatenated at indices 2:4
    alpha_wind = args.get('wind_alpha', 3.0) # Used for color scaling

    # 4. Perform Inference
    print("[*] Running Neural Model Inference...")
    with torch.no_grad():
        # Depending on your exact forward pass, you might not need 'graph'
        if graph is not None and len(graph.shape) > 2:
            pred_cost, _, pi = model(nodes, graph, return_pi=True)
        else:
            pred_cost, _, pi = model(nodes, return_pi=True)
            
    pred_tour = pi[0].cpu().numpy()
    model_cost = pred_cost[0].item()

    # 5. Get Optimal LKH Baseline Solution
    print("[*] Running LKH-3 Baseline...")
    lkh_tour = solve_lkh(cost_matrix)
    
    # Calculate exact LKH Cost using the asymmetric matrix
    lkh_cost = 0.0
    for i in range(len(lkh_tour)):
        u = lkh_tour[i]
        v = lkh_tour[(i + 1) % len(lkh_tour)]
        lkh_cost += cost_matrix[u, v]

    # 6. Plotting
    print("[*] Generating Visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Fetch run name or fallback to folder name
    model_name = args.get('run_name', os.path.basename(os.path.normpath(model_folder)))
    
    draw_tour(ax1, loc, pred_tour, cost_matrix, wind_vector, f"Model: {model_name}", model_cost, alpha_wind)
    draw_tour(ax2, loc, lkh_tour, cost_matrix, wind_vector, "Optimal: LKH-3", lkh_cost, alpha_wind)
    
    # Add a global title indicating the seed
    plt.suptitle(f"Windy TSP Inference - Seed {seed}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Save in an image
    out_img = f"visualizations/solution_visualization/windy_tsp_comparison_seed{seed}.png"
    fig.savefig(out_img, dpi=300)
    print(f"[*] Visualization saved to '{out_img}'")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Visualize Windy TSP Inference with Reproducibility")
    # parser.add_argument("folder", help="Path to the model directory (must contain args.json and epoch-99.pt)")
    # parser.add_argument("--seed", type=int, default=1234, help="Random seed for deterministic graph generation")
    # parsed = parser.parse_args()
    
    # main(parsed.folder, parsed.seed)
    # FOLDER = "outputs/windy_tsp_20-20/003_exp1_new_stats/06ConfNewStats_ent0.1_exp1/stats_learned_ent0.1_20251227T132122"
    FOLDER = "outputs/knn_neighbors/tsp20_backward_hybrid_ent0.05_percentage_n0.15_20260304T153537"
    SEED = 1234
    main(FOLDER, SEED)