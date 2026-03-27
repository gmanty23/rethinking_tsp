import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pickle
from matplotlib.colors import Normalize
from utils.functions import load_model, move_to, get_best

def get_matrix_cost(cost_matrix, tour):
    """Calculates the exact cost using vectorized array indexing."""
    return cost_matrix[tour, np.roll(tour, -1)].sum()

def estimate_prevailing_wind(coords, cost_matrix):
    """
    Approximates the global wind direction vector based on the 
    skew-symmetric component of the cost matrix.
    """
    W = np.zeros(2)
    n = len(coords)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = coords[j] - coords[i]
                dist = np.linalg.norm(d)
                if dist > 1e-9:
                    # Positive asymmetry means i->j is easier than j->i, 
                    # so wind flows in the direction of d.
                    asymmetry = cost_matrix[j, i] - cost_matrix[i, j]
                    W += asymmetry * (d / dist)
    
    wind_norm = np.linalg.norm(W)
    if wind_norm > 1e-9:
        W = W / wind_norm # Normalize to unit vector for plotting
    return W

def plot_tour_heatmap(ax, coords, tour, cost_matrix, title, cost, wind_vector):
    """Draws the tour with saturated green/red edges and a central wind arrow."""
    ax.scatter(coords[:, 0], coords[:, 1], c='black', s=40, zorder=5)
    
    cmap = plt.get_cmap('RdYlGn_r') 
    
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        
        edge_cost = cost_matrix[u, v]
        geo_dist = np.linalg.norm(coords[v] - coords[u])
        
        ratio = edge_cost / (geo_dist + 1e-9)
        log_ratio = np.log(ratio + 1e-9) 
        
        # Tighter bounds: log_ratio in [-1.0, 1.0] forces brighter, more saturated colors
        norm_val = (log_ratio + 1.0) / 2.0 
        norm_val = max(0.0, min(1.0, norm_val)) 
        
        color = cmap(norm_val)
        
        ax.annotate("", xy=coords[v], xytext=coords[u],
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.5, 
                                    shrinkA=5, shrinkB=5, connectionstyle="arc3,rad=0.05"),
                    zorder=3)
        
    # Plot global wind vector in the center
    center = coords.mean(axis=0)
    # Scale arrow size relative to coordinate bounds
    scale = (coords.max() - coords.min()) * 0.15 
    ax.annotate("", xy=center + wind_vector * scale, xytext=center - wind_vector * scale,
                arrowprops=dict(arrowstyle="->", color='blue', lw=4, alpha=0.4),
                zorder=2)
    
    # Add wind label
    ax.text(center[0], center[1], "WIND", color='blue', alpha=0.6, 
            fontsize=10, fontweight='bold', ha='center', va='center')

    ax.set_title(f"{title}\nTotal Cost: {cost:.4f}", fontsize=13, fontweight='bold')
    ax.axis('off')

def compare_model_to_lkh(model_path, dataset_path, lkh_path, seed, index):
    print(f"[*] Loading LKH Baseline from {lkh_path}...")
    
    with open(lkh_path, 'rb') as f:
        lkh_data = pickle.load(f)
        
    # Convert iterator to list to allow len() and indexing
    if not isinstance(lkh_data, list):
        lkh_data = list(lkh_data)
        
    total_instances = len(lkh_data)
    
    if seed is not None:
        np.random.seed(seed)
        target_idx = np.random.randint(0, total_instances)
    else:
        target_idx = index
        
    print(f"[*] Evaluating Instance Index: {target_idx}")

    print(f"[*] Loading model from {model_path}...")
    model, model_args = load_model(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    dataset = model.problem.make_dataset(
        filename=dataset_path, batch_size=1, num_samples=1, offset=target_idx, 
        neighbors=model_args.get('neighbors', 20), 
        knn_strat=model_args.get('knn_strat', 'None'), 
        node_feature_type=model_args.get('node_feature_type', 'coords'),
        supervised=True
    )
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
    nodes = move_to(batch['nodes'], device)
    graph = move_to(batch['graph'], device)

    current_wind = batch['nodes'][0, 0, 2:4].cpu().numpy() # Assuming features contain wind
    mag = np.linalg.norm(current_wind)
    print(f"Current Instance Wind Magnitude: {mag:.4f}")
    
    cost_matrix = batch['cost_matrix'][0].cpu().numpy()
    
    WIDTH = 100
    with torch.no_grad():
        if WIDTH == 0 or WIDTH == 1:
            # Fallback to Greedy
            model.set_decode_type("greedy")
            _, _, sequences, _ = model(nodes, graph, return_pi=True, return_entropy=True)
            # Here sequences is still a PyTorch tensor, so we need .cpu().numpy()
            model_seq = sequences[0].cpu().numpy()
        else:
            # Use Beam Search
            cum_log_p, sequences, raw_costs, ids, batch_size = model.beam_search(
                nodes, graph, beam_size=WIDTH, compress_mask=False, max_calc_batch_size=10000
            )
            sequences, _ = get_best(
                sequences, raw_costs.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None, batch_size
            )
            # get_best ALREADY returns a numpy array, so we just take the first one!
            model_seq = sequences[0]
        
    model_seq = sequences[0].cpu().numpy()
    coords = nodes[0].cpu().numpy()[:, :2]
    
    # Extract LKH Tour and adjust for 1-based indexing if necessary
    lkh_seq = np.array(lkh_data[target_idx][1]).astype(int)
    if lkh_seq.min() == 1:
        lkh_seq -= 1 

    matrix_model_cost = get_matrix_cost(cost_matrix, model_seq)
    matrix_lkh_cost = get_matrix_cost(cost_matrix, lkh_seq)
    opt_gap = ((matrix_model_cost / matrix_lkh_cost) - 1.0) * 100

    print("\n" + "="*40)
    print(f"GROUND TRUTH VERIFICATION REPORT (Instance {target_idx})")
    print("="*40)
    print(f"Model Cost (Matrix):    {matrix_model_cost:.4f}")
    print(f"LKH Cost (Matrix):      {matrix_lkh_cost:.4f}")
    print(f"True Optimality Gap:    {opt_gap:.2f}%")
    print("="*40 + "\n")

    # Estimate wind vector mathematically
    wind_vector = estimate_prevailing_wind(coords, cost_matrix)

    # Extract model name from the path for the title
    model_name = os.path.basename(os.path.dirname(model_path))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    plot_tour_heatmap(ax1, coords, model_seq, cost_matrix, f"Model: {model_name}", matrix_model_cost, wind_vector)
    plot_tour_heatmap(ax2, coords, lkh_seq, cost_matrix, "LKH Baseline (Optimal)", matrix_lkh_cost, wind_vector)
    
    plt.suptitle(f"Windy TSP Route Comparison (Instance {target_idx} | True Gap: {opt_gap:.2f}%)\nGreen = Tailwind | Red = Headwind", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    out_img = f"gap_comparison_idx{target_idx}_OJO.png"
    plt.tight_layout()
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"[*] Visual comparison saved to '{out_img}'")

if __name__ == "__main__":
    MODEL = "outputs/knn_neighbors_4/tsp20_forward_hybrid_ent0.05_percentage_n0.2_20260302T175516/epoch-99.pt"
    DATASET = "data/windy_tsp/windy_tsp20_val.pkl"
    LKH_BASELINE = "results/windy_tsp20_val/windy_tsp20_valn1280-lkh_windy.pkl"
    INDEX = None
    SEED = 12

    compare_model_to_lkh(MODEL, DATASET, LKH_BASELINE, SEED, INDEX)