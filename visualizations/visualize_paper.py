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

def extract_exact_wind(coords, cost_matrix):
    """
    Extracts the exact wind vector (w * alpha) using least squares,
    based on the relationship: c_ij = d_ij * exp(-alpha * (w \cdot u_ij))
    """
    n = len(coords)
    A = []
    b = []
    for i in range(n):
        for j in range(i + 1, n): # Only need upper triangle
            d_ij = np.linalg.norm(coords[j] - coords[i])
            if d_ij > 1e-9:
                delta_p = coords[j] - coords[i]
                c_ij = cost_matrix[i, j]
                c_ji = cost_matrix[j, i]
                
                # (alpha * w) \cdot (p_j - p_i) = 0.5 * d_ij * ln(c_ji / c_ij)
                val = 0.5 * d_ij * np.log((c_ji + 1e-12) / (c_ij + 1e-12))
                
                A.append(delta_p)
                b.append(val)
                
    A = np.array(A)
    b = np.array(b)
    
    # Solve least squares Aw = b for the scaled wind vector
    wind_vector, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    wind_mag = np.linalg.norm(wind_vector)
    if wind_mag > 1e-9:
        wind_dir = wind_vector / wind_mag # Normalized for plotting
    else:
        wind_dir = np.zeros(2)
        
    return wind_dir, wind_mag

def plot_tour_heatmap(ax, coords, tour, cost_matrix, title, cost, wind_vector, wind_mag):
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
        
        BOUND = 0.6 
        
        norm_val = (log_ratio + BOUND) / (2.0 * BOUND)
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
    ax.text(center[0], center[1], f"WIND\nMag: {wind_mag:.4f}", color='blue', alpha=0.6, 
            fontsize=10, fontweight='bold', ha='center', va='center')
    

    ax.set_title(f"{title}\nTotal Cost: {cost:.4f}", fontsize=13, fontweight='bold')
    ax.axis('off')

def compare_models_to_lkh(orig_model_path, imp_model_path, dataset_path, lkh_path, seed, index):
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Inference Helper ---
    # MODIFIED: Un-nested variables so we can pass specific nodes/graphs per model
    def get_tour(model, current_nodes, current_graph, WIDTH=10):
        with torch.no_grad():
            if WIDTH == 0 or WIDTH == 1:
                # Fallback to Greedy
                model.set_decode_type("greedy")
                _, _, sequences, _ = model(current_nodes, current_graph, return_pi=True, return_entropy=True)
                return sequences[0].cpu().numpy()
            else:
                # Use Beam Search
                cum_log_p, sequences, raw_costs, ids, batch_size = model.beam_search(
                    current_nodes, current_graph, beam_size=WIDTH, compress_mask=False, max_calc_batch_size=10000
                )
                sequences, _ = get_best(
                    sequences, raw_costs.cpu().numpy(),
                    ids.cpu().numpy() if ids is not None else None, batch_size
                )
                
                # Safely convert to numpy regardless of what get_best returns
                seq = sequences[0]
                if torch.is_tensor(seq):
                    return seq.cpu().numpy()
                return np.array(seq)

    # --- Load Original Model to setup problem and dataset ---
    print(f"[*] Loading Original Model from {orig_model_path}...")
    orig_model, orig_args = load_model(orig_model_path)
    orig_model.to(device)
    orig_model.eval()
    
    orig_dataset = orig_model.problem.make_dataset(
        filename=dataset_path, batch_size=1, num_samples=1, offset=target_idx, 
        neighbors=orig_args.get('neighbors', 20), 
        knn_strat=orig_args.get('knn_strat', 'None'), 
        node_feature_type=orig_args.get('node_feature_type', 'coords'),
        supervised=True
    )
    orig_batch = next(iter(torch.utils.data.DataLoader(orig_dataset, batch_size=1)))
    orig_nodes = move_to(orig_batch['nodes'], device)
    orig_graph = move_to(orig_batch['graph'], device)

    # Extract drawing coordinates and wind from the original (vanilla) dataset
    cost_matrix = orig_batch['cost_matrix'][0].cpu().numpy()
    coords = orig_nodes[0].cpu().numpy()[:, :2]

    # Calculate exact wind from cost matrix mathematically
    wind_vector, wind_mag = extract_exact_wind(coords, cost_matrix)
    print(f"[*] Extracted Wind Magnitude (alpha * |w|): {wind_mag:.4f}")
    
    # Generate Original Model Tour passing its specific nodes/graph
    orig_seq = get_tour(orig_model, orig_nodes, orig_graph)
    orig_name = os.path.basename(os.path.dirname(orig_model_path))

    # --- Load Improved Model and Generate Tour ---
    print(f"[*] Loading Improved Model from {imp_model_path}...")
    
    # FIX 1: Capture imp_args instead of ignoring them with `_`
    imp_model, imp_args = load_model(imp_model_path)
    imp_model.to(device)
    imp_model.eval()
    
    # FIX 1: Generate a fresh dataset specifically for the improved model
    # This guarantees it receives the hybrid features and sparsified graphs it expects
    imp_dataset = imp_model.problem.make_dataset(
        filename=dataset_path, batch_size=1, num_samples=1, offset=target_idx, 
        neighbors=imp_args.get('neighbors', 20), 
        knn_strat=imp_args.get('knn_strat', 'None'), 
        node_feature_type=imp_args.get('node_feature_type', 'coords'),
        supervised=True
    )
    imp_batch = next(iter(torch.utils.data.DataLoader(imp_dataset, batch_size=1)))
    imp_nodes = move_to(imp_batch['nodes'], device)
    imp_graph = move_to(imp_batch['graph'], device)

    # Generate Improved Model Tour passing its specific nodes/graph
    imp_seq = get_tour(imp_model, imp_nodes, imp_graph)
    imp_name = os.path.basename(os.path.dirname(imp_model_path))

    # --- Extract LKH Tour ---
    lkh_seq = np.array(lkh_data[target_idx][1]).astype(int)
    if lkh_seq.min() == 1:
        lkh_seq -= 1 

    # --- Calculate Costs and Gaps (USING MODEL ENVIRONMENT) ---
    
    # 1. Convert numpy sequences to PyTorch tensors
    orig_seq_tensor = torch.tensor(orig_seq, dtype=torch.long, device=device).unsqueeze(0)
    imp_seq_tensor = torch.tensor(imp_seq, dtype=torch.long, device=device).unsqueeze(0)

    # 2. Pass their RESPECTIVE nodes and sequences to the problem environment
    orig_true_costs, _ = orig_model.problem.get_costs(orig_nodes, orig_seq_tensor)
    imp_true_costs, _ = imp_model.problem.get_costs(imp_nodes, imp_seq_tensor)
    
    # 3. Extract the clean scalar float values
    matrix_orig_cost = orig_true_costs.item()
    matrix_imp_cost = imp_true_costs.item()
    
    # FIX 3: Pull exact LKH cost directly from the file to avoid environment calculation skew
    matrix_lkh_cost = lkh_data[target_idx][0]
    
    orig_opt_gap = ((matrix_orig_cost / matrix_lkh_cost) - 1.0) * 100
    imp_opt_gap = ((matrix_imp_cost / matrix_lkh_cost) - 1.0) * 100

    print("\n" + "="*50)
    print(f"GROUND TRUTH VERIFICATION REPORT (Instance {target_idx})")
    print("="*50)
    print(f"Original Model Cost:   {matrix_orig_cost:.4f} (Gap: {orig_opt_gap:.2f}%)")
    print(f"Improved Model Cost:   {matrix_imp_cost:.4f} (Gap: {imp_opt_gap:.2f}%)")
    print(f"LKH Cost (Exact):      {matrix_lkh_cost:.4f}")
    print("="*50 + "\n")

    # --- Plotting 1x3 Grid ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7)) 
    
    plot_tour_heatmap(ax1, coords, orig_seq, cost_matrix, f"Original Model", matrix_orig_cost, wind_vector, wind_mag)
    plot_tour_heatmap(ax2, coords, imp_seq, cost_matrix, f"Improved Model", matrix_imp_cost, wind_vector, wind_mag)
    plot_tour_heatmap(ax3, coords, lkh_seq, cost_matrix, "LKH Baseline (Optimal)", matrix_lkh_cost, wind_vector, wind_mag)
    
    plt.suptitle(f"Windy TSP Route Comparison (Instance {target_idx})\nGreen = Tailwind | Red = Headwind", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    out_img = f"visualizations/solution_visualization/PAPER_gap_comparison_idx{target_idx}_seed{seed}.png"
    plt.tight_layout()
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"[*] Visual comparison saved to '{out_img}'")

if __name__ == "__main__":
    ORIG_MODEL = "outputs/002_exp1_confidence_og_stats/05ConfOgStats_ent0.2_exp1/exp1_coords_ent0.2_20251223T185323/epoch-99.pt"
    IMP_MODEL = "outputs/windy_tsp_20-20/008_exp3_graph_sparsification/13Random_Sparsification/tsp20_dual_hybrid_ent0.05_random20_20260302T160327/epoch-99.pt"
    DATASET = "data/windy_tsp/windy_tsp20_val.pkl"
    LKH_BASELINE = "results/windy_tsp20_val/windy_tsp20_valn1280-lkh_windy.pkl"
    INDEX = None
    SEED = 1513

    compare_models_to_lkh(ORIG_MODEL, IMP_MODEL, DATASET, LKH_BASELINE, SEED, INDEX)

    