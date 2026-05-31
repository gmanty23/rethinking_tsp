import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pickle
from matplotlib.colors import Normalize
from utils.functions import load_model, move_to, get_best

# --- ACADEMIC PLOTTING CONFIGURATION ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'figure.titlesize': 12,
    'text.usetex': False, # Set to True if you have a full LaTeX engine installed
    'figure.dpi': 300,
    'pdf.fonttype': 42,   # Ensures fonts are embedded in the PDF
    'ps.fonttype': 42
})

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
        for j in range(i + 1, n): 
            d_ij = np.linalg.norm(coords[j] - coords[i])
            if d_ij > 1e-9:
                delta_p = coords[j] - coords[i]
                c_ij = cost_matrix[i, j]
                c_ji = cost_matrix[j, i]
                
                val = 0.5 * d_ij * np.log((c_ji + 1e-12) / (c_ij + 1e-12))
                
                A.append(delta_p)
                b.append(val)
                
    A = np.array(A)
    b = np.array(b)
    
    wind_vector, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    wind_mag = np.linalg.norm(wind_vector)
    if wind_mag > 1e-9:
        wind_dir = wind_vector / wind_mag 
    else:
        wind_dir = np.zeros(2)
        
    return wind_dir, wind_mag

def plot_tour_heatmap(ax, coords, tour, cost_matrix, title, cost, gap, wind_vector, wind_mag):
    """Draws the tour with colorblind-safe edges, central wind arrow, and metrics."""
    # Lighter scatter points to let arrows stand out
    ax.scatter(coords[:, 0], coords[:, 1], c='dimgrey', s=30, zorder=5)
    
    # Use RdBu_r: Blue for tailwind (efficient), Red for headwind (costly)
    cmap = plt.get_cmap('RdBu_r') 
    
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
        
        # Thinner arrows for a cleaner academic look
        ax.annotate("", xy=coords[v], xytext=coords[u],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, 
                                    shrinkA=4, shrinkB=4, connectionstyle="arc3,rad=0.08"),
                    zorder=3)
        
    center = coords.mean(axis=0)
    scale = (coords.max() - coords.min()) * 0.15 
    
    # Dark grey for wind arrow to avoid clashing with the heatmap
    ax.annotate("", xy=center + wind_vector * scale, xytext=center - wind_vector * scale,
                arrowprops=dict(arrowstyle="->", color='black', lw=2.5, alpha=0.5),
                zorder=2)
    
    ax.text(center[0], center[1] - (scale * 0.5), f"WIND\nMag: {wind_mag:.2f}", 
            color='black', alpha=0.7, fontsize=9, ha='center', va='top')
    
    # Clean, non-bold titles for academic formatting
    ax.set_title(f"{title}\nCost: {cost:.2f} | Gap: {gap:.2f}%")
    ax.axis('off')

def compare_models_to_lkh(coords_model_path, stats_model_path, hybrid_model_path, dataset_path, lkh_path, seed, index):
    print(f"[*] Loading LKH Baseline from {lkh_path}...")
    
    with open(lkh_path, 'rb') as f:
        lkh_data = pickle.load(f)
        
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

    def get_tour(model, current_nodes, current_graph, WIDTH=10):
        with torch.no_grad():
            if WIDTH == 0 or WIDTH == 1:
                model.set_decode_type("greedy")
                _, _, sequences, _ = model(current_nodes, current_graph, return_pi=True, return_entropy=True)
                return sequences[0].cpu().numpy()
            else:
                cum_log_p, sequences, raw_costs, ids, batch_size = model.beam_search(
                    current_nodes, current_graph, beam_size=WIDTH, compress_mask=False, max_calc_batch_size=10000
                )
                sequences, _ = get_best(
                    sequences, raw_costs.cpu().numpy(),
                    ids.cpu().numpy() if ids is not None else None, batch_size
                )
                
                seq = sequences[0]
                if torch.is_tensor(seq):
                    return seq.cpu().numpy()
                return np.array(seq)

    # --- Load Coords Model (Original) ---
    print(f"[*] Loading Coords Model from {coords_model_path}...")
    coords_model, coords_args = load_model(coords_model_path)
    coords_model.to(device)
    coords_model.eval()
    
    coords_dataset = coords_model.problem.make_dataset(
        filename=dataset_path, batch_size=1, num_samples=1, offset=target_idx, 
        neighbors=coords_args.get('neighbors', 20), 
        knn_strat=coords_args.get('knn_strat', 'None'), 
        node_feature_type=coords_args.get('node_feature_type', 'coords'),
        supervised=True
    )
    coords_batch = next(iter(torch.utils.data.DataLoader(coords_dataset, batch_size=1)))
    coords_nodes = move_to(coords_batch['nodes'], device)
    coords_graph = move_to(coords_batch['graph'], device)

    cost_matrix = coords_batch['cost_matrix'][0].cpu().numpy()
    coords = coords_nodes[0].cpu().numpy()[:, :2]

    wind_vector, wind_mag = extract_exact_wind(coords, cost_matrix)
    print(f"[*] Extracted Wind Magnitude (alpha * |w|): {wind_mag:.4f}")
    
    coords_seq = get_tour(coords_model, coords_nodes, coords_graph)

    # --- Load Stats Model (New) ---
    print(f"[*] Loading Stats Model from {stats_model_path}...")
    stats_model, stats_args = load_model(stats_model_path)
    stats_model.to(device)
    stats_model.eval()
    
    stats_dataset = stats_model.problem.make_dataset(
        filename=dataset_path, batch_size=1, num_samples=1, offset=target_idx, 
        neighbors=stats_args.get('neighbors', 20), 
        knn_strat=stats_args.get('knn_strat', 'None'), 
        node_feature_type=stats_args.get('node_feature_type', 'coords'),
        supervised=True
    )
    stats_batch = next(iter(torch.utils.data.DataLoader(stats_dataset, batch_size=1)))
    stats_nodes = move_to(stats_batch['nodes'], device)
    stats_graph = move_to(stats_batch['graph'], device)

    stats_seq = get_tour(stats_model, stats_nodes, stats_graph)

    # --- Load Hybrid Model (Best) ---
    print(f"[*] Loading Hybrid Model from {hybrid_model_path}...")
    hybrid_model, hybrid_args = load_model(hybrid_model_path)
    hybrid_model.to(device)
    hybrid_model.eval()
    
    hybrid_dataset = hybrid_model.problem.make_dataset(
        filename=dataset_path, batch_size=1, num_samples=1, offset=target_idx, 
        neighbors=hybrid_args.get('neighbors', 20), 
        knn_strat=hybrid_args.get('knn_strat', 'None'), 
        node_feature_type=hybrid_args.get('node_feature_type', 'coords'),
        supervised=True
    )
    hybrid_batch = next(iter(torch.utils.data.DataLoader(hybrid_dataset, batch_size=1)))
    hybrid_nodes = move_to(hybrid_batch['nodes'], device)
    hybrid_graph = move_to(hybrid_batch['graph'], device)

    hybrid_seq = get_tour(hybrid_model, hybrid_nodes, hybrid_graph)

    # --- Extract LKH Tour ---
    lkh_seq = np.array(lkh_data[target_idx][1]).astype(int)
    if lkh_seq.min() == 1:
        lkh_seq -= 1 

    # --- Calculate Costs and Gaps ---
    coords_seq_tensor = torch.tensor(coords_seq, dtype=torch.long, device=device).unsqueeze(0)
    stats_seq_tensor = torch.tensor(stats_seq, dtype=torch.long, device=device).unsqueeze(0)
    hybrid_seq_tensor = torch.tensor(hybrid_seq, dtype=torch.long, device=device).unsqueeze(0)

    coords_true_costs, _ = coords_model.problem.get_costs(coords_nodes, coords_seq_tensor)
    stats_true_costs, _ = stats_model.problem.get_costs(stats_nodes, stats_seq_tensor)
    hybrid_true_costs, _ = hybrid_model.problem.get_costs(hybrid_nodes, hybrid_seq_tensor)
    
    matrix_coords_cost = coords_true_costs.item()
    matrix_stats_cost = stats_true_costs.item()
    matrix_hybrid_cost = hybrid_true_costs.item()
    matrix_lkh_cost = lkh_data[target_idx][0]
    
    coords_opt_gap = ((matrix_coords_cost / matrix_lkh_cost) - 1.0) * 100
    stats_opt_gap = ((matrix_stats_cost / matrix_lkh_cost) - 1.0) * 100
    hybrid_opt_gap = ((matrix_hybrid_cost / matrix_lkh_cost) - 1.0) * 100

    print("\n" + "="*50)
    print(f"GROUND TRUTH VERIFICATION REPORT (Instance {target_idx})")
    print("="*50)
    print(f"Coords Model Cost:   {matrix_coords_cost:.4f} (Gap: {coords_opt_gap:.2f}%)")
    print(f"Stats Model Cost:    {matrix_stats_cost:.4f} (Gap: {stats_opt_gap:.2f}%)")
    print(f"Hybrid Model Cost:   {matrix_hybrid_cost:.4f} (Gap: {hybrid_opt_gap:.2f}%)")
    print(f"LKH Cost (Exact):    {matrix_lkh_cost:.4f} (Gap: 0.00%)")
    print("="*50 + "\n")

    # --- Plotting 1x4 Grid ---
    # Constrained layout manages margins better than tight_layout
    # Size reduced from 28x7 to 15x3.8 to scale well into a paper page width
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True) 
    ax1, ax2, ax3, ax4 = axes
    
    plot_tour_heatmap(ax1, coords, coords_seq, cost_matrix, "Coords Only (Original)", matrix_coords_cost, coords_opt_gap, wind_vector, wind_mag)
    plot_tour_heatmap(ax2, coords, stats_seq, cost_matrix, "Stats Only", matrix_stats_cost, stats_opt_gap, wind_vector, wind_mag)
    plot_tour_heatmap(ax3, coords, hybrid_seq, cost_matrix, "Sparse Hybrid", matrix_hybrid_cost, hybrid_opt_gap, wind_vector, wind_mag)
    plot_tour_heatmap(ax4, coords, lkh_seq, cost_matrix, "LKH Baseline", matrix_lkh_cost, 0.00, wind_vector, wind_mag)
    
    # Kept simple for the paper; caption will do the heavy lifting
    plt.suptitle(f"Windy TSP Route Comparison (Instance {target_idx}) — Blue = Tailwind | Red = Headwind", 
                 fontsize=13, y=1.05)
    
    # Save as PDF for vector graphics
    out_img = f"visualizations/solution_visualization/PAPER_2_gap_comparison_idx{target_idx}_seed{seed}.png"
    
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, format='png', bbox_inches='tight')
    print(f"[*] Visual comparison saved to '{out_img}'")

    #save it too in svg format for vector graphics in paper
    out_svg = f"visualizations/solution_visualization/PAPER_2_gap_comparison_idx{target_idx}_seed{seed}.svg"
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    print(f"[*] Visual comparison saved to '{out_svg}'")

if __name__ == "__main__":
    COORDS_MODEL = "outputs/002_exp1_confidence_og_stats/05ConfOgStats_ent0.2_exp1/exp1_coords_ent0.2_20251223T185323/epoch-99.pt"
    STATS_MODEL = "outputs/windy_tsp_20-20/003_exp1_new_stats/06ConfNewStats_ent0.05_exp1/stats_learned_ent0.05_20251227T132116/epoch-99.pt"
    HYBRID_MODEL = "outputs/windy_tsp_20-20/008_exp3_graph_sparsification/13Random_Sparsification/tsp20_dual_hybrid_ent0.05_random20_20260302T160327/epoch-99.pt"
    DATASET = "data/windy_tsp/windy_tsp20_val.pkl"
    LKH_BASELINE = "results/windy_tsp20_val/windy_tsp20_valn1280-lkh_windy.pkl"
    INDEX = 803
    SEED = None

    compare_models_to_lkh(COORDS_MODEL, STATS_MODEL, HYBRID_MODEL, DATASET, LKH_BASELINE, SEED, INDEX)