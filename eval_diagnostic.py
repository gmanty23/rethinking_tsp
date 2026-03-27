import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils.functions import load_model, move_to
from problems.tsp.problem_tsp import TSPDataset

def manual_windy_cost(coords, wind, alpha, tour):
    """Manually calculates the Windy TSP cost to verify physics."""
    cost = 0.0
    n = len(tour)
    for step in range(n):
        i = tour[step]
        j = tour[(step + 1) % n] # Wrap around to start
        
        diff = coords[j] - coords[i]
        dist = np.linalg.norm(diff)
        
        if dist < 1e-8:
            continue
            
        u_hat = diff / dist
        wind_proj = np.dot(u_hat, wind)
        # Match the physics in problems/tsp/problem_tsp.py
        step_cost = dist * np.exp(-1.0 * alpha * wind_proj)
        cost += step_cost
    return cost

def diagnose_model(model_path, dataset_path, opts):
    # 1. Load Model and Environment
    print(f"[*] Loading model from {model_path}...")
    model, model_args = load_model(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 2. Load exactly 1 batch of data
    dataset = model.problem.make_dataset(
        filename=dataset_path, batch_size=2, num_samples=2, 
        neighbors=model_args.get('neighbors', 0.05), # Force low neighbors
        knn_strat=model_args.get('knn_strat', 'knn'), 
        node_feature_type=model_args.get('node_feature_type', 'coords'),
        supervised=True
    )
    
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
    nodes = move_to(batch['nodes'], device)
    graph = move_to(batch['graph'], device)
    
    # 3. Decode
    print("[*] Decoding with Greedy Search...")
    with torch.no_grad():
        model.set_decode_type("greedy")
        costs, ll, sequences, _ = model(nodes, graph, return_pi=True, return_entropy=True)
        
    # Extract Instance 0
    seq = sequences[0].cpu().numpy()
    model_cost = costs[0].item()
    node_feats = nodes[0].cpu().numpy()
    graph_mask = graph[0].cpu().numpy() # 0 = edge exists, 1 = masked
    
    coords = node_feats[:, :2]
    wind = node_feats[0, 2:4] # Wind is repeated, take first
    alpha = node_feats[0, 4]
    
    # --- DIAGNOSTIC CHECKS ---
    print("\n" + "="*40)
    print("DIAGNOSTIC REPORT")
    print("="*40)
    
    # Check 1: Validity
    is_valid = len(set(seq)) == len(coords) and len(seq) == len(coords)
    print(f"[1] Tour Validity: {'PASSED' if is_valid else 'FAILED'} (Visits: {len(set(seq))}/{len(coords)})")
    
    # Check 2: Cost Calculation Parity
    manual_cost = manual_windy_cost(coords, wind, alpha, seq)
    cost_diff = abs(model_cost - manual_cost)
    print(f"[2] Cost Check: Model Output = {model_cost:.4f} | Manual Eval = {manual_cost:.4f}")
    print(f"    Parity: {'PASSED' if cost_diff < 1e-4 else 'FAILED (Diff: ' + str(cost_diff) + ')'}")
    
    # Check 3: Graph Compliance (The Smoking Gun)
    illegal_edges = 0
    for step in range(len(seq)):
        i, j = seq[step], seq[(step + 1) % len(seq)]
        if graph_mask[i, j] == 1: # 1 means the edge was masked out!
            illegal_edges += 1
            
    print(f"[3] Graph Mask Compliance: {len(seq)-illegal_edges}/{len(seq)} edges are legal.")
    if illegal_edges > 0:
        print(f"    >>> ALERT: Model traversed {illegal_edges} edges that were supposed to be masked!")
    print("="*40 + "\n")

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c='black', s=50, zorder=5)
    
    # Plot Allowed Edges (Light Gray)
    for i in range(len(coords)):
        for j in range(len(coords)):
            if graph_mask[i, j] == 0 and i != j:
                plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], 
                         c='lightgray', linewidth=0.5, alpha=0.5, zorder=1)

    # Plot Model Tour (Red Arrows)
    for step in range(len(seq)):
        i, j = seq[step], seq[(step + 1) % len(seq)]
        color = 'red' if graph_mask[i, j] == 1 else 'green' # Red if illegal, Green if legal
        plt.annotate("", xy=coords[j], xytext=coords[i],
                     arrowprops=dict(arrowstyle="->", color=color, lw=2), zorder=3)
        
    plt.title(f"Windy TSP Tour (n={model_args.get('neighbors', 0.05)})\nGreen = Legal Edge | Red = Mask Violation")
    # Draw Wind Direction in corner
    plt.quiver(0.05, 0.95, wind[0], wind[1], scale=5, color='blue', alpha=0.5, label='Wind Vector')
    plt.legend()
    plt.savefig("diagnostic_tour.png", dpi=300)
    print("[*] Diagnostic plot saved to 'diagnostic_tour.png'")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("model", help="Path to the model checkpoint (.pt)")
    # parser.add_argument("dataset", help="Path to the dataset (.pkl)")
    # args = parser.parse_args()
    
    # diagnose_model(args.model, args.dataset, None)
    MODEL = "outputs/knn_neighbors_5/tsp20_dual_hybrid_ent0.05_percentage_n0.05_20260304T165731/epoch-99.pt"
    DATASET = "data/windy_tsp/windy_tsp20_val.pkl"

    diagnose_model(MODEL, DATASET, None)