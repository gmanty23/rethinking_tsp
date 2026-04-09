import torch
import numpy as np
import argparse
import os
import pickle
from tqdm import tqdm  # Added for the progress bar
from utils.functions import load_model, move_to, get_best

def extract_exact_wind(coords, cost_matrix):
    """
    Extracts the exact wind vector (w * alpha) using least squares.
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

def get_tour(model, current_nodes, current_graph, WIDTH=10):
    """Helper for model inference."""
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

def find_matching_instances(orig_model_path, imp_model_path, dataset_path, lkh_path, 
                            min_wind_strength, min_og_opt_gap, max_imp_opt_gap):
    
    # print(f"[*] Loading LKH Baseline from {lkh_path}...")
    with open(lkh_path, 'rb') as f:
        lkh_data = pickle.load(f)
    if not isinstance(lkh_data, list):
        lkh_data = list(lkh_data)
        
    total_instances = len(lkh_data)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load models ONCE outside the loop to save time
    # print(f"[*] Loading Original Model...")
    orig_model, orig_args = load_model(orig_model_path)
    orig_model.to(device)
    orig_model.eval()

    # print(f"[*] Loading Improved Model...")
    imp_model, imp_args = load_model(imp_model_path)
    imp_model.to(device)
    imp_model.eval()

    matching_instances = []

    print(f"\n[*] Beginning search across {total_instances} instances...")
    print(f"    Criteria: Wind >= {min_wind_strength} | Orig Gap >= {min_og_opt_gap}% | Imp Gap <= {max_imp_opt_gap}%\n")

    # Initialize the progress bar
    pbar = tqdm(range(total_instances), desc="Scanning Instances", unit="inst")

    for i in pbar:
        # ---------------------------------------------------------
        # 1. Wind Calculation (Short-circuit check #1)
        # ---------------------------------------------------------
        orig_dataset = orig_model.problem.make_dataset(
            filename=dataset_path, batch_size=1, num_samples=1, offset=i, 
            neighbors=orig_args.get('neighbors', 20), 
            knn_strat=orig_args.get('knn_strat', 'None'), 
            node_feature_type=orig_args.get('node_feature_type', 'coords'),
            supervised=True
        )
        orig_batch = next(iter(torch.utils.data.DataLoader(orig_dataset, batch_size=1)))
        orig_nodes = move_to(orig_batch['nodes'], device)
        orig_graph = move_to(orig_batch['graph'], device)

        cost_matrix = orig_batch['cost_matrix'][0].cpu().numpy()
        coords = orig_nodes[0].cpu().numpy()[:, :2]

        _, wind_mag = extract_exact_wind(coords, cost_matrix)
        
        if wind_mag < min_wind_strength:
            continue

        # ---------------------------------------------------------
        # 2. Original Model Inference (Short-circuit check #2)
        # ---------------------------------------------------------
        matrix_lkh_cost = lkh_data[i][0]
        
        orig_seq = get_tour(orig_model, orig_nodes, orig_graph)
        orig_seq_tensor = torch.tensor(orig_seq, dtype=torch.long, device=device).unsqueeze(0)
        orig_true_costs, _ = orig_model.problem.get_costs(orig_nodes, orig_seq_tensor)
        
        orig_opt_gap = ((orig_true_costs.item() / matrix_lkh_cost) - 1.0) * 100
        
        if orig_opt_gap < min_og_opt_gap:
            continue

        # ---------------------------------------------------------
        # 3. Improved Model Inference (Final check)
        # ---------------------------------------------------------
        imp_dataset = imp_model.problem.make_dataset(
            filename=dataset_path, batch_size=1, num_samples=1, offset=i, 
            neighbors=imp_args.get('neighbors', 20), 
            knn_strat=imp_args.get('knn_strat', 'None'), 
            node_feature_type=imp_args.get('node_feature_type', 'coords'),
            supervised=True
        )
        imp_batch = next(iter(torch.utils.data.DataLoader(imp_dataset, batch_size=1)))
        imp_nodes = move_to(imp_batch['nodes'], device)
        imp_graph = move_to(imp_batch['graph'], device)

        imp_seq = get_tour(imp_model, imp_nodes, imp_graph)
        imp_seq_tensor = torch.tensor(imp_seq, dtype=torch.long, device=device).unsqueeze(0)
        imp_true_costs, _ = imp_model.problem.get_costs(imp_nodes, imp_seq_tensor)
        
        imp_opt_gap = ((imp_true_costs.item() / matrix_lkh_cost) - 1.0) * 100

        if imp_opt_gap > max_imp_opt_gap:
            continue

        # ---------------------------------------------------------
        # Match Found!
        # ---------------------------------------------------------
        matching_instances.append({
            'index': i,
            'wind_mag': wind_mag,
            'orig_gap': orig_opt_gap,
            'imp_gap': imp_opt_gap
        })
        
        # Dynamically update the progress bar with the number of matches found
        pbar.set_postfix(matches=len(matching_instances))

    # Print Final Summary
    print("\n" + "="*60)
    print(f"SEARCH COMPLETE: Found {len(matching_instances)} matching instances.")
    print("="*60)
    
    if matching_instances:
        print(f"{'INDEX':<8} | {'WIND MAG':<12} | {'ORIG GAP (%)':<15} | {'IMP GAP (%)':<15}")
        print("-" * 60)
        for match in matching_instances:
            print(f"{match['index']:<8} | {match['wind_mag']:<12.4f} | {match['orig_gap']:<15.2f} | {match['imp_gap']:<15.2f}")
    
    return matching_instances


if __name__ == "__main__":
    ORIG_MODEL = "outputs/002_exp1_confidence_og_stats/05ConfOgStats_ent0.2_exp1/exp1_coords_ent0.2_20251223T185323/epoch-99.pt"
    IMP_MODEL = "outputs/windy_tsp_20-20/008_exp3_graph_sparsification/13Random_Sparsification/tsp20_dual_hybrid_ent0.05_random20_20260302T160327/epoch-99.pt"
    DATASET = "data/windy_tsp/windy_tsp20_val.pkl"
    LKH_BASELINE = "results/windy_tsp20_val/windy_tsp20_valn1280-lkh_windy.pkl"
    
    # --- Define your search criteria here ---
    MIN_WIND = 0.1         # Example: Wind magnitude must be at least 0.5
    MIN_ORIG_GAP = 1     # Example: Original model must be at least 5% worse than LKH
    MAX_IMP_GAP = 2.0      # Example: Improved model must be at most 1% worse than LKH
    
    matches = find_matching_instances(
        orig_model_path=ORIG_MODEL, 
        imp_model_path=IMP_MODEL, 
        dataset_path=DATASET, 
        lkh_path=LKH_BASELINE,
        min_wind_strength=MIN_WIND,
        min_og_opt_gap=MIN_ORIG_GAP,
        max_imp_opt_gap=MAX_IMP_GAP
    )