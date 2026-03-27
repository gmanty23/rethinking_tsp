#!/usr/bin/env python

import math
import os
import argparse
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from your codebase
from utils import load_model, move_to, get_best

def ensure_lkh_baseline(dataset_path, val_size):
    """Loads LKH Baseline (Assuming it's already generated based on your previous logs)"""
    dataset_basename = os.path.basename(dataset_path)
    dataset_name_no_ext = os.path.splitext(dataset_basename)[0]
    lkh_dir = os.path.join("results", dataset_name_no_ext)
    lkh_filename = f"{dataset_name_no_ext}n{val_size}-lkh_windy.pkl"
    lkh_file_path = os.path.join(lkh_dir, lkh_filename)

    if os.path.isfile(lkh_file_path):
        with open(lkh_file_path, 'rb') as f:
            lkh_data = pickle.load(f)
            valid_rows = [row for row in lkh_data if row is not None]
            lkh_costs = np.array([row[0] for row in valid_rows])
            lkh_tours = np.array([row[1] for row in valid_rows])
            return lkh_costs, lkh_tours
    else:
        raise FileNotFoundError(f"LKH baseline not found at {lkh_file_path}. Run eval_baseline.py first.")

def extract_winning_instances(model_path, dataset_path, val_size, width, device):
    print(f"Loading Model: {model_path}...")
    model, train_opts = load_model(os.path.dirname(model_path), epoch=int(os.path.basename(model_path).split('-')[1].split('.')[0]))
    model.to(device)
    model.eval()
    model.set_decode_type("sampling", temp=1.0) # Beam search uses internal temp handling

    # Reconstruct Dataset Parameters
    detected_mode = "learned" if "learned" in model_path else "hybrid" if "hybrid" in model_path else "coords"
    eval_neighbors = train_opts.get('neighbors', 20)
    eval_knn_strat = train_opts.get('knn_strat', 'None')

    print("Generating/Loading Dataset...")
    dataset = model.problem.make_dataset(
        filename=dataset_path, batch_size=128, num_samples=val_size, 
        neighbors=eval_neighbors, knn_strat=eval_knn_strat,
        node_feature_type=detected_mode, supervised=True, nar=False
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    print("Loading LKH Baseline...")
    lkh_costs, lkh_tours = ensure_lkh_baseline(dataset_path, val_size)

    all_model_costs = []
    all_model_tours = []
    all_nodes = []
    all_graphs = []

    print(f"Running Beam Search (Width: {width})...")
    for batch in tqdm(dataloader, ascii=True):
        nodes, graph = move_to(batch['nodes'], device), move_to(batch['graph'], device)
        
        with torch.no_grad():
            cum_log_p, sequences, raw_costs, ids, batch_size = model.beam_search(
                nodes, graph, beam_size=width, compress_mask=False, max_calc_batch_size=10000
            )
            
            if sequences is not None:
                sequences, _ = get_best(
                    sequences.cpu().numpy(), raw_costs.cpu().numpy(),
                    ids.cpu().numpy() if ids is not None else None, batch_size
                )
                seq_tensor = torch.tensor(sequences, device=device)
                true_costs, _ = model.problem.get_costs(nodes, seq_tensor)
                
                all_model_costs.extend(true_costs.cpu().numpy())
                all_model_tours.extend(sequences)
                all_nodes.extend(nodes.cpu().numpy())
                all_graphs.extend(graph.cpu().numpy())

    all_model_costs = np.array(all_model_costs)
    
    # CALCULATE INSTANCE-LEVEL GAPS
    gaps = ((all_model_costs / lkh_costs) - 1) * 100
    
    # SORT ALL GAPS
    sorted_indices = np.argsort(gaps)
    
    # Grab the 10 closest to LKH (Best) and 10 furthest (Worst)
    best_indices = sorted_indices[:10]
    worst_indices = sorted_indices[-10:][::-1] # Reverse to get the absolute worst first
    
    target_indices = np.concatenate([best_indices, worst_indices])
    
    print(f"\n[+] Extracted Top 10 Best and Top 10 Worst instances.")
    
    winning_data = []
    for idx in target_indices:
        category = "BEST" if idx in best_indices else "WORST"
        winning_data.append({
            'category': category,
            'instance_id': idx,
            'model_cost': all_model_costs[idx],
            'lkh_cost': lkh_costs[idx],
            'gap_percentage': gaps[idx],
            'model_tour': all_model_tours[idx],
            'lkh_tour': lkh_tours[idx],
            'nodes_and_wind': all_nodes[idx],
            'graph_matrix': all_graphs[idx]
        })
        
    # Save the extracted graphs
    output_file = "results/winning_graphs.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(winning_data, f)
        
    print(f"Saved detailed data for winning graphs to: {output_file}")
    
    # Print the top 5 biggest wins
    if len(winning_data) > 0:
        print("\nTop 5 Biggest Wins:")
        winning_data.sort(key=lambda x: x['gap_percentage']) # Sort by most negative gap
        for i, data in enumerate(winning_data[:5]):
            print(f"  Instance {data['instance_id']:<4} | Model Cost: {data['model_cost']:.4f} | LKH Cost: {data['lkh_cost']:.4f} | Gap: {data['gap_percentage']:.2f}%")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("dataset", help="Path to the validation dataset")
    # parser.add_argument("model", help="Path to the model checkpoint (.pt)")
    # parser.add_argument("--val_size", type=int, default=1280)
    # parser.add_argument("--width", type=int, default=1280, help="Beam search width")
    # opts = parser.parse_args()
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # extract_winning_instances(opts.model, opts.dataset, opts.val_size, opts.width, device)

    DATASET_PATH = "data/windy_tsp/windy_tsp20_val.pkl"
    MODEL_PATH = "outputs/neighbours/tsp20_forward_hybrid_ent0.05_random_percentage_n0.15_20260305T005423/epoch-99.pt"
    VAL_SIZE = 1280
    BEAM_WIDTH = 1280
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extract_winning_instances(MODEL_PATH, DATASET_PATH, VAL_SIZE, BEAM_WIDTH, DEVICE)