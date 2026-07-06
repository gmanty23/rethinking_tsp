#!/usr/bin/env python

import math
import os
import time
import argparse
import numpy as np
import torch
import csv
import pickle
import subprocess
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import utilities from your existing codebase
from utils import load_model, move_to, get_best
from utils.functions import parse_softmax_temperature
from nets.nar_model import NARModel

def compute_embedding_variance(embeddings):
    """
    Calcula la varianza espacial promedio de los embeddings de los nodos
    para medir el over-smoothing en el GNN.
    Args:
        embeddings: Tensor de dimensiones (batch_size, num_nodes, embedding_dim)
    """
    node_variance = torch.var(embeddings, dim=1, unbiased=False) 
    mean_variance_per_graph = node_variance.mean(dim=-1) 
    return mean_variance_per_graph

def compute_dirichlet_energy(embeddings, graph=None):
    """
    Calcula la Energía de Dirichlet promedio del batch.
    A prueba de fallos: Si 'graph' está vacío o es inválido, asume Fully Connected.
    """
    B, N, D = embeddings.shape
    # Distancias cuadradas entre todos los pares: ||h_i - h_j||^2 -> (B, N, N)
    sq_dists = torch.cdist(embeddings, embeddings, p=2).pow(2)
    
    # Máscara por defecto: Fully Connected (todos con todos, excluyendo la diagonal)
    mask = torch.ones(B, N, N, device=embeddings.device) - torch.eye(N, device=embeddings.device).unsqueeze(0)
    
    # Si recibimos un grafo, vamos a limpiarlo y verificar si tiene aristas reales
    if graph is not None and isinstance(graph, torch.Tensor) and graph.dim() == 3 and graph.shape[1] == N:
        # 1. Binarizar por si el tensor contiene costes/pesos en lugar de adyacencia
        binary_graph = (graph != 0).float()
        
        # 2. Eliminar las conexiones del nodo consigo mismo (self-loops)
        binary_graph = binary_graph * mask
        
        # 3. Solo usamos este grafo si realmente tiene aristas definidas
        if binary_graph.sum() > 0:
            mask = binary_graph

    # Calcular la energía promedio sobre las aristas válidas de la máscara
    energy_per_graph = (sq_dists * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-9)
        
    return energy_per_graph.mean()

def ensure_lkh_baseline(dataset_path, val_size):
    """
    Checks if LKH baseline exists. If not, it triggers the generation 
    using eval_baseline.py (via subprocess).
    Returns (lkh_costs, lkh_durations)
    """
    dataset_basename = os.path.basename(dataset_path)
    dataset_name_no_ext = os.path.splitext(dataset_basename)[0]
    
    lkh_dir = os.path.join("results", dataset_name_no_ext)
    lkh_filename = f"{dataset_name_no_ext}n{val_size}-lkh_windy.pkl"
    lkh_file_path = os.path.join(lkh_dir, lkh_filename)

    # 1. Check if file exists
    if not os.path.isfile(lkh_file_path):
        print(f"[-] LKH Baseline not found at {lkh_file_path}")
        print(f"[-] Attempting to generate it now using eval_baseline.py...")
        
        os.makedirs(lkh_dir, exist_ok=True)
        
        cmd = [
            sys.executable, "eval_baseline.py", 
            "lkh_windy", 
            dataset_path, 
            "-n", str(val_size),
            "--disable_cache", 
            "-f"
        ]
        
        try:
            subprocess.check_call(cmd)
            print("[-] LKH Generation Complete.")
        except subprocess.CalledProcessError as e:
            print(f"[!] Error generating LKH baseline: {e}")
            print("[!] Continuing without LKH (Gaps will be 0.0)")
            return None, None
            
    # 2. Load the file
    if os.path.isfile(lkh_file_path):
        print(f"Loading LKH Baseline from: {lkh_file_path}")
        with open(lkh_file_path, 'rb') as f:
            lkh_data = pickle.load(f)
            valid_rows = [row for row in lkh_data if row is not None]
            
            lkh_costs = np.array([row[0] for row in valid_rows])
            lkh_times = np.array([row[2] for row in valid_rows]) 
            
            return lkh_costs, lkh_times
    else:
        return None, None

def eval_dataset(model, dataset, lkh_costs, decode_strategy, width, softmax_temp, opts, device):
    """
    Evaluates a single model on the dataset and returns stats.
    """
    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if decode_strategy == 'greedy' else "sampling",
        temp=softmax_temp
    )

    dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    results = []
    batch_variances = []  
    batch_energies = []   
    
    for batch in tqdm(dataloader, disable=opts.no_progress_bar, ascii=True):
        nodes, graph = move_to(batch['nodes'], device), move_to(batch['graph'], device)
        cost_matrix = move_to(batch['cost_matrix'], device) if 'cost_matrix' in batch else None
        
        # ==========================================================
        # 1. PURE INFERENCE (TIMED)
        # ==========================================================
        start = time.time()
        with torch.no_grad():
            
            # --- STRATEGY 1: GREEDY ---
            if decode_strategy == 'greedy':
                costs, ll, sequences, entropy = model(
                    nodes, graph, 
                    cost_matrix=cost_matrix, 
                    return_pi=True, 
                    return_entropy=True
                )
                
                seq_len = nodes.size(1)
                confidence = (ll / seq_len).exp().cpu().numpy()
                costs = costs.cpu().numpy()
                sequences = sequences.cpu().numpy()
                
            # --- STRATEGY 2: BEAM SEARCH ---
            elif decode_strategy == 'bs':
                run_width = max(1, width) 
                
                cum_log_p, sequences, raw_costs, ids, batch_size = model.beam_search(
                    nodes, graph, beam_size=run_width,
                    compress_mask=opts.compress_mask,
                    max_calc_batch_size=opts.max_calc_batch_size,
                    cost_matrix=cost_matrix
                )
                
                seq_len = nodes.size(1) 
                confidence = (cum_log_p / seq_len).exp().cpu().numpy()

                if sequences is None:
                    costs = [math.inf] * batch_size
                    confidence = [0] * batch_size
                else:
                    sequences, _ = get_best(
                        sequences.cpu().numpy(), raw_costs.cpu().numpy(),
                        ids.cpu().numpy() if ids is not None else None,
                        batch_size
                    )
                    
                    seq_tensor = torch.tensor(np.array(sequences), device=device)
                    true_costs, _ = model.problem.get_costs(nodes, seq_tensor)
                    costs = true_costs.cpu().numpy()

        # >> STOP TIMER: Only standard network inference is captured! <<
        duration = time.time() - start


        # ==========================================================
        # 2. LATENT METRICS ANALYSIS (NOT TIMED)
        # ==========================================================
        with torch.no_grad():
            try:
                if hasattr(model, 'embedder'):
                    h = model._init_embed(nodes)
                    
                    # Generate NAB locally to avoid the CachedLookup object entirely
                    nab_bias = None
                    if getattr(model, 'nab_mode', 'none') in ['encoder', 'decoder', 'both'] and cost_matrix is not None:
                        coords = nodes[..., 0:2]
                        safe_costs_nab = torch.log(cost_matrix + 1e-8)
                        nab_bias = model.nab_generator(coords, safe_costs_nab)
                        
                    enc_bias = nab_bias if getattr(model, 'nab_mode', 'none') in ['encoder', 'both'] else None
                    
                    # Generate clean embeddings directly from the embedder
                    embeddings = model.embedder(h, graph.clone(), cost_matrix=cost_matrix, nab_bias=enc_bias)
                    
                    var = compute_embedding_variance(embeddings)
                    batch_variances.append(var.mean().item())
                    
                    energy = compute_dirichlet_energy(embeddings, graph.clone())
                    batch_energies.append(energy.item())
                    
            except Exception as e:
                print(f"\n[!] Error calculando métricas latentes: {e}")

        # ==========================================================

        # Format results per instance
        for i, cost in enumerate(costs):
            results.append((cost, duration, confidence[i] if isinstance(confidence, (list, np.ndarray)) else 0))

    # --- AGGREGATE RESULTS ---
    costs, durations, confs = zip(*results)
    costs = np.array(costs)
    durations = np.array(durations)
    confs = np.array(confs)
    
    avg_cost = costs.mean()
    avg_time = durations.sum() / len(costs) 
    avg_conf = confs.mean()

    gap_mean_of_ratios = 0.0
    gap_std_of_ratios = 0.0
    gap_ratio_of_means = 0.0

    if lkh_costs is not None:
        if len(lkh_costs) == len(costs):
            gaps = ((costs / lkh_costs) - 1) * 100
            gap_mean_of_ratios = gaps.mean()
            gap_std_of_ratios = gaps.std()
            gap_ratio_of_means = ((costs.sum() / lkh_costs.sum()) - 1) * 100
        else:
            print(f"Size mismatch: LKH {len(lkh_costs)} vs Pred {len(costs)}. Gap not calculated.")

    avg_variance = sum(batch_variances) / len(batch_variances) if batch_variances else 0.0
    avg_energy = sum(batch_energies) / len(batch_energies) if batch_energies else 0.0 

    return avg_cost, gap_mean_of_ratios, gap_std_of_ratios, gap_ratio_of_means, avg_time, avg_conf, avg_variance, avg_energy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Filename of the dataset to evaluate")
    parser.add_argument("--models", nargs='+', required=True, help="List of model paths (checkpoints) to evaluate")
    parser.add_argument("--csv_out", default="results/comprehensive_results.csv", help="Path to save CSV results")
    
    parser.add_argument('--val_size', type=int, default=1280, help='Number of instances to evaluate')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_calc_batch_size', type=int, default=10000)
    
    parser.add_argument('--widths', type=int, nargs='+', default=[0, 10, 100, 1280], 
                        help='Widths for Beam Search (0=Greedy)')
    
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--num_workers', type=int, default=0, help='Num workers')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--legacy_mode', action='store_true', help='Bypass new normalization to evaluate old models')

    opts = parser.parse_args()

    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # --- LOAD / GENERATE LKH BASELINE ---
    lkh_costs, lkh_times = ensure_lkh_baseline(opts.dataset, opts.val_size)
    
    lkh_time_avg = 0.0
    lkh_cost_avg = 0.0
    if lkh_times is not None:
        lkh_time_avg = lkh_times.mean()
        lkh_cost_avg = lkh_costs.mean()
        print(f"LKH Baseline Loaded. Avg Time per Instance: {lkh_time_avg:.4f}s")
    else:
        print("Warning: Running without LKH Baseline.")

    file_exists = os.path.isfile(opts.csv_out)
    with open(opts.csv_out, mode='w' if not file_exists else 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Model_Name', 'Strategy', 'Width', 'Avg_Cost', 'Gap_MoR', 'Gap_STDoR', 'Gap_RoM', 'Time_Per_Inst', 'Avg_Confidence','Avg_Embedding_Variance', 'Avg_Dirichlet_Energy'])
    
    if lkh_times is not None:
        with open(opts.csv_out, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['LKH_Baseline', 'opt', 0, f"{lkh_cost_avg:.4f}", f"0.0000", f"0.0000", "0.0000", f"{lkh_time_avg:.4f}", "1.0000", "0.0000", "0.0000"])

    # --- MAIN LOOP ---
    for model_path in opts.models:
        model_dir = os.path.dirname(model_path)
        epoch = int(os.path.basename(model_path).split('-')[1].split('.')[0])
        model_name = f"{os.path.basename(model_dir)}_epoch{epoch}"

        print(f"Loading Model: {model_name} (Epoch {epoch})...")
        model, train_opts = load_model(model_dir, epoch=epoch)

        # 1. Extract Settings Directly from Training Options
        detected_mode = train_opts.get('node_feature_type', 'coords')
        emb_type = train_opts.get('node_embedding_type', 'original')
        use_wind = train_opts.get('use_wind', False)
        
        # 2. Detect Neighbors (Assume Full Graph / None unless specified)
        eval_neighbors = train_opts.get('neighbors', 20)
        eval_knn_strat = train_opts.get('knn_strat', 'None')

        print(f"Arch: {emb_type} | Feature: {detected_mode} | Wind: {use_wind} | Neighbors: {eval_neighbors} | KNN: {eval_knn_strat}")

        # Generate Dataset
        dataset = model.problem.make_dataset(
            filename=opts.dataset, 
            batch_size=opts.batch_size, 
            num_samples=opts.val_size, 
            neighbors=eval_neighbors,
            knn_strat=eval_knn_strat,
            node_feature_type=detected_mode, 
            supervised=True, 
            nar=False,
            legacy_mode=opts.legacy_mode
        )

        for width in opts.widths:
            strategy = 'greedy' if width == 0 else 'bs'
            
            print(f"  -> Running {strategy.upper()} width={width}...")
            
            cost, gap_mor, std_mor, gap_rom, duration, conf, variance, energy = eval_dataset(
                model, dataset, lkh_costs, strategy, width, 1.0, opts, device
            )
            
            with open(opts.csv_out, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([model_name, strategy, width, f"{cost:.4f}", f"{gap_mor:.4f}", f"{std_mor:.4f}", f"{gap_rom:.4f}", f"{duration:.4f}", f"{conf:.4f}", f"{variance:.4f}", f"{energy:.4f}"])            
            
            print(f"     Gap (MoR): {gap_mor:.2f}% | Gap (RoM): {gap_rom:.2f}% | Time: {duration:.4f}s | Var: {variance:.4f} | Dir. Energy: {energy:.4f}")

    print(f"\nResults saved to {opts.csv_out}")