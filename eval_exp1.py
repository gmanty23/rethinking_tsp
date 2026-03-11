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
    
    UPDATED: Matches the folder structure output by eval_baseline.py
    Folder: results/<dataset_name_no_ext>/
    File:   <dataset_name_no_ext>n<val_size>-lkh_windy.pkl
    """
    dataset_basename = os.path.basename(dataset_path)
    dataset_name_no_ext = os.path.splitext(dataset_basename)[0]
    
    # Match the path structure used in your scripts: results/dataset_name/filename.pkl
    lkh_dir = os.path.join("results", dataset_name_no_ext)
    # lkh_dir = "/home/pfc/gms/code/rethinking_tsp/results/lkh_windy"
    # Construct filename e.g. windy_tsp20_valn1280-lkh_windy.pkl
    lkh_filename = f"{dataset_name_no_ext}n{val_size}-lkh_windy.pkl"
    lkh_file_path = os.path.join(lkh_dir, lkh_filename)
    # lkh_file_path = "/home/pfc/gms/code/rethinking_tsp/results/lkh_windy/windy_tsp50_val_lkh.pkl"

    # 1. Check if file exists
    if not os.path.isfile(lkh_file_path):
        print(f"[-] LKH Baseline not found at {lkh_file_path}")
        print(f"[-] Attempting to generate it now using eval_baseline.py...")
        
        # Ensure directory exists
        os.makedirs(lkh_dir, exist_ok=True)
        
        # Call the baseline script: python eval_baseline.py lkh_windy <dataset> -n <size>
        cmd = [
            sys.executable, "eval_baseline.py", 
            "lkh_windy", 
            dataset_path, 
            "-n", str(val_size),
            "--disable_cache", # Force fresh run to ensure we get a file
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
            # lkh_data is a list of tuples: (cost, tour, duration)
            # We filter out None rows just in case
            valid_rows = [row for row in lkh_data if row is not None]
            
            lkh_costs = np.array([row[0] for row in valid_rows])
            lkh_times = np.array([row[2] for row in valid_rows]) # Index 2 is duration
            
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
    batch_variances = []  # For storing embedding variances per batch
    batch_energies = []   # NUEVO: Para almacenar la energía de Dirichlet
    
    for batch in tqdm(dataloader, disable=opts.no_progress_bar, ascii=True):
        # Move to GPU
        nodes, graph = move_to(batch['nodes'], device), move_to(batch['graph'], device)
        
        start = time.time()
        with torch.no_grad():

            # --- CAPTURE EMBEDDINGS AND CALCULATE VARIANCE & ENERGY ---
            try:
                if hasattr(model, 'embedder'):
                    h = model._init_embed(nodes)
                    embeddings = model.embedder(h, graph)
                    
                    # Varianza
                    var = compute_embedding_variance(embeddings)
                    batch_variances.append(var.mean().item())
                    
                    # NUEVO: Energía de Dirichlet
                    # Pasamos 'graph' para que evalúe la topología local si es posible
                    energy = compute_dirichlet_energy(embeddings, graph)
                    batch_energies.append(energy.item())
                    
            except Exception as e:
                print(f"\n[!] Error calculando métricas latentes: {e}")
            # -------------------------------------------------
            
            # --- STRATEGY 1: GREEDY ---
            if decode_strategy == 'greedy':
                # Greedy returns the CORRECT cost automatically because it calls problem.get_costs internally
                costs, ll, sequences, entropy = model(
                    nodes, graph, 
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
                
                # We get 'raw_costs' here, which we suspect are WRONG/Internal Scores
                cum_log_p, sequences, raw_costs, ids, batch_size = model.beam_search(
                    nodes, graph, beam_size=run_width,
                    compress_mask=opts.compress_mask,
                    max_calc_batch_size=opts.max_calc_batch_size
                )
                
                # Calculate Confidence
                seq_len = nodes.size(1) 
                confidence = (cum_log_p / seq_len).exp().cpu().numpy()

                if sequences is None:
                    costs = [math.inf] * batch_size
                    confidence = [0] * batch_size
                else:
                    # 1. Get the best sequences using the internal scores
                    sequences, _ = get_best(
                        sequences.cpu().numpy(), raw_costs.cpu().numpy(),
                        ids.cpu().numpy() if ids is not None else None,
                        batch_size
                    )
                    
                    # 2. CRITICAL FIX: RE-CALCULATE COSTS
                    # We take the sequences found by BS and measure them with the Real Windy Ruler
                    # We must ensure sequences is a Tensor on GPU for get_costs
                    seq_tensor = torch.tensor(sequences, device=device)
                    
                    # problem.get_costs expects (input_data, pi)
                    # 'nodes' contains the coordinates + wind features
                    true_costs, _ = model.problem.get_costs(nodes, seq_tensor)
                    costs = true_costs.cpu().numpy()
                    
            # --- STRATEGY 3: SAMPLING ---
            else:
                sequences, raw_costs = model.sample_many(nodes, graph, batch_rep=width, iter_rep=1)
                confidence = [0] * len(raw_costs) 
                
                # Re-calculate costs for Sampling too, just to be safe
                seq_tensor = torch.tensor(sequences, device=device) if not torch.is_tensor(sequences) else sequences
                true_costs, _ = model.problem.get_costs(nodes, seq_tensor)
                costs = true_costs.cpu().numpy()

        duration = time.time() - start
        
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

    # Calculate Gap if Baseline exists
    gap_mean_of_ratios = 0.0
    gap_ratio_of_means = 0.0

    if lkh_costs is not None:
        if len(lkh_costs) == len(costs):
            # 1. Mean of Ratios (Eval Standard): Average of individual gaps
            gaps = ((costs / lkh_costs) - 1) * 100
            gap_mean_of_ratios = gaps.mean()

            # 2. Ratio of Means (Train Standard): Gap of the totals
            gap_ratio_of_means = ((costs.sum() / lkh_costs.sum()) - 1) * 100
        else:
            print(f"Size mismatch: LKH {len(lkh_costs)} vs Pred {len(costs)}. Gap not calculated.")

    avg_variance = sum(batch_variances) / len(batch_variances) if batch_variances else 0.0
    avg_energy = sum(batch_energies) / len(batch_energies) if batch_energies else 0.0 

  
    return avg_cost, gap_mean_of_ratios, gap_ratio_of_means, avg_time, avg_conf, avg_variance, avg_energy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Filename of the dataset to evaluate")
    parser.add_argument("--models", nargs='+', required=True, help="List of model paths (checkpoints) to evaluate")
    parser.add_argument("--csv_out", default="results/comprehensive_results.csv", help="Path to save CSV results")
    
    # Evaluation Settings
    parser.add_argument('--val_size', type=int, default=1280, help='Number of instances to evaluate')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_calc_batch_size', type=int, default=10000)
    
    # Grid Search Settings (Hardcoded defaults as per experiment, or overrideable)
    parser.add_argument('--widths', type=int, nargs='+', default=[0, 10, 100, 1280], 
                        help='Widths for Beam Search (0=Greedy)')
    
    # Misc
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--num_workers', type=int, default=0, help='Num workers')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')


    opts = parser.parse_args()

    # Setup Device
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # --- LOAD / GENERATE LKH BASELINE ---
    # We load this ONCE at the start to get costs AND timing
    lkh_costs, lkh_times = ensure_lkh_baseline(opts.dataset, opts.val_size)
    
    # Calculate Average LKH Time (Computation Time per Instance)
    lkh_time_avg = 0.0
    lkh_cost_avg = 0.0
    if lkh_times is not None:
        lkh_time_avg = lkh_times.mean()
        lkh_cost_avg = lkh_costs.mean()
        print(f"LKH Baseline Loaded. Avg Time per Instance: {lkh_time_avg:.4f}s")
    else:
        print("Warning: Running without LKH Baseline.")

    # Initialize CSV
    file_exists = os.path.isfile(opts.csv_out)
    with open(opts.csv_out, mode='w' if not file_exists else 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # NUEVO: Añadida 'Avg_Dirichlet_Energy' al final
            writer.writerow(['Model_Name', 'Strategy', 'Width', 'Avg_Cost', 'Gap_MoR', 'Gap_RoM', 'Time_Per_Inst', 'Avg_Confidence','Avg_Embedding_Variance', 'Avg_Dirichlet_Energy'])
    
    # Write LKH as its own Row (Reference)
    if lkh_times is not None:
        with open(opts.csv_out, mode='a', newline='') as f:
            writer = csv.writer(f)
            # NUEVO: Añadido otro "0.0000" al final para la energía
            writer.writerow(['LKH_Baseline', 'opt', 0, f"{lkh_cost_avg:.4f}", "0.0000", "0.0000", f"{lkh_time_avg:.4f}", "1.0000", "0.0000", "0.0000"])

    # --- MAIN LOOP ---
    for model_path in opts.models:
        # Load Model
        # We assume the model path ends in 'epoch-X.pt', we need the directory for load_model
        model_dir = os.path.dirname(model_path)
        epoch = int(os.path.basename(model_path).split('-')[1].split('.')[0])
        
        # Extract a pretty name (e.g., 'exp1_hybrid')
        model_name = os.path.basename(model_dir)

        print(f"Loading Model: {model_name} (Epoch {epoch})...")
        model, train_opts = load_model(model_dir, epoch=epoch)

        # 1. Detect Mode from Filename
        if "learned" in model_name:
            detected_mode = "learned"
        elif "hybrid" in model_name:
            detected_mode = "hybrid"
        elif "coords" in model_name:
            detected_mode = "coords"
        else:
            print(f"[!] Warning: Could not detect mode from name '{model_name}'. Defaulting to 'coords'.")
            detected_mode = "coords"

        # 2. Detect Neighbors (Assume Full Graph / None unless specified)
        # If you know you trained with KNN, change this default to 20
        eval_neighbors = train_opts.get('neighbors', 20)
        eval_knn_strat= train_opts.get('knn_strat', 'None')
        # print(train_opts)
        # wait = input("Press Enter to continue with these settings...")
        print(f"Detected Feature Type: {detected_mode} | Neighbors: {eval_neighbors} | KNN Strat: {eval_knn_strat}")
        # wait = input("Press Enter to confirm and continue...")
        # Generate Dataset (Once per model to ensure correct feature type usage)
        # Note: model.problem.make_dataset uses the model's args (coords/hybrid/etc) automatically
        # Generate Dataset matching the run.py style
        dataset = model.problem.make_dataset(
            filename=opts.dataset, 
            batch_size=opts.batch_size, 
            num_samples=opts.val_size, 
            neighbors=eval_neighbors,
            knn_strat=eval_knn_strat,
            node_feature_type=detected_mode, 
            supervised=True, 
            nar=False
        )

        for width in opts.widths:
            # Determine Strategy
            strategy = 'greedy' if width == 0 else 'bs'
            
            # Special Case: If you want to test Sampling separately, you can add logic here.
            
            print(f"  -> Running {strategy.upper()} width={width}...")
            
            # NUEVO: Añadido 'energy' al desempaquetado (7 valores)
            cost, gap_mor, gap_rom, duration, conf, variance, energy = eval_dataset(
                model, dataset, lkh_costs, strategy, width, 1.0, opts, device
            )
            
            # Write to CSV immediately
            with open(opts.csv_out, mode='a', newline='') as f:
                writer = csv.writer(f)
                # NUEVO: Añadido f"{energy:.4f}" al final
                writer.writerow([model_name, strategy, width, f"{cost:.4f}", f"{gap_mor:.4f}", f"{gap_rom:.4f}", f"{duration:.4f}", f"{conf:.4f}", f"{variance:.4f}", f"{energy:.4f}"])
            
            # Console Log
            # NUEVO: Añadida la Energía al print
            print(f"     Gap (MoR): {gap_mor:.2f}% | Gap (RoM): {gap_rom:.2f}% | Time: {duration:.4f}s | Var: {variance:.4f} | Dir. Energy: {energy:.4f}")

    print(f"\nResults saved to {opts.csv_out}")