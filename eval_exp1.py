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
    
    # Construct filename e.g. windy_tsp20_valn1280-lkh_windy.pkl
    lkh_filename = f"{dataset_name_no_ext}n{val_size}-lkh_windy.pkl"
    lkh_file_path = os.path.join(lkh_dir, lkh_filename)
    
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
    
    for batch in tqdm(dataloader, disable=opts.no_progress_bar, ascii=True):
        # Move to GPU
        nodes, graph = move_to(batch['nodes'], device), move_to(batch['graph'], device)
        
        start = time.time()
        with torch.no_grad():
            
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
    avg_gap = 0.0
    if lkh_costs is not None:
        if len(lkh_costs) == len(costs):
            gaps = ((costs / lkh_costs) - 1) * 100
            avg_gap = gaps.mean()
        else:
            print(f"Size mismatch: LKH {len(lkh_costs)} vs Pred {len(costs)}. Gap not calculated.")

    return avg_cost, avg_gap, avg_time, avg_conf

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
            writer.writerow(['Model_Name', 'Strategy', 'Width', 'Avg_Cost', 'Gap_Percent', 'Time_Per_Inst', 'Avg_Confidence'])
    
    # Write LKH as its own Row (Reference)
    # We write this once per script run to ensure the baseline is visible for this session
    if lkh_times is not None:
        with open(opts.csv_out, mode='a', newline='') as f:
            writer = csv.writer(f)
            # LKH row format: Name="LKH_Baseline", Strategy="opt", Width=0, Cost=lkh_avg, Gap=0, Time=lkh_time, Conf=1.0
            writer.writerow(['LKH_Baseline', 'opt', 0, f"{lkh_cost_avg:.4f}", "0.0000", f"{lkh_time_avg:.4f}", "1.0000"])

    # --- MAIN LOOP ---
    for model_path in opts.models:
        # Load Model
        # We assume the model path ends in 'epoch-X.pt', we need the directory for load_model
        model_dir = os.path.dirname(model_path)
        epoch = int(os.path.basename(model_path).split('-')[1].split('.')[0])
        
        # Extract a pretty name (e.g., 'exp1_hybrid')
        model_name = os.path.basename(model_dir)

        print(f"Loading Model: {model_name} (Epoch {epoch})...")
        model, _ = load_model(model_dir, epoch=epoch)
        
        # Generate Dataset (Once per model to ensure correct feature type usage)
        # Note: model.problem.make_dataset uses the model's args (coords/hybrid/etc) automatically
        dataset = model.problem.make_dataset(
            filename=opts.dataset, batch_size=opts.batch_size, num_samples=opts.val_size, 
            neighbors=20, knn_strat='percentage' # Standard defaults
        )

        for width in opts.widths:
            # Determine Strategy
            strategy = 'greedy' if width == 0 else 'bs'
            
            # Special Case: If you want to test Sampling separately, you can add logic here.
            
            print(f"  -> Running {strategy.upper()} width={width}...")
            
            cost, gap, duration, conf = eval_dataset(
                model, dataset, lkh_costs, strategy, width, 1.0, opts, device
            )
            
            # Write to CSV immediately - OPTION B: Removed lkh_time_avg from this row
            with open(opts.csv_out, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([model_name, strategy, width, f"{cost:.4f}", f"{gap:.4f}", f"{duration:.4f}", f"{conf:.4f}"])
            
            # Console Log: Cleaned up LKH time from the loop print as requested (implicitly) to focus on model
            print(f"     Gap: {gap:.2f}% | Time: {duration:.4f}s")

    print(f"\nResults saved to {opts.csv_out}")