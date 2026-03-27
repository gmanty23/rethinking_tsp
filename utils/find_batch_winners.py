#!/usr/bin/env python

import os
import argparse
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader

# Import utilities from your codebase
from utils import load_model, move_to, get_best

def ensure_lkh_baseline(dataset_path, val_size):
    """Loads the pre-calculated LKH Baseline."""
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
            return lkh_costs
    else:
        raise FileNotFoundError(f"LKH baseline not found at {lkh_file_path}. Please run eval_baseline.py first.")

def batch_evaluate_models(models, dataset_path, val_size, output_file, device):
    print(f"[*] Loading LKH Baseline for {val_size} instances...")
    lkh_costs = ensure_lkh_baseline(dataset_path, val_size)
    
    widths = [0, 10, 100, 1280]
    
    print(f"[*] Found {len(models)} models to evaluate.")
    print("=" * 80)

    # Prepare document for saving
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    report_lines = [
        "WINDY TSP NCO ABLATION - WINNING MODELS REPORT",
        "=" * 80,
        f"Dataset: {dataset_path} ({val_size} instances)",
        "=" * 80,
        ""
    ]

    for model_path in models:
        model_name = os.path.basename(os.path.dirname(model_path))
        print(f"\nAnalyzing: {model_name}")
        report_lines.append(f"Model: {model_name}")
        
        try:
            # 1. Load Model
            epoch = int(os.path.basename(model_path).split('-')[1].split('.')[0])
            model, train_opts = load_model(os.path.dirname(model_path), epoch=epoch)
            model.to(device)
            model.eval()
            
            # 2. Reconstruct Dataset Parameters
            detected_mode = "learned" if "learned" in model_path else "hybrid" if "hybrid" in model_path else "coords"
            eval_neighbors = train_opts.get('neighbors', 20)
            eval_knn_strat = train_opts.get('knn_strat', 'None')
            
            dataset = model.problem.make_dataset(
                filename=dataset_path, batch_size=128, num_samples=val_size, 
                neighbors=eval_neighbors, knn_strat=eval_knn_strat,
                node_feature_type=detected_mode, supervised=True, nar=False
            )
            dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
            
            any_success_for_model = False
            
            # 3. Evaluate each strategy
            for width in widths:
                strat_name = "Greedy " if width == 0 else f"BS-{width:<4}"
                all_model_costs = []
                
                for batch in dataloader:
                    nodes, graph = move_to(batch['nodes'], device), move_to(batch['graph'], device)
                    
                    with torch.no_grad():
                        if width == 0:
                            model.set_decode_type("greedy")
                            costs, _, sequences, _ = model(nodes, graph, return_pi=True, return_entropy=True)
                            all_model_costs.extend(costs.cpu().numpy())
                        else:
                            cum_log_p, sequences, raw_costs, ids, batch_size = model.beam_search(
                                nodes, graph, beam_size=width, compress_mask=False, max_calc_batch_size=10000
                            )
                            if sequences is not None:
                                sequences, _ = get_best(
                                    sequences.cpu().numpy(), raw_costs.cpu().numpy(),
                                    ids.cpu().numpy() if ids is not None else None, batch_size
                                )
                                seq_tensor = torch.tensor(np.array(sequences), device=device)
                                true_costs, _ = model.problem.get_costs(nodes, seq_tensor)
                                all_model_costs.extend(true_costs.cpu().numpy())
                            else:
                                all_model_costs.extend([float('inf')] * batch_size)
                                
                # 4. Calculate Gaps and Find Winners
                all_model_costs = np.array(all_model_costs)
                gaps = ((all_model_costs / lkh_costs) - 1.0) * 100
                
                # Threshold set to strictly below -0.01% to ignore floating point rounding ties
                winning_indices = np.where(gaps < -0.01)[0] 
                
                if len(winning_indices) > 0:
                    any_success_for_model = True
                    
                    # Sort indices by how negative the gap is (best wins first)
                    sorted_winners = sorted(winning_indices, key=lambda idx: gaps[idx])
                    
                    # Format strings mapping instance ID to exact gap
                    win_details = [f"Idx {idx} ({gaps[idx]:.4f}%)" for idx in sorted_winners]
                    
                    # Console formatting (truncate if too many to keep terminal clean)
                    console_str = ", ".join(win_details[:5]) + ("..." if len(win_details) > 5 else "")
                    print(f"  [+] {strat_name} : {len(winning_indices):>3} wins | {console_str}")
                    
                    # File formatting (save everything)
                    report_lines.append(f"  [+] {strat_name} : {len(winning_indices)} wins.")
                    report_lines.append(f"      Details: {', '.join(win_details)}")
                else:
                    print(f"  [-] {strat_name} :   0 wins.")
                    report_lines.append(f"  [-] {strat_name} : 0 wins.")

            if not any_success_for_model:
                report_lines.append("  [!] NONE: Did not beat LKH on any instance.")
                
            report_lines.append("-" * 60)

        except Exception as e:
            err_msg = f"  [ERROR] Failed to evaluate {model_name}: {e}"
            print(err_msg)
            report_lines.append(err_msg)
            report_lines.append("-" * 60)

    # 5. Write to File
    with open(output_file, 'w') as f:
        f.write("\n".join(report_lines))
        
    print("\n" + "=" * 80)
    print(f"Batch Evaluation Complete. Full report saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to the validation dataset")
    parser.add_argument("--models", nargs='+', required=True, help="List of model checkpoints. You can use bash wildcards here.")
    parser.add_argument("--val_size", type=int, default=1280, help="Number of validation instances.")
    parser.add_argument("--out_file", type=str, default="reunion/winning_models_report.txt", help="Path to save the text report.")
    opts = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_evaluate_models(opts.models, opts.dataset, opts.val_size, opts.out_file, device)