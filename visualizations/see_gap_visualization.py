import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from torch.utils.data import DataLoader
from utils.functions import load_model, move_to, get_best


def get_tour_batch(model, nodes, graph, WIDTH=0):
    """
    Runs inference on a full batch and returns (sequences, costs).
    - sequences: LongTensor of shape (batch,)
    - costs:     FloatTensor of shape (batch,) — always computed via
                 model.problem.get_costs so the windy ruler is used.
    """
    with torch.no_grad():
        # Cache priming (helps models that use lazy graph init)
        try:
            if hasattr(model, 'embedder'):
                h = model._init_embed(nodes)
                _ = model.embedder(h, graph.clone())
        except Exception:
            pass

        if WIDTH == 0 or WIDTH == 1:
            # --- GREEDY ---
            model.set_decode_type("greedy")
            # model() returns (costs, ll, sequences, entropy)
            # 'costs' here are already from get_costs internally, so they are correct
            costs, _, sequences, _ = model(nodes, graph, return_pi=True, return_entropy=True)
            return sequences, costs                        # ✅ return full batch

        else:
            # --- BEAM SEARCH ---
            model.set_decode_type("greedy")               # ✅ set BEFORE beam search

            cum_log_p, sequences, raw_costs, ids, batch_size = model.beam_search(
                nodes, graph, beam_size=WIDTH,
                compress_mask=False, max_calc_batch_size=10000
            )

            # ✅ convert to numpy BEFORE calling get_best
            sequences, _ = get_best(
                sequences.cpu().numpy(),
                raw_costs.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )

            # ✅ recompute real windy costs (raw_costs are internal log-prob scores)
            seq_tensor = torch.tensor(
                np.array(sequences), dtype=torch.long, device=nodes.device
            )
            costs, _ = model.problem.get_costs(nodes, seq_tensor)
            return seq_tensor, costs                       # ✅ return full batch


def get_feature_type(model_path):
    """Hardcoded feature detection exactly as implemented in eval_exp1.py"""
    model_name = os.path.basename(os.path.dirname(model_path))
    if "learned" in model_name:
        return "learned"
    elif "hybrid" in model_name:
        return "hybrid"
    return "coords"


def evaluate_dataset_and_plot(orig_model_path, imp_model_path, dataset_path, lkh_path, batch_size=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    with open(lkh_path, 'rb') as f:
        lkh_data = list(pickle.load(f))

    valid_lkh_data = [row for row in lkh_data if row is not None]
    all_lkh_costs  = np.array([row[0] for row in valid_lkh_data])
    total_instances = len(all_lkh_costs)

    orig_model, orig_args = load_model(orig_model_path)
    orig_model.to(device).eval()
    orig_mode = get_feature_type(orig_model_path)

    imp_model, imp_args = load_model(imp_model_path)
    imp_model.to(device).eval()
    imp_mode = get_feature_type(imp_model_path)

    orig_dataset = orig_model.problem.make_dataset(
        filename=dataset_path, num_samples=total_instances, offset=0,
        batch_size=batch_size,
        neighbors=orig_args.get('neighbors', 20),
        knn_strat=orig_args.get('knn_strat', 'None'),
        node_feature_type=orig_mode, supervised=True, nar=False
    )
    orig_dataloader = DataLoader(orig_dataset, batch_size=batch_size, shuffle=False)

    imp_dataset = imp_model.problem.make_dataset(
        filename=dataset_path, num_samples=total_instances, offset=0,
        batch_size=batch_size,
        neighbors=imp_args.get('neighbors', 20),
        knn_strat=imp_args.get('knn_strat', 'None'),
        node_feature_type=imp_mode, supervised=True, nar=False
    )
    imp_dataloader = DataLoader(imp_dataset, batch_size=batch_size, shuffle=False)

    all_orig_costs = []
    all_imp_costs  = []

    current_idx = 0
    for orig_batch, imp_batch in zip(orig_dataloader, imp_dataloader):
        orig_nodes = move_to(orig_batch['nodes'], device)
        orig_graph = move_to(orig_batch['graph'], device)
        imp_nodes  = move_to(imp_batch['nodes'],  device)
        imp_graph  = move_to(imp_batch['graph'],  device)

        # ✅ get_tour_batch now returns (sequences, costs) for the full batch
        _, orig_costs = get_tour_batch(orig_model, orig_nodes, orig_graph, WIDTH=0)
        _, imp_costs  = get_tour_batch(imp_model,  imp_nodes,  imp_graph,  WIDTH=0)

        all_orig_costs.extend(orig_costs.cpu().numpy().tolist())
        all_imp_costs.extend(imp_costs.cpu().numpy().tolist())

        current_idx += orig_nodes.shape[0]
        print(f"    Processed {current_idx}/{total_instances} instances...")

    all_orig_costs = np.array(all_orig_costs)
    all_imp_costs  = np.array(all_imp_costs)

    # Mean-of-Ratios gap, consistent with eval_exp1.py
    orig_gaps = ((all_orig_costs / all_lkh_costs) - 1.0) * 100
    imp_gaps  = ((all_imp_costs  / all_lkh_costs) - 1.0) * 100

    orig_mean, orig_std = np.mean(orig_gaps), np.std(orig_gaps)
    imp_mean,  imp_std  = np.mean(imp_gaps),  np.std(imp_gaps)

    print("\n" + "="*50)
    print("DATASET AGGREGATE VERIFICATION REPORT")
    print("="*50)
    print(f"Original Model Mean Gap: {orig_mean:.2f}% (Std: {orig_std:.2f}%)")
    print(f"Improved Model Mean Gap: {imp_mean:.2f}% (Std: {imp_std:.2f}%)")
    print("="*50 + "\n")

    # --- Bar Chart ---
    print("[*] Generating bar chart...")
    labels = ['Original Model', 'Improved Model']
    means  = [orig_mean, imp_mean]
    stds   = [orig_std,  imp_std]
    colors = ['#EF5350', '#66BB6A']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, means, yerr=stds, capsize=8, color=colors,
                   alpha=0.8, edgecolor='black', zorder=3)

    plt.ylabel('Optimality Gap (%)', fontsize=12, fontweight='bold')
    plt.title(f'Optimality Gap Comparison over {total_instances} Instances',
              fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    for bar, std in zip(bars, stds):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,
                 yval + std + (max(means) * 0.05),
                 f'{yval:.2f}%', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    os.makedirs("visualizations/aggregate_stats", exist_ok=True)
    out_img = "visualizations/aggregate_stats/PAPER_dataset_gap_comparison.png"
    plt.tight_layout()
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"[*] Chart successfully saved to '{out_img}'")


if __name__ == "__main__":
    ORIG_MODEL    = "outputs/002_exp1_confidence_og_stats/05ConfOgStats_ent0.2_exp1/exp1_coords_ent0.2_20251223T185323/epoch-99.pt"
    IMP_MODEL     = "outputs/windy_tsp_20-20/008_exp3_graph_sparsification/13Random_Sparsification/tsp20_dual_hybrid_ent0.05_random20_20260302T160327/epoch-99.pt"
    DATASET       = "data/windy_tsp/windy_tsp20_val.pkl"
    LKH_BASELINE  = "results/windy_tsp20_val/windy_tsp20_valn1280-lkh_windy.pkl"

    evaluate_dataset_and_plot(ORIG_MODEL, IMP_MODEL, DATASET, LKH_BASELINE, batch_size=128)