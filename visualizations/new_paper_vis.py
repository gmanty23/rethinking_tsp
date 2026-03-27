#!/usr/bin/env python
"""
visualize_tours.py
──────────────────
Visualizes and compares tours from 3 sources on the Windy TSP:
  • Original model (OG)
  • Improved model (IMP)
  • LKH baseline

For every instance it shows:
  • Directed tour edges colored green→red by wind effect (tailwind vs headwind)
  • A cyan arrow for the reconstructed wind vector
  • Wind magnitude label
  • Tour length & optimality gap vs LKH

Usage
-----
# Visualize a specific instance by index:
    python visualize_tours.py --instance 7

# Pick a random instance with a fixed seed (for reproducibility):
    python visualize_tours.py --seed 42

# Interactive loop – press Enter to cycle through, Ctrl-C to quit:
    python visualize_tours.py --interactive

# Override paths:
    python visualize_tours.py --instance 0 --dataset path/to/data.pkl ...
"""

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")          # headless – swap for "TkAgg" if interactive display is available
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

# ─── Project imports ──────────────────────────────────────────────────────────
from utils import load_model, move_to

# ─── Default paths ────────────────────────────────────────────────────────────
ORIG_MODEL    = ("outputs/002_exp1_confidence_og_stats/05ConfOgStats_ent0.2_exp1/"
                 "exp1_coords_ent0.2_20251223T185323/epoch-99.pt")
IMP_MODEL     = ("outputs/windy_tsp_20-20/008_exp3_graph_sparsification/"
                 "13Random_Sparsification/"
                 "tsp20_dual_hybrid_ent0.05_random20_20260302T160327/epoch-99.pt")
DATASET       = "data/windy_tsp/windy_tsp20_val.pkl"
LKH_BASELINE  = "results/windy_tsp20_val/windy_tsp20_valn1280-lkh_windy.pkl"


# ══════════════════════════════════════════════════════════════════════════════
#  Wind extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_exact_wind(coords: np.ndarray, cost_matrix: np.ndarray):
    """
    Recover the scaled wind vector (alpha * w) via least squares from the
    asymmetric cost matrix.

    Model:  c_ij = d_ij * exp(-alpha * (w · u_ij))
    Rearranged per edge pair (i,j):
        (alpha*w) · (p_j - p_i) = 0.5 * d_ij * ln(c_ji / c_ij)

    Returns
    -------
    wind_dir : unit direction (2,)
    wind_mag : magnitude of recovered alpha*w  (scalar)
    wind_vec : full scaled vector (2,)
    """
    n = len(coords)
    A, b = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = np.linalg.norm(coords[j] - coords[i])
            if d_ij > 1e-9:
                delta_p = coords[j] - coords[i]
                c_ij    = cost_matrix[i, j]
                c_ji    = cost_matrix[j, i]
                val     = 0.5 * d_ij * np.log((c_ji + 1e-12) / (c_ij + 1e-12))
                A.append(delta_p)
                b.append(val)

    A = np.array(A)
    b = np.array(b)
    wind_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    wind_mag = np.linalg.norm(wind_vec)
    wind_dir = wind_vec / wind_mag if wind_mag > 1e-9 else np.zeros(2)
    return wind_dir, wind_mag, wind_vec


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_instance(dataset_path: str, idx: int):
    """
    Load instance `idx` from the Windy-TSP pkl file.

    Expected dict layout:  {'loc': (N,2), 'wind': (2,), 'alpha': ()}
    Returns (coords, wind_vec, wind_mag):
      • coords   – node positions  (N, 2)   float32
      • wind_vec – raw wind direction  (2,)  (used for arrow display)
      • wind_mag – ||alpha * wind||  (scalar, the effective strength)

    Note: cost_matrix is NOT returned here — it is read directly from the
    dataset batch via DataLoader (batch['cost_matrix']), which is cheaper
    and guaranteed consistent with whatever graph the model sees.
    """
    with open(dataset_path, "rb") as f:
        raw = pickle.load(f)

    inst = raw[idx]

    if isinstance(inst, dict) and "loc" in inst and "wind" in inst:
        coords   = np.array(inst["loc"],  dtype=np.float32)
        wind_vec = np.array(inst["wind"], dtype=np.float64)
        alpha    = float(inst["alpha"])
        wind_mag = float(np.linalg.norm(alpha * wind_vec))
        return coords, wind_vec, wind_mag

    raise KeyError(
        f"Expected keys {{'loc','wind','alpha'}} in instance dict, "
        f"got: {list(inst.keys()) if isinstance(inst, dict) else type(inst)}"
    )


def load_lkh_results(lkh_path: str):
    """Returns list of (cost, tour, duration) after filtering None rows."""
    with open(lkh_path, "rb") as f:
        data = pickle.load(f)
    return [r for r in data if r is not None]


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading & inference
# ══════════════════════════════════════════════════════════════════════════════

def _detect_feature_mode(model_path: str) -> str:
    name = os.path.basename(os.path.dirname(model_path))
    for kw in ("learned", "hybrid", "coords"):
        if kw in name:
            return kw
    print(f"[!] Could not detect feature mode from '{name}', defaulting to 'coords'.")
    return "coords"


def load_model_from_path(model_path: str):
    """Load model + train_opts from a 'epoch-N.pt' checkpoint path."""
    model_dir = os.path.dirname(model_path)
    epoch = int(os.path.basename(model_path).split("-")[1].split(".")[0])
    model, train_opts = load_model(model_dir, epoch=epoch)
    model.eval()
    return model, train_opts


def build_dataset(model, train_opts, model_path: str, dataset_path: str, num_samples: int):
    """Build the dataset for a given model (detects feature type from model dir name)."""
    feat_mode = _detect_feature_mode(model_path)
    neighbors = train_opts.get("neighbors", 20)
    knn_strat = train_opts.get("knn_strat", "None")
    return model.problem.make_dataset(
        filename=dataset_path,
        batch_size=1,
        num_samples=num_samples,
        neighbors=neighbors,
        knn_strat=knn_strat,
        node_feature_type=feat_mode,
        supervised=True,
        nar=False,
    )


def run_greedy_on_instance(model, dataset, idx: int, device):
    """
    Run greedy decoding on instance `idx` from a pre-built dataset.
    Returns (cost: float, tour: list[int])
    """
    sample = dataset[idx]
    nodes  = sample["nodes"].unsqueeze(0).to(device)
    graph  = sample["graph"].unsqueeze(0).to(device)

    model.set_decode_type("greedy")
    with torch.no_grad():
        costs, ll, sequences, _ = model(nodes, graph, return_pi=True, return_entropy=True)

    tour = sequences[0].cpu().tolist()
    cost = costs[0].item()
    return cost, tour


# ══════════════════════════════════════════════════════════════════════════════
#  Per-edge wind value
# ══════════════════════════════════════════════════════════════════════════════

def edge_wind_benefit(i: int, j: int,
                      coords: np.ndarray,
                      cost_matrix: np.ndarray) -> float:
    """
    Returns a signed scalar for directed edge i → j:
      > 0  →  tailwind  (edge is cheaper than Euclidean distance, GREEN)
      < 0  →  headwind  (edge is more expensive than Euclidean,  RED)

    Value = log(d_ij / c_ij)
    """
    d = np.linalg.norm(coords[j] - coords[i])
    if d < 1e-9:
        return 0.0
    c = cost_matrix[i, j] + 1e-12
    return float(np.log(d / c))


# ══════════════════════════════════════════════════════════════════════════════
#  Drawing
# ══════════════════════════════════════════════════════════════════════════════

_DARK_BG   = "#0f0f1a"
_PANEL_BG  = "#16213e"
_NODE_CLR  = "#4fc3f7"
_START_CLR = "#ffd54f"
_WIND_CLR  = "#00e5ff"
_TEXT_CLR  = "#e0e0e0"
_CMAP      = "RdYlGn"


def draw_tour_panel(ax, coords, cost_matrix, tour, title, wind_dir, wind_mag):
    """Draw one tour subplot with wind-heatmap edges and wind arrow."""
    n        = len(tour)
    ax.set_facecolor(_PANEL_BG)

    # ── Compute per-edge wind benefit ───────────────────────────────────────
    benefits = [edge_wind_benefit(tour[k], tour[(k + 1) % n], coords, cost_matrix)
                for k in range(n)]

    abs_max = max(abs(v) for v in benefits) if benefits else 1.0
    abs_max = abs_max if abs_max > 1e-9 else 1.0
    norm     = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    cmap     = cm.get_cmap(_CMAP)

    # ── Tour edges as directed arrows ───────────────────────────────────────
    for k in range(n):
        i, j   = tour[k], tour[(k + 1) % n]
        x0, y0 = coords[i]
        x1, y1 = coords[j]
        color  = cmap(norm(benefits[k]))

        # Slightly shorten arrow so the head is visible
        dx, dy  = x1 - x0, y1 - y0
        length  = np.hypot(dx, dy)
        shrink  = min(0.04, length * 0.18)
        ex      = dx / (length + 1e-12) * shrink
        ey      = dy / (length + 1e-12) * shrink

        ax.annotate(
            "",
            xy       = (x1 - ex, y1 - ey),
            xytext   = (x0 + ex, y0 + ey),
            arrowprops=dict(
                arrowstyle    = "-|>",
                color         = color,
                lw            = 2.0,
                mutation_scale= 13,
            ),
            zorder=3,
        )

    # ── Nodes ───────────────────────────────────────────────────────────────
    ax.scatter(coords[:, 0], coords[:, 1],
               s=55, zorder=5, color=_NODE_CLR,
               edgecolors="white", linewidths=0.7)

    # Highlight depot / start node
    ax.scatter(coords[tour[0], 0], coords[tour[0], 1],
               s=130, zorder=6, color=_START_CLR,
               edgecolors="black", linewidths=1.0, marker="*")

    # Node index labels
    for i, (x, y) in enumerate(coords):
        ax.text(x, y + 0.032, str(i),
                ha="center", va="bottom",
                fontsize=6, color=_TEXT_CLR, zorder=7)

    # ── Wind arrow ──────────────────────────────────────────────────────────
    cx, cy = coords.mean(axis=0)

    # Scale arrow to ~20 % of the unit square, clamped
    scale = 0.18
    tail_x = cx - wind_dir[0] * scale * 0.35
    tail_y = cy - wind_dir[1] * scale * 0.35
    head_x = cx + wind_dir[0] * scale * 0.65
    head_y = cy + wind_dir[1] * scale * 0.65

    ax.annotate(
        "",
        xy       = (head_x, head_y),
        xytext   = (tail_x, tail_y),
        arrowprops=dict(
            arrowstyle    = "fancy,head_width=0.4,head_length=0.3",
            color         = _WIND_CLR,
            lw            = 2.0,
            mutation_scale= 18,
        ),
        zorder=10,
    )

    # Wind magnitude label beside the arrow head
    label_x = head_x + 0.04 * (1 if head_x < 0.75 else -1)
    label_y = head_y + 0.04 * (1 if head_y < 0.75 else -1)
    ax.text(label_x, label_y, f"|α·w|={wind_mag:.3f}",
            color=_WIND_CLR, fontsize=7.5, fontweight="bold", zorder=11)

    # ── Axes styling ────────────────────────────────────────────────────────
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.06)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, color=_TEXT_CLR, fontsize=9.5, fontweight="bold", pad=6)


# ══════════════════════════════════════════════════════════════════════════════
#  Main visualisation routine
# ══════════════════════════════════════════════════════════════════════════════

def visualize_instance(idx: int, args, device,
                       orig_model=None, orig_opts=None,
                       imp_model=None,  imp_opts=None,
                       lkh_results=None):
    """
    Build and save the 3-panel comparison figure for instance `idx`.

    Pre-loaded model objects and LKH results can be passed in to avoid
    reloading across multiple calls (interactive mode).
    """
    print(f"\n[*] ─── Instance {idx} ────────────────────────────────────────")

    # 1. Wind params from raw pkl (coords + wind direction/magnitude)
    coords, wind_vec, wind_mag = load_raw_instance(args.dataset, idx)
    wn       = float(np.linalg.norm(wind_vec))
    wind_dir = wind_vec / wn if wn > 1e-9 else np.zeros(2)
    print(f"    Wind dir=({wind_dir[0]:+.3f}, {wind_dir[1]:+.3f})  "
          f"|α·w|={wind_mag:.4f}")

    # 2. Build datasets for both models (only up to idx+1 for speed)
    num_samples = idx + 1
    orig_dataset = build_dataset(orig_model, orig_opts, args.orig_model,
                                 args.dataset, num_samples)
    imp_dataset  = build_dataset(imp_model,  imp_opts,  args.imp_model,
                                 args.dataset, num_samples)

    # 3. Cost matrix — pull directly from the dataset batch (no recomputation)
    orig_batch   = next(iter(torch.utils.data.DataLoader(
                       torch.utils.data.Subset(orig_dataset, [idx]), batch_size=1)))
    cost_matrix  = orig_batch["cost_matrix"][0].cpu().numpy()

    # 4. LKH
    if lkh_results is None:
        lkh_results = load_lkh_results(args.lkh_baseline)
    lkh_cost = float(lkh_results[idx][0])
    lkh_tour = list(lkh_results[idx][1])
    if min(lkh_tour) == 1:           # normalise 1-based → 0-based if needed
        lkh_tour = [t - 1 for t in lkh_tour]
    print(f"    LKH   cost={lkh_cost:.4f}")

    # 5. OG model inference
    orig_cost, orig_tour = run_greedy_on_instance(orig_model, orig_dataset, idx, device)
    orig_gap = ((orig_cost / lkh_cost) - 1) * 100
    print(f"    OG    cost={orig_cost:.4f}  gap={orig_gap:+.2f}%")

    # 6. Improved model inference
    imp_cost, imp_tour = run_greedy_on_instance(imp_model, imp_dataset, idx, device)
    imp_gap = ((imp_cost / lkh_cost) - 1) * 100
    print(f"    IMP   cost={imp_cost:.4f}  gap={imp_gap:+.2f}%")

    # 6. Build figure
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    fig.patch.set_facecolor(_DARK_BG)

    def make_title(name, cost, gap=None):
        if gap is not None:
            marker = "▲" if gap > 0 else "▼" if gap < -0.01 else "≈"
            return f"{name}\nLen: {cost:.4f}   {marker} Gap: {gap:+.2f}%"
        return f"{name}\nLen: {cost:.4f}   (LKH reference)"

    draw_tour_panel(axes[0], coords, cost_matrix, orig_tour,
                    make_title("Original Model", orig_cost, orig_gap),
                    wind_dir, wind_mag)

    draw_tour_panel(axes[1], coords, cost_matrix, imp_tour,
                    make_title("Improved Model", imp_cost, imp_gap),
                    wind_dir, wind_mag)

    draw_tour_panel(axes[2], coords, cost_matrix, lkh_tour,
                    make_title("LKH Baseline", lkh_cost),
                    wind_dir, wind_mag)

    # ── Shared colourbar ────────────────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=_CMAP,
                           norm=mcolors.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(),
                        orientation="horizontal",
                        fraction=0.028, pad=0.06, aspect=45)
    cbar.set_label(
        "Edge wind effect   ◀  headwind (red)  ·  neutral  ·  tailwind (green)  ▶",
        color=_TEXT_CLR, fontsize=8.5)
    cbar.ax.xaxis.set_tick_params(color=_TEXT_CLR)
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color=_TEXT_CLR)
    cbar.outline.set_edgecolor(_TEXT_CLR)

    # ── Super-title ─────────────────────────────────────────────────────────
    fig.suptitle(
        f"Windy TSP — Instance #{idx}     "
        f"Wind dir ({wind_dir[0]:+.3f}, {wind_dir[1]:+.3f})   |α·w| = {wind_mag:.3f}     "
        f"★ = tour start",
        color=_TEXT_CLR, fontsize=11, fontweight="bold", y=1.02,
    )

    plt.tight_layout()

    out_path = args.out.replace(".png", f"_inst{idx:04d}.png") if args.interactive else args.out
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Saved → {out_path}")

    return orig_model, orig_opts, imp_model, imp_opts, lkh_results


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare Windy-TSP tours: OG model vs Improved model vs LKH",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Instance selection ──────────────────────────────────────────────────
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--instance", "-i", type=int, default=None,
                     help="Validation-set index to visualize (0-based).")
    sel.add_argument("--seed", "-s", type=int, default=None,
                     help="Randomly pick an instance using this seed.")
    sel.add_argument("--interactive", action="store_true",
                     help="Loop: prompt for a new index after each render.")

    # ── Paths ───────────────────────────────────────────────────────────────
    p.add_argument("--orig_model",   default=ORIG_MODEL)
    p.add_argument("--imp_model",    default=IMP_MODEL)
    p.add_argument("--dataset",      default=DATASET)
    p.add_argument("--lkh_baseline", default=LKH_BASELINE)
    p.add_argument("--out", default="tour_comparison.png",
                   help="Output PNG path.  In interactive mode '_instNNNN' is appended.")

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--val_size", type=int, default=1280,
                   help="Total number of instances in the val set.")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda:0"
                          if torch.cuda.is_available() and not args.no_cuda
                          else "cpu")
    print(f"[*] Device: {device}")

    # ── Determine initial index ─────────────────────────────────────────────
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
        idx = int(rng.integers(0, args.val_size))
        print(f"[*] Seed {args.seed} → instance {idx}")
    elif args.instance is not None:
        idx = args.instance
    else:
        idx = 0   # default

    # ── Pre-load models & LKH once (reused across iterations) ──────────────
    print("[*] Loading Original model …")
    orig_model, orig_opts = load_model_from_path(args.orig_model)
    orig_model.to(device)

    print("[*] Loading Improved model …")
    imp_model, imp_opts = load_model_from_path(args.imp_model)
    imp_model.to(device)

    print("[*] Loading LKH baseline …")
    lkh_results = load_lkh_results(args.lkh_baseline)

    # ── Non-interactive single render ───────────────────────────────────────
    if not args.interactive:
        visualize_instance(
            idx, args, device,
            orig_model=orig_model, orig_opts=orig_opts,
            imp_model=imp_model,   imp_opts=imp_opts,
            lkh_results=lkh_results,
        )
        return

    # ── Interactive loop ────────────────────────────────────────────────────
    print("\n[*] Interactive mode — type an index and press Enter (Ctrl-C to quit).")
    while True:
        try:
            user_in = input(f"\n  Instance index [0-{args.val_size - 1}]"
                            f"  (default={idx}, 'r' for random): ").strip()
            if user_in == "":
                pass                              # keep current idx
            elif user_in.lower() == "r":
                idx = int(np.random.randint(0, args.val_size))
                print(f"  → Random pick: {idx}")
            else:
                new_idx = int(user_in)
                if 0 <= new_idx < args.val_size:
                    idx = new_idx
                else:
                    print(f"  [!] Out of range, must be 0–{args.val_size - 1}.")
                    continue

            visualize_instance(
                idx, args, device,
                orig_model=orig_model, orig_opts=orig_opts,
                imp_model=imp_model,   imp_opts=imp_opts,
                lkh_results=lkh_results,
            )
        except KeyboardInterrupt:
            print("\n[*] Exiting.")
            break
        except ValueError:
            print("  [!] Please enter a valid integer.")


if __name__ == "__main__":
    main()