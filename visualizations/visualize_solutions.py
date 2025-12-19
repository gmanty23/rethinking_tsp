import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import glob
import subprocess
from problems.tsp.problem_tsp import WindyTSP

# ==========================================
# 0. MONKEY PATCH for torch.load
# ==========================================
import builtins
original_load = torch.load

def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = safe_load

from utils.functions import load_model

# ==========================================
# 1. CONFIGURATION
# ==========================================
LKH_PATH = "./LKH" 

ALPHA = 5.0
MAX_WIND = 0.5
NUM_NODES = 20
SEED = 12345 
BASE_DIR = "outputs/windy_tsp_20-20/exp1_TSP20_windstrong"

MODEL_A_KEY = "coords"
MODEL_B_KEY = "learned"
MODEL_C_KEY = "hybrid"

# Animation Settings
INTERVAL = 500
SOLID_COLOR = 'black'
CMAP_NAME = 'RdYlGn_r'

# ==========================================
# 2. HELPER FUNCTIONS (New Rotation Logic)
# ==========================================
def rotate_to_depot(tour):
    """
    Rotates the tour array so that Node 0 (Depot) is always the first element.
    Ex: [5, 2, 0, 4] -> [0, 4, 5, 2]
    """
    if 0 not in tour:
        return tour # Safety check
    idx = np.where(tour == 0)[0][0]
    return np.roll(tour, -idx)

# ==========================================
# 3. LKH SOLVER WRAPPER
# ==========================================
def solve_lkh(cost_matrix):
    if not os.path.exists(LKH_PATH):
        print(f"Error: LKH executable not found at {LKH_PATH}")
        return np.arange(len(cost_matrix))

    scale_factor = 10000
    int_matrix = (cost_matrix * scale_factor).astype(int)
    num_nodes = len(cost_matrix)
    
    filename = "temp_problem.atsp"
    par_filename = "temp_problem.par"
    tour_filename = "temp_problem.tour"
    
    with open(filename, 'w') as f:
        f.write("NAME: temp\n")
        f.write("TYPE: ATSP\n")
        f.write(f"DIMENSION: {num_nodes}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for row in int_matrix:
            f.write(" ".join(map(str, row)) + "\n")
        f.write("EOF\n")

    with open(par_filename, 'w') as f:
        f.write(f"PROBLEM_FILE = {filename}\n")
        f.write(f"TOUR_FILE = {tour_filename}\n")
        f.write("RUNS = 1\n")
        f.write("TRACE_LEVEL = 0\n") 

    try:
        subprocess.run([LKH_PATH, par_filename], check=True, stdout=subprocess.DEVNULL)
    except Exception as e:
        print(f"LKH Run Failed: {e}")
        return np.arange(num_nodes)

    tour = []
    with open(tour_filename, 'r') as f:
        lines = f.readlines()
        reading = False
        for line in lines:
            if "TOUR_SECTION" in line:
                reading = True
                continue
            if "EOF" in line or "-1" in line:
                reading = False
                break
            if reading:
                node_idx = int(line.strip()) - 1
                tour.append(node_idx)
    
    for f in [filename, par_filename, tour_filename]:
        if os.path.exists(f):
            os.remove(f)
            
    return np.array(tour)

# ==========================================
# 4. DATA & MODEL LOADING
# ==========================================
def generate_chaotic_instance(seed=1234):
    np.random.seed(seed)
    loc = np.random.rand(NUM_NODES, 2)
    
    wind_angle = np.random.uniform(0, 2 * np.pi)
    wind_mag = np.random.uniform(0.5, MAX_WIND)
    wind = np.array([wind_mag * np.cos(wind_angle), wind_mag * np.sin(wind_angle)])
    alpha = ALPHA
    
    diff = loc[None, :, :] - loc[:, None, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = diff / dists[:, :, None]
    u[np.isnan(u)] = 0
    wind_proj = np.dot(u, wind)
    costs = dists * np.exp(-1.0 * alpha * wind_proj)
    np.fill_diagonal(costs, 0)
    
    stat_out = np.sum(costs, axis=1, keepdims=True) / (NUM_NODES - 1)
    stat_in = np.sum(costs, axis=0, keepdims=True).T / (NUM_NODES - 1)
    
    wind_repeated = np.tile(wind, (NUM_NODES, 1))
    alpha_repeated = np.full((NUM_NODES, 1), alpha)
    features = np.concatenate([loc, wind_repeated, alpha_repeated, stat_out, stat_in], axis=-1)
    
    tensor_input = torch.FloatTensor(features).unsqueeze(0)
    mask_graph = torch.zeros((1, NUM_NODES, NUM_NODES)).bool()
    
    return tensor_input, mask_graph, {'loc': loc, 'wind': wind, 'costs': costs}

def find_latest_model(keyword):
    search_path = os.path.join(BASE_DIR, f"*{keyword}*")
    candidates = glob.glob(search_path)
    if not candidates:
        raise FileNotFoundError(f"No folder with '{keyword}' in {BASE_DIR}")
    return max(candidates, key=os.path.getmtime)

def solve_tour_model(model, nodes, graph):
    model.eval()
    model.set_decode_type("greedy")
    with torch.no_grad():
        _, _, pi = model(nodes, graph, return_pi=True)
    return pi.cpu().numpy()[0]

# ==========================================
# 5. ANIMATION SETUP
# ==========================================
def run_animation():
    print("--- Setting up 4-Panel Animation (Rotated to Depot) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Generate Instance
    nodes_t, graph_t, meta = generate_chaotic_instance(seed=SEED)
    
    # 2. Load Models
    print("Loading Neural Models...")
    path_a = find_latest_model(MODEL_A_KEY)
    path_b = find_latest_model(MODEL_B_KEY)
    path_c = find_latest_model(MODEL_C_KEY)
    
    model_a, _ = load_model(path_a)
    model_b, _ = load_model(path_b)
    model_c, _ = load_model(path_c)
    
    # 3. Solve Neural Tours
    tour_a = solve_tour_model(model_a.to(device), nodes_t.to(device), graph_t.to(device))
    tour_b = solve_tour_model(model_b.to(device), nodes_t.to(device), graph_t.to(device))
    tour_c = solve_tour_model(model_c.to(device), nodes_t.to(device), graph_t.to(device))
    
    # 4. Solve LKH Tour
    print(f"Solving Optimal Baseline with {LKH_PATH}...")
    tour_d = solve_lkh(meta['costs'])
    
    # --- CHANGED: Force all tours to start at 0 ---
    tour_a = rotate_to_depot(tour_a)
    tour_b = rotate_to_depot(tour_b)
    tour_c = rotate_to_depot(tour_c)
    tour_d = rotate_to_depot(tour_d)

    # Close loops (Append 0 to the end)
    tour_a = np.append(tour_a, 0)
    tour_b = np.append(tour_b, 0)
    tour_c = np.append(tour_c, 0)
    tour_d = np.append(tour_d, 0)
    
    loc = meta['loc']
    matrix = meta['costs']
    
    # 5. Helper for Factors
    def get_edges_and_costs(tour):
        edges = []
        raw_costs = []
        factors = []  
        cumulative = [0.0]
        for i in range(len(tour)-1):
            u, v = tour[i], tour[i+1]
            cost = matrix[u, v]
            dist = np.linalg.norm(loc[u] - loc[v])
            factor = cost / (dist + 1e-9)
            edges.append((u, v))
            raw_costs.append(cost)
            factors.append(factor)
            cumulative.append(cumulative[-1] + cost)
        return edges, raw_costs, factors, cumulative

    edges_a, _, factors_a, total_a = get_edges_and_costs(tour_a)
    edges_b, _, factors_b, total_b = get_edges_and_costs(tour_b)
    edges_c, _, factors_c, total_c = get_edges_and_costs(tour_c)
    edges_d, _, factors_d, total_d = get_edges_and_costs(tour_d)
    
    norm = mcolors.Normalize(vmin=0.5, vmax=2.5)
    cmap = plt.get_cmap(CMAP_NAME)

    # 6. Plotting (4 Columns)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 7.5))
    
    def setup_ax(ax, title, final_cost):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.axis('off')
        ax.scatter(loc[:,0], loc[:,1], c='gray', s=50, zorder=5)
        ax.scatter(loc[tour_a[0],0], loc[tour_a[0],1], c='yellow', edgecolors='black', s=150, zorder=6)
        center = np.mean(loc, axis=0)
        w = meta['wind']
        ax.arrow(center[0]-0.2, center[1], w[0]*0.3, w[1]*0.3, width=0.03, color='lightblue', alpha=0.5)
        
        # Final Cost Display
        ax.text(0.5, -0.1, f"Final Cost: {final_cost:.2f}", 
                transform=ax.transAxes, 
                ha='center', va='top', 
                fontsize=16, fontweight='bold', 
                color='black',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))

    setup_ax(ax1, f"Coords (Geometry)", total_a[-1])
    setup_ax(ax2, f"Learned (Physics)", total_b[-1])
    setup_ax(ax3, f"Hybrid (Proposed)", total_c[-1])
    setup_ax(ax4, f"LKH-3 (Optimal)", total_d[-1])

    lines_a, lines_b, lines_c, lines_d = [], [], [], []
    
    def draw_dotted(ax, edges, factors, line_store):
        for (u, v), factor in zip(edges, factors):
            color = cmap(norm(factor))
            line, = ax.plot([loc[u,0], loc[v,0]], [loc[u,1], loc[v,1]], 
                            linestyle=':', linewidth=3, color=color, alpha=0.7)
            line_store.append(line)
    
    draw_dotted(ax1, edges_a, factors_a, lines_a)
    draw_dotted(ax2, edges_b, factors_b, lines_b)
    draw_dotted(ax3, edges_c, factors_c, lines_c)
    draw_dotted(ax4, edges_d, factors_d, lines_d)
    
    txt_a = ax1.text(0.05, 0.95, "Running: 0.00", transform=ax1.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    txt_b = ax2.text(0.05, 0.95, "Running: 0.00", transform=ax2.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    txt_c = ax3.text(0.05, 0.95, "Running: 0.00", transform=ax3.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    txt_d = ax4.text(0.05, 0.95, "Running: 0.00", transform=ax4.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2, ax3, ax4], orientation='horizontal', fraction=0.05, pad=0.2)
    cbar.set_label('Wind Efficiency (Green=Tailwind, Red=Headwind)')

    def update(frame):
        for lines, total, txt, frames_len in [
            (lines_a, total_a, txt_a, len(edges_a)),
            (lines_b, total_b, txt_b, len(edges_b)),
            (lines_c, total_c, txt_c, len(edges_c)),
            (lines_d, total_d, txt_d, len(edges_d))
        ]:
            if frame < frames_len:
                lines[frame].set_linestyle('-')
                lines[frame].set_linewidth(4)
                lines[frame].set_color(SOLID_COLOR)
                lines[frame].set_alpha(1.0)
                txt.set_text(f"Running: {total[frame+1]:.2f}")
                
        return lines_a + lines_b + lines_c + lines_d + [txt_a, txt_b, txt_c, txt_d]

    num_frames = max(len(edges_a), len(edges_b), len(edges_c), len(edges_d))
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=INTERVAL, blit=True, repeat=True)
    
    save_path = "animation_4models_lkh_rotated.gif"
    print(f"Saving animation to {save_path}...")
    ani.save(save_path, writer='pillow', fps=2)
    print("Done!")

if __name__ == "__main__":
    run_animation()