import os
import glob
import uuid
import stat
import torch
import numpy as np
import subprocess
import builtins

# ==========================================
# 0. HEADLESS MODE & TORCH PATCH
# ==========================================
import matplotlib
# Force Agg backend so this runs on servers without screens/X11
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm

# Monkey patch for torch.load to handle 'weights_only' default in newer versions
original_load = torch.load

def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = safe_load

# Local Project Imports
try:
    from problems.tsp.problem_tsp import WindyTSP
    from utils.functions import load_model
except ImportError as e:
    print("Error importing local modules. Ensure you are running from the project root.")
    print(f"Details: {e}")
    exit(1)

# ==========================================
# 1. CONFIGURATION
# ==========================================
LKH_PATH = "./LKH" 

# Physics Parameters
ALPHA = 3.0
MAX_WIND = 1.0
NUM_NODES = 20
SEED = 12345 

# Path Settings
BASE_DIR = "outputs/windy_tsp_20-20/dims_4/exp1_TSP20_windstrong"
MODEL_A_KEY = "coords"   # Baseline (Euclidean training)
MODEL_B_KEY = "learned"  # Physics-aware training
MODEL_C_KEY = "hybrid"   # Proposed method

# Animation Settings
INTERVAL = 500
SOLID_COLOR = 'black'
CMAP_NAME = 'RdYlGn_r'  # Red (bad) to Green (good)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def rotate_to_depot(tour):
    """
    Rotates tour so Node 0 is first.
    Safety: returns tour as-is if empty or 0 is missing.
    """
    if len(tour) == 0:
        return np.array([0])
    if 0 not in tour:
        return tour
    idx = np.where(tour == 0)[0][0]
    return np.roll(tour, -idx)

def find_latest_model(keyword):
    """Finds the most recently modified model folder containing 'keyword'."""
    search_path = os.path.join(BASE_DIR, f"*{keyword}*")
    candidates = glob.glob(search_path)
    if not candidates:
        raise FileNotFoundError(f"No folder with '{keyword}' in {BASE_DIR}")
    return max(candidates, key=os.path.getmtime)

# ==========================================
# 3. LKH SOLVER WRAPPER (ROBUST)
# ==========================================
def solve_lkh(cost_matrix):
    """
    Solves ATSP using LKH-3.
    Uses unique filenames to allow parallel execution.
    """
    if not os.path.exists(LKH_PATH):
        print(f"Error: LKH executable not found at {LKH_PATH}")
        return np.arange(len(cost_matrix))

    # Ensure LKH is executable
    st = os.stat(LKH_PATH)
    os.chmod(LKH_PATH, st.st_mode | stat.S_IEXEC)

    scale_factor = 10000
    int_matrix = (cost_matrix * scale_factor).astype(int)
    num_nodes = len(cost_matrix)
    
    # Generate unique ID for this run
    uid = str(uuid.uuid4())[:8]
    filename = f"temp_{uid}.atsp"
    par_filename = f"temp_{uid}.par"
    tour_filename = f"temp_{uid}.tour"
    
    # Write Problem File
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

    # Write Parameters File
    with open(par_filename, 'w') as f:
        f.write(f"PROBLEM_FILE = {filename}\n")
        f.write(f"TOUR_FILE = {tour_filename}\n")
        f.write("RUNS = 1\n")
        f.write("TRACE_LEVEL = 0\n") 

    tour = []
    try:
        subprocess.run([LKH_PATH, par_filename], check=True, stdout=subprocess.DEVNULL)
        
        # Read Result
        if os.path.exists(tour_filename):
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
                        tour.append(int(line.strip()) - 1)
    except Exception as e:
        print(f"LKH Run Failed: {e}")
    finally:
        # Cleanup
        for f in [filename, par_filename, tour_filename]:
            if os.path.exists(f):
                os.remove(f)
            
    return np.array(tour) if tour else np.arange(num_nodes)

# ==========================================
# 4. DATA & MODEL LOADING
# ==========================================
def generate_chaotic_instance(seed=1234):
    np.random.seed(seed)
    loc = np.random.rand(NUM_NODES, 2)
    
    # Generate Wind
    wind_angle = np.random.uniform(0, 2 * np.pi)
    wind_mag = np.random.uniform(0.5, MAX_WIND)
    wind = np.array([wind_mag * np.cos(wind_angle), wind_mag * np.sin(wind_angle)])
    
    # Calculate Costs
    diff = loc[None, :, :] - loc[:, None, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, 1.0) # Avoid div/0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        u = diff / dists[:, :, None]
    u[np.isnan(u)] = 0
    
    wind_proj = np.dot(u, wind)
    # Physics Model: Cost = Dist * exp(-alpha * wind_alignment)
    costs = dists * np.exp(-1.0 * ALPHA * wind_proj)
    np.fill_diagonal(costs, 0)
    
    # Features for Neural Net
    stat_out = np.sum(costs, axis=1, keepdims=True) / (NUM_NODES - 1)
    stat_in = np.sum(costs, axis=0, keepdims=True).T / (NUM_NODES - 1)
    
    wind_repeated = np.tile(wind, (NUM_NODES, 1))
    alpha_repeated = np.full((NUM_NODES, 1), ALPHA)
    features = np.concatenate([loc, wind_repeated, alpha_repeated, stat_out, stat_in], axis=-1)
    
    tensor_input = torch.FloatTensor(features).unsqueeze(0)
    mask_graph = torch.zeros((1, NUM_NODES, NUM_NODES)).bool()
    
    return tensor_input, mask_graph, {'loc': loc, 'wind': wind, 'costs': costs}

def solve_tour_model(model, nodes, graph):
    model.eval()
    model.set_decode_type("greedy")
    with torch.no_grad():
        _, _, pi = model(nodes, graph, return_pi=True)
    return pi.cpu().numpy()[0]

# ==========================================
# 5. ANIMATION EXECUTION
# ==========================================
def run_animation():
    print("--- Setting up 4-Panel Animation (Rotated to Depot) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Generate Instance
    nodes_t, graph_t, meta = generate_chaotic_instance(seed=SEED)
    
    # 2. Load Models
    print("Loading Neural Models...")
    try:
        path_a = find_latest_model(MODEL_A_KEY)
        path_b = find_latest_model(MODEL_B_KEY)
        path_c = find_latest_model(MODEL_C_KEY)
        
        model_a, _ = load_model(path_a)
        model_b, _ = load_model(path_b)
        model_c, _ = load_model(path_c)
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return

    # 3. Solve Tours
    print("Solving Neural Tours...")
    tour_a = solve_tour_model(model_a.to(device), nodes_t.to(device), graph_t.to(device))
    tour_b = solve_tour_model(model_b.to(device), nodes_t.to(device), graph_t.to(device))
    tour_c = solve_tour_model(model_c.to(device), nodes_t.to(device), graph_t.to(device))
    
    print(f"Solving Optimal Baseline with {LKH_PATH}...")
    tour_d = solve_lkh(meta['costs'])
    
    # Normalize Tours (Start at 0, Close Loop)
    tour_a = np.append(rotate_to_depot(tour_a), 0)
    tour_b = np.append(rotate_to_depot(tour_b), 0)
    tour_c = np.append(rotate_to_depot(tour_c), 0)
    tour_d = np.append(rotate_to_depot(tour_d), 0)

    loc = meta['loc']
    matrix = meta['costs']
    
    # 4. Calculate Metrics & Factors
    def get_plotting_data(tour):
        edges = []
        factors = []
        cumulative = [0.0]
        
        for i in range(len(tour)-1):
            u, v = tour[i], tour[i+1]
            cost = matrix[u, v]
            dist = np.linalg.norm(loc[u] - loc[v])
            
            # Factor > 1.0 means HEADWIND (Bad), < 1.0 means TAILWIND (Good)
            factor = cost / (dist + 1e-9)
            
            edges.append((u, v))
            factors.append(factor)
            cumulative.append(cumulative[-1] + cost)
            
        return edges, factors, cumulative

    edges_a, factors_a, total_a = get_plotting_data(tour_a)
    edges_b, factors_b, total_b = get_plotting_data(tour_b)
    edges_c, factors_c, total_c = get_plotting_data(tour_c)
    edges_d, factors_d, total_d = get_plotting_data(tour_d)
    
    # 5. Setup Plotting
    # Use LogNorm because costs are exponential: exp(-3) to exp(3)
    norm = LogNorm(vmin=np.exp(-ALPHA), vmax=np.exp(ALPHA))
    cmap = plt.get_cmap(CMAP_NAME)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 7.5))
    
    def setup_ax(ax, title, final_cost):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.axis('off')
        
        # Static elements: Nodes
        ax.scatter(loc[:,0], loc[:,1], c='gray', s=50, zorder=5)
        # Depot (Node 0)
        ax.scatter(loc[0,0], loc[0,1], c='yellow', edgecolors='black', s=150, zorder=6, label='Depot')
        
        # Wind Arrow
        center = np.mean(loc, axis=0)
        w = meta['wind']
        # Normalize arrow size for display
        w_norm = w / (np.linalg.norm(w) + 1e-9)
        ax.arrow(center[0]-0.2, center[1], w_norm[0]*0.2, w_norm[1]*0.2, 
                 width=0.02, color='lightblue', alpha=0.6, zorder=1)
        
        # Cost Box
        ax.text(0.5, -0.1, f"Final Cost: {final_cost:.2f}", 
                transform=ax.transAxes, ha='center', va='top', 
                fontsize=16, fontweight='bold', color='black',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))

    setup_ax(ax1, f"Coords (Geometry)", total_a[-1])
    setup_ax(ax2, f"Learned (Physics)", total_b[-1])
    setup_ax(ax3, f"Hybrid (Proposed)", total_c[-1])
    setup_ax(ax4, f"LKH-3 (Optimal)", total_d[-1])

    # Store line objects for animation
    lines_a, lines_b, lines_c, lines_d = [], [], [], []
    
    def draw_dotted_init(ax, edges, factors, line_store):
        for (u, v), factor in zip(edges, factors):
            color = cmap(norm(factor))
            # Initial state: dotted lines
            line, = ax.plot([loc[u,0], loc[v,0]], [loc[u,1], loc[v,1]], 
                            linestyle=':', linewidth=2, color=color, alpha=0.4)
            line_store.append(line)

    draw_dotted_init(ax1, edges_a, factors_a, lines_a)
    draw_dotted_init(ax2, edges_b, factors_b, lines_b)
    draw_dotted_init(ax3, edges_c, factors_c, lines_c)
    draw_dotted_init(ax4, edges_d, factors_d, lines_d)
    
    # Running cost text
    txt_a = ax1.text(0.05, 0.95, "Running: 0.00", transform=ax1.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    txt_b = ax2.text(0.05, 0.95, "Running: 0.00", transform=ax2.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    txt_c = ax3.text(0.05, 0.95, "Running: 0.00", transform=ax3.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    txt_d = ax4.text(0.05, 0.95, "Running: 0.00", transform=ax4.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2, ax3, ax4], orientation='horizontal', fraction=0.05, pad=0.15)
    cbar.set_label(f'Wind Efficiency (Green=Tailwind, Red=Headwind)')

    # 6. Update Loop
    def update(frame):
        artists = []
        for lines, total, txt, frames_len in [
            (lines_a, total_a, txt_a, len(edges_a)),
            (lines_b, total_b, txt_b, len(edges_b)),
            (lines_c, total_c, txt_c, len(edges_c)),
            (lines_d, total_d, txt_d, len(edges_d))
        ]:
            if frame < frames_len:
                # Solidify the current line
                line = lines[frame]
                line.set_linestyle('-')
                line.set_linewidth(4)
                line.set_color(SOLID_COLOR)
                line.set_alpha(1.0)
                
                # Update Text
                txt.set_text(f"Running: {total[frame+1]:.2f}")
                artists.append(line)
                artists.append(txt)
                
        return artists

    num_frames = max(len(edges_a), len(edges_b), len(edges_c), len(edges_d))
    
    print(f"Generating animation with {num_frames} frames...")
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=INTERVAL, blit=True, repeat=True)
    
    save_path = "animation_4models_lkh_robust.gif"
    print(f"Saving to {save_path}...")
    ani.save(save_path, writer='pillow', fps=2)
    print("Done!")

if __name__ == "__main__":
    run_animation()