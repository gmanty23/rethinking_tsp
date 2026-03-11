import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.lines as mlines

# --- 1. CONFIGURATION ---
INPUT_FILE = "/home/pfc/gms/code/rethinking_tsp/results/evaluations/02experiment1_TSP50.csv"
OUTPUT_FILE = "visualizations/csv_eval/TSP50_EXP1/eval_pareto_frontier_TSP50.png"

# --- 2. DATA PROCESSING ---
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Get LKH Time for Speedup Calculation
    try:
        # Note: Added a check for Strategy 'opt' if the name varies
        lkh_row = df[df['Model_Name'] == 'LKH_Baseline']
        lkh_time = lkh_row['Time_Per_Solution'].iloc[0]
    except IndexError:
        print("Warning: LKH Baseline not found. Using default 0.0201s")
        lkh_time = 0.0201

    def parse_row(row):
        name = row['Model_Name']
        if name == "LKH_Baseline":
            return "LKH", "baseline", 0.0, "LKH", "baseline"
        
        parts = name.split('_')
        
        # Data Type (Sparse vs Fully Connected)
        data_type = "fully_connected" if "fully_connected" in name else "sparse"
        
        # Feature Type
        ftype = "unknown"
        for t in ['blank', 'coords', 'hybrid', 'learned']:
            if t in parts:
                ftype = t
                break
        
        # Entropy
        entropy = 0.0
        for p in parts:
            if p.startswith('ent'):
                try:
                    # Handle potential timestamp trailing after entropy
                    raw_ent = p.replace('ent', '')
                    entropy = float(raw_ent)
                except:
                    pass
        
        # Architecture Label (Grouping by type and dims)
        arch_label = f"{ftype.capitalize()} ({row['Feature_Dims']}d)"
        
        return parts[0], ftype, entropy, arch_label, data_type

    df[['Experiment', 'Feature_Type', 'Entropy', 'Arch_Group', 'Data_Type']] = df.apply(
        lambda row: pd.Series(parse_row(row)), axis=1
    )
    
    # Speedup Calculation
    df['Time_Per_Solution'] = df['Time_Per_Solution'].replace(0, 1e-9)
    df['Speedup_Factor'] = lkh_time / df['Time_Per_Solution']
    
    # Unique ID for Trajectories: Now includes Data_Type and Distribution
    df['Model_ID'] = df['Arch_Group'] + "_" + df['Data_Type'] + "_" + df['Distribution'] + "_" + df['Entropy'].astype(str)
    
    return df

# --- 3. PLOTTING ---
def plot_pareto(df):
    neural_df = df[df['Model_Name'] != 'LKH_Baseline'].copy()
    
    plt.figure(figsize=(20, 10))
    sns.set_style("whitegrid")

    # --- COLOR MAP (Architecture Groups) ---
    unique_archs = sorted(neural_df['Arch_Group'].unique())
    
    def sort_key(s):
        if "Coords" in s: prio = 0
        elif "Learned" in s: prio = 1
        elif "Hybrid" in s: prio = 2
        else: prio = 3
        try: d = int(s.split('(')[1].split('d')[0])
        except: d = 0
        return (prio, d)
        
    unique_archs = sorted(unique_archs, key=sort_key)
    
    # Base Colors
    base_colormaps = {
        'Coords': plt.get_cmap("Purples"),
        'Learned': plt.get_cmap("Oranges"),
        'Hybrid': plt.get_cmap("Greens")
    }

    # Assign specific color based on Arch_Group and Data_Type shade
    color_mapping = {} 
    for arch in unique_archs:
        # Determine base map
        cmap = next((v for k, v in base_colormaps.items() if k in arch), plt.get_cmap("Greys"))
        
        # Data_Type shade logic: Sparse (Light) vs Fully Connected (Dark)
        color_mapping[(arch, 'sparse')] = cmap(0.4)
        color_mapping[(arch, 'fully_connected')] = cmap(0.8)

    # --- STYLE MAP (Distribution) ---
    # in -> Solid, out -> Dashed
    style_map = {'in': '-', 'out': '--'}

    # --- MARKER MAP (Entropy) ---
    unique_ents = sorted(neural_df['Entropy'].unique())
    markers = ['^', 's', 'D', 'p', 'o', 'H', '8']
    marker_map = {ent: markers[i % len(markers)] for i, ent in enumerate(unique_ents)}

    # --- SIZE MAP (Strategy) ---
    width_map = {0: 50, 10: 100, 100: 180, 1280: 300}
    
    # --- DRAW TRAJECTORIES ---
    for model_id in neural_df['Model_ID'].unique():
        subset = neural_df[neural_df['Model_ID'] == model_id].sort_values(by='Width')
        
        arch = subset['Arch_Group'].iloc[0]
        dtype = subset['Data_Type'].iloc[0]
        dist = subset['Distribution'].iloc[0]
        
        col = color_mapping.get((arch, dtype), 'gray')
        sty = style_map.get(dist, '-')
        
        plt.plot(subset['Speedup_Factor'], subset['Gap_MoR'], 
                 color=col, linestyle=sty, alpha=0.4, linewidth=1.5, zorder=1)

    # --- DRAW POINTS ---
    for idx, row in neural_df.iterrows():
        col = color_mapping.get((row['Arch_Group'], row['Data_Type']), 'gray')
        
        plt.scatter(
            row['Speedup_Factor'], 
            row['Gap_MoR'], 
            c=[col], 
            marker=marker_map.get(row['Entropy'], 'o'), 
            s=width_map.get(row['Width'], 100),
            edgecolor='white', linewidth=0.5, alpha=0.9, zorder=2
        )

    # --- DRAW LKH REFERENCE ---
    plt.axvline(x=1.0, color='black', linestyle='-', linewidth=1.5, zorder=0)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3, zorder=0)
    plt.scatter(1.0, 0, color='black', marker='*', s=500, label='LKH Solver', zorder=3)
    plt.text(1.1, 0.5, "LKH Baseline", verticalalignment='bottom', fontsize=10, fontweight='bold')

    # --- AXES & LABELS ---
    plt.xscale('log')
    plt.xlabel("Speedup Factor relative to LKH (Log Scale)", fontsize=12, fontweight='bold')
    plt.ylabel("Optimality Gap (%)", fontsize=12, fontweight='bold')
    plt.title("Pareto Frontier: Sparse vs Fully Connected Windy TSP", fontsize=16)

    ticks = [0.1, 0.5, 1, 10, 50, 100]
    plt.xticks(ticks, [f"{t}x" for t in ticks])

    # --- LEGENDS ---
    
    # 1. Architecture & Shade (Color)
    arch_lines = []
    for arch in unique_archs:
        arch_lines.append(mlines.Line2D([], [], color=color_mapping[(arch, 'fully_connected')], 
                          marker='s', linestyle='', label=f"{arch} (Full)"))
        arch_lines.append(mlines.Line2D([], [], color=color_mapping[(arch, 'sparse')], 
                          marker='s', linestyle='', label=f"{arch} (Sparse)"))
    
    leg1 = plt.legend(handles=arch_lines, title="Arch & Density", loc='upper left', 
                      bbox_to_anchor=(1.02, 1.0), fontsize=9)
    plt.gca().add_artist(leg1)

    # 2. Distribution (Line Style)
    dist_lines = [mlines.Line2D([], [], color='black', linestyle=s, label=f"Dist: {l}") for l, s in style_map.items()]
    leg_dist = plt.legend(handles=dist_lines, title="Distribution", loc='upper left', bbox_to_anchor=(1.02, 0.6))
    plt.gca().add_artist(leg_dist)

    # 3. Entropy (Shape)
    shape_lines = [mlines.Line2D([], [], color='gray', marker=marker_map[e], linestyle='', markersize=10) for e in unique_ents]
    leg2 = plt.legend(shape_lines, [f"Ent={e}" for e in unique_ents], title="Entropy", loc='upper left', bbox_to_anchor=(1.02, 0.45))
    plt.gca().add_artist(leg2)
    
    # 4. Strategy (Size)
    size_lines = [mlines.Line2D([], [], color='gray', marker='o', linestyle='', markersize=np.sqrt(s)/1.5) for s in width_map.values()]
    plt.legend(size_lines, ["Greedy", "BS-10", "BS-100", "BS-1280"], title="Strategy", loc='upper left', bbox_to_anchor=(1.02, 0.20))

    plt.subplots_adjust(right=0.82) # Give it just a bit more breathing room on the right
    # Adding bbox_inches='tight' forces matplotlib to calculate the true bounding box of ALL elements
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved to {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    df = process_data(INPUT_FILE)
    plot_pareto(df)