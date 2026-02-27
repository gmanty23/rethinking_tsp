import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.lines as mlines

# --- 1. CONFIGURATION ---
INPUT_FILE = "/home/pfc/gms/code/rethinking_tsp/results/evaluations/01experiment1NoBlank_TSP2020.csv"
OUTPUT_FILE = "visualizations/csv_eval/TSP20_EXP1/eval_pareto_frontier_TSP20_exp1.png"

# --- 2. DATA PROCESSING ---
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Get LKH Time for Speedup Calculation
    try:
        lkh_time = df[df['Model_Name'] == 'LKH_Baseline']['Time_Per_Solution'].iloc[0]
    except IndexError:
        print("Warning: LKH Baseline not found. Using default 0.0037s")
        lkh_time = 0.0037

    def parse_row(row):
        name = row['Model_Name']
        if name == "LKH_Baseline":
            return "LKH", "baseline", 0.0, "LKH"
        
        parts = name.split('_')
        exp_name = parts[0]
        
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
                    entropy = float(p.replace('ent', ''))
                except:
                    pass
        
        # Architecture Label
        arch_label = f"{ftype.capitalize()} ({row['Feature_Dims']}d)"
        
        return exp_name, ftype, entropy, arch_label

    df[['Experiment', 'Feature_Type', 'Entropy', 'Model_Category']] = df.apply(
        lambda row: pd.Series(parse_row(row)), axis=1
    )
    
    # Speedup Calculation
    df['Time_Per_Solution'] = df['Time_Per_Solution'].replace(0, 1e-9)
    df['Speedup_Factor'] = lkh_time / df['Time_Per_Solution']
    
    # Unique ID for Trajectories
    df['Model_ID'] = df['Experiment'] + "_" + df['Model_Category'] + "_" + df['Entropy'].astype(str)
    
    return df

# --- 3. PLOTTING ---
def plot_pareto(df):
    neural_df = df[df['Model_Name'] != 'LKH_Baseline'].copy()
    
    plt.figure(figsize=(16, 10))
    sns.set_style("whitegrid")

    # --- COLOR MAP (Architecture) ---
    unique_cats = sorted(neural_df['Model_Category'].unique())
    
    # Sort: Coords -> Learned -> Hybrid, then Dims
    def sort_key(s):
        if "Coords" in s: prio = 0
        elif "Learned" in s: prio = 1
        elif "Hybrid" in s: prio = 2
        else: prio = 3
        try: d = int(s.split('(')[1].split('d')[0])
        except: d = 0
        return (prio, d)
        
    unique_cats = sorted(unique_cats, key=sort_key)
    
    # Generate Swapped Elegant Palette
    color_map = {}
    groups = {'Coords': [], 'Learned': [], 'Hybrid': []}
    for c in unique_cats:
        for k in groups:
            if k in c: groups[k].append(c)
            
    if groups['Coords']:
        colors = plt.get_cmap("Purples")(np.linspace(0.5, 0.9, len(groups['Coords'])))
        for i, c in enumerate(groups['Coords']): color_map[c] = colors[i]
    if groups['Learned']:
        colors = plt.get_cmap("Oranges")(np.linspace(0.5, 0.9, len(groups['Learned'])))
        for i, c in enumerate(groups['Learned']): color_map[c] = colors[i]
    if groups['Hybrid']:
        colors = plt.get_cmap("Greens")(np.linspace(0.5, 0.9, len(groups['Hybrid'])))
        for i, c in enumerate(groups['Hybrid']): color_map[c] = colors[i]

    # --- STYLE MAP (Experiment) ---
    unique_exps = sorted(neural_df['Experiment'].unique())
    style_map = {}
    
    # Logic: 05Conf and 06Conf get Solid lines. Others get dashed/dotted.
    # You can add more specific rules here if needed.
    available_styles = ['--', ':', '-.'] # Dashed styles for others
    style_idx = 0
    
    for exp in unique_exps:
        if "05ConfOgStats" in exp or "06ConfNewStats" in exp:
            style_map[exp] = '-' # Solid Line for main results
        else:
            # Cycle through dashed styles for older/other experiments
            style_map[exp] = available_styles[style_idx % len(available_styles)]
            style_idx += 1

    # --- MARKER MAP (Entropy) ---
    unique_ents = sorted(neural_df['Entropy'].unique())
    markers = ['^', 's', 'D', 'p', 'o', 'H', '8']
    marker_map = {ent: markers[i % len(markers)] for i, ent in enumerate(unique_ents)}

    # --- SIZE MAP (Strategy) ---
    width_map = {0: 50, 10: 100, 100: 180, 1280: 300}
    
    # --- DRAW TRAJECTORIES ---
    print("Drawing trajectories...")
    for model_id in neural_df['Model_ID'].unique():
        subset = neural_df[neural_df['Model_ID'] == model_id].sort_values(by='Width')
        
        cat = subset['Model_Category'].iloc[0]
        exp = subset['Experiment'].iloc[0]
        
        col = color_map.get(cat, 'gray')
        sty = style_map.get(exp, '-') # Default to solid if unknown
        
        plt.plot(subset['Speedup_Factor'], subset['Gap_Percent'], 
                 color=col, linestyle=sty, alpha=0.5, linewidth=1.5, zorder=1)

    # --- DRAW POINTS ---
    print("Drawing points...")
    for idx, row in neural_df.iterrows():
        cat = row['Model_Category']
        ent = row['Entropy']
        width = row['Width']
        
        plt.scatter(
            row['Speedup_Factor'], 
            row['Gap_Percent'], 
            c=[color_map.get(cat, 'gray')], 
            marker=marker_map.get(ent, 'o'), 
            s=width_map.get(width, 100),
            edgecolor='white', linewidth=0.5, alpha=0.9, zorder=2
        )

    # --- DRAW LKH REFERENCE ---
    plt.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, zorder=0)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3, zorder=0)
    plt.scatter(1.0, 0, color='black', marker='*', s=500, label='LKH Solver', zorder=3)
    plt.text(1.1, 0.5, "LKH Baseline (3.7ms)", verticalalignment='bottom', fontsize=10, fontweight='bold')

    # --- AXES & LABELS ---
    plt.xscale('log')
    plt.xlabel("Speedup Factor relative to LKH (Log Scale)", fontsize=12, fontweight='bold')
    plt.ylabel("Optimality Gap (%)", fontsize=12, fontweight='bold')
    plt.title("Pareto Frontier: Windy TSP Efficiency vs Optimality", fontsize=16)

    ticks = [0.1, 0.5, 1, 10, 50, 100]
    plt.xticks(ticks, [f"{t}x" for t in ticks])

    # --- LEGENDS ---
    
    # 1. Architecture (Color)
    leg_lines = [mlines.Line2D([], [], color=color_map[c], marker='o', linestyle='', markersize=10) for c in unique_cats]
    leg1 = plt.legend(leg_lines, unique_cats, title="Architecture", loc='upper left', bbox_to_anchor=(1.02, 1.0))
    plt.gca().add_artist(leg1)

    # 2. Experiment (Line Style)
    # Only show unique styles used
    used_styles = {}
    for exp, sty in style_map.items():
        if exp in neural_df['Experiment'].values:
            label = "Main (05/06)" if sty == '-' else exp
            if label not in used_styles: # Avoid duplicates in legend
                used_styles[label] = sty
    
    style_lines = [mlines.Line2D([], [], color='black', linestyle=s, label=l) for l, s in used_styles.items()]
    leg_style = plt.legend(style_lines, used_styles.keys(), title="Experiment Series", loc='upper left', bbox_to_anchor=(1.02, 0.65))
    plt.gca().add_artist(leg_style)

    # 3. Entropy (Shape)
    shape_lines = [mlines.Line2D([], [], color='gray', marker=marker_map[e], linestyle='', markersize=10) for e in unique_ents]
    leg2 = plt.legend(shape_lines, [f"Ent={e}" for e in unique_ents], title="Entropy", loc='upper left', bbox_to_anchor=(1.02, 0.45))
    plt.gca().add_artist(leg2)
    
    # 4. Strategy (Size)
    size_lines = [mlines.Line2D([], [], color='gray', marker='o', linestyle='', markersize=np.sqrt(s)/1.5) for s in width_map.values()]
    leg3 = plt.legend(size_lines, ["Greedy", "BS-10", "BS-100", "BS-1280"], title="Strategy", loc='upper left', bbox_to_anchor=(1.02, 0.20))

    plt.subplots_adjust(right=0.8)
    plt.savefig(
        OUTPUT_FILE, 
        dpi=300, 
        bbox_inches='tight', 
        bbox_extra_artists=(leg1, leg_style, leg2, leg3)
    )
    
    print(f"Saved to {OUTPUT_FILE}")
    plt.show()

# --- RUN ---
if __name__ == "__main__":
    df = process_data(INPUT_FILE)
    plot_pareto(df)