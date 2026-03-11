import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# --- CONFIGURATION ---
INPUT_FILE = "results/evaluations/03experiment1_compknn_TSP20.csv"
OUTPUT_FILE = "visualizations/csv_eval/TSP20_EXP3/eval_embedding_variance_TSP20_CONNECTIONS.png"

# --- DATA PROCESSING ---
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    
    def parse_row(row):
        name = row['Model_Name']
        if "LKH" in name:
            # LKH has no embeddings, we will filter it out for this plot
            return "LKH", "unknown", "LKH", "unknown"
        
        parts = name.split('_')
        exp_name = parts[0]
        
        ent_idx = next((i for i, p in enumerate(parts) if p.startswith('ent')), -1)
        if ent_idx != -1 and len(parts) > ent_idx + 1:
            connection = parts[ent_idx + 1]
            ftype = parts[ent_idx - 1].capitalize()
            msg_passing = "_".join(parts[1:ent_idx-1])
        else:
            connection = "Unknown"
            ftype = "Unknown"
            msg_passing = "Unknown"
        
        dims = row.get('Feature_Dims', 0)
        arch_group = f"{ftype} ({dims}d) {connection}"
        
        return exp_name, ftype, arch_group, msg_passing

    df[['Experiment', 'Feature_Type', 'Arch_Group', 'Msg_Passing']] = df.apply(
        lambda row: pd.Series(parse_row(row)), axis=1
    )
    
    # Filter out LKH and Blank models completely (LKH doesn't have neural embeddings)
    df = df[~df['Model_Name'].isin(['LKH_Baseline']) & (df['Feature_Type'] != 'blank')].copy()
    
    # Calculate Speedup (Optional for this plot, but keeps df consistent)
    df['Time_Per_Solution'] = df['Time_Per_Solution'].replace(0, 1e-9)
    
    def get_strategy_label(width):
        if width == 0: return "Greedy"
        return f"Beam-{int(width)}"
        
    df['Strategy'] = df['Width'].apply(get_strategy_label)
    
    width_sort = {0: 0, 10: 1, 100: 2, 1280: 3}
    df['Sort_Index'] = df['Width'].map(width_sort)
    df = df.sort_values(by='Sort_Index')
    
    return df

# --- COLOR PALETTE GENERATION ---
def get_elegant_palette(df):
    unique_labels = sorted(df['Arch_Group'].unique())
    
    def sort_key(s):
        if "Coords" in s: prio = 0
        elif "Learned" in s: prio = 1
        elif "Hybrid" in s: prio = 2
        else: prio = 3
        try: d = int(s.split('(')[1].split('d')[0])
        except: d = 0
        
        if 'fullyConnected' in s: c_prio = 0
        elif 'random20' in s: c_prio = 1
        elif 'knn20' in s: c_prio = 2
        else: c_prio = 3
        
        return (prio, d, c_prio)
        
    hue_order = sorted(unique_labels, key=sort_key)
    
    palette = {}
    for label in hue_order:
        if "Coords" in label: cmap = plt.get_cmap("Purples")
        elif "Learned" in label: cmap = plt.get_cmap("Oranges")
        elif "Hybrid" in label: cmap = plt.get_cmap("Greens")
        else: cmap = plt.get_cmap("Greys")
        
        if 'fullyConnected' in label: alpha = 0.9
        elif 'random20' in label: alpha = 0.55
        elif 'knn20' in label: alpha = 0.2
        else: alpha = 0.5
        
        palette[label] = cmap(alpha)
            
    return hue_order, palette

# --- Consistent Message Passing Markers ---
MSG_MARKERS = {
    'forward': 'o',        # Circle
    'backward': 's',       # Square
    'concat_dual': 'X',    # Cross
    'gated_dual': 'D',     # Diamond
    'sum_dual': '*'        # Star
}

# --- PLOTTING ---
def plot_embedding_variance(df, hue_order, palette):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # 1. Explicitly define 4 highly distinct sizes mapped to the categorical Strategy names
    strategy_sizes = {
        "Greedy": 50,      # Smallest
        "Beam-10": 120,    # Medium-Small
        "Beam-100": 220,   # Medium-Large
        "Beam-1280": 400   # Largest
    }
    
    ax = sns.scatterplot(
        data=df,
        x='Avg_Embedding_Variance',
        y='Gap_MoR',
        hue='Arch_Group',
        hue_order=hue_order,
        palette=palette,
        style='Msg_Passing',
        markers=MSG_MARKERS,
        size='Strategy',       # <-- CHANGED: Use categorical name, not numeric Width
        sizes=strategy_sizes,  # <-- CHANGED: Pass our explicit dictionary
        alpha=0.85,
        edgecolor='black',
        linewidth=0.5
    )
    
    plt.title("Latent Space Expressivity vs. Competence\n(Impact of Node Embedding Variance on Optimality Gap)", fontsize=16, fontweight='bold')
    plt.ylabel("Optimality Gap MoR (%) (Lower is Better)", fontsize=14, fontweight='bold')
    plt.xlabel("Average Embedding Variance (Higher = More Distinct Node Features)", fontsize=14, fontweight='bold')
    
    # Remove default legend and build clean custom ones
    ax.get_legend().remove()
    
    # 1. Architecture
    arch_handles = [mpatches.Patch(facecolor=palette[lbl], edgecolor='black', label=lbl) for lbl in hue_order]
    leg1 = plt.legend(handles=arch_handles, title="Topological Backbone", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.gca().add_artist(leg1)
    
    # 2. Message Passing
    existing_msgs = sorted(df['Msg_Passing'].unique())
    msg_handles = [mlines.Line2D([], [], color='white', markeredgecolor='black', markerfacecolor='gray', 
                                 marker=MSG_MARKERS.get(m, 'o'), linestyle='', markersize=10, 
                                 label=m.replace('_', ' ').title()) for m in existing_msgs]
    leg2 = plt.legend(handles=msg_handles, title="Message Passing Scheme", bbox_to_anchor=(1.02, 0.65), loc='upper left')
    plt.gca().add_artist(leg2)
    
    # 3. Strategy / Width (Updated to match discrete sizing)
    # Matplotlib markersize is the square root of scatter area (s), so we use np.sqrt(size)
    strat_order = ["Greedy", "Beam-10", "Beam-100", "Beam-1280"]
    existing_strats = [s for s in strat_order if s in df['Strategy'].unique()]
    
    size_handles = [mlines.Line2D([], [], color='gray', marker='o', linestyle='', 
                                  markersize=np.sqrt(strategy_sizes[s]), 
                                  label=s) for s in existing_strats]
    plt.legend(handles=size_handles, title="Decoding Strategy", bbox_to_anchor=(1.02, 0.35), loc='upper left')
    
    plt.subplots_adjust(right=0.75)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', bbox_extra_artists=(leg1, leg2))
    print(f"Chart saved to {OUTPUT_FILE}")
    plt.show()

# --- EXECUTION ---
if __name__ == "__main__":
    df = process_data(INPUT_FILE)
    hue_order, palette = get_elegant_palette(df)
    plot_embedding_variance(df, hue_order, palette)