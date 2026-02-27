import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# --- 1. CONFIGURATION ---
INPUT_FILE = "/home/pfc/gms/code/rethinking_tsp/results/evaluations/02experiment1_TSP50.csv"
OUTPUT_FILE_ENTROPY_GRID = "visualizations/csv_eval/TSP50_EXP1/eval_analysis_entropy_sensitivity_variant.png"
OUTPUT_FILE_CONFIDENCE = "visualizations/csv_eval/TSP50_EXP1/eval_analysis_confidence_calibration_variant.png"

# --- 2. DATA PROCESSING ---
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Ensure Distribution exists
    if 'Distribution' not in df.columns:
        df['Distribution'] = 'in'
        
    def parse_row(row):
        name = row['Model_Name']
        if name == "LKH_Baseline":
            return "LKH", "baseline", 0.0, "LKH", "baseline", "in"
        
        parts = name.split('_')
        exp_name = parts[0]
        
        # Data Type (Density)
        data_type = "fully_connected" if "fully_connected" in name else "sparse"
        
        # Extract Feature Type
        ftype = "unknown"
        for t in ['blank', 'coords', 'hybrid', 'learned']:
            if t in parts:
                ftype = t
                break
        
        # Extract Entropy
        entropy = 0.0
        for p in parts:
            if p.startswith('ent'):
                try:
                    entropy = float(p.replace('ent', ''))
                except:
                    pass
        
        # Create "Architecture (Nd)" Label
        arch_dims_label = f"{ftype.capitalize()} ({row['Feature_Dims']}d)"
        
        dist = row['Distribution']
        
        return exp_name, ftype, entropy, arch_dims_label, data_type, dist

    df[['Experiment', 'Feature_Type', 'Entropy', 'Arch_Dims', 'Data_Type', 'Distribution']] = df.apply(
        lambda row: pd.Series(parse_row(row)), axis=1
    )
    
    # Create combined Config for Color mapping (Arch + Density)
    df['Model_Config'] = df['Arch_Dims'] + "|" + df['Data_Type']
    
    # Filter out LKH and Blank models
    df = df[~df['Model_Name'].isin(['LKH_Baseline']) & (df['Feature_Type'] != 'blank')].copy()
    
    return df

# --- 3. COLOR PALETTE GENERATION ---
def get_elegant_palette(df):
    unique_configs = sorted(df['Model_Config'].unique())
    unique_archs = sorted(df['Arch_Dims'].unique())
    
    def sort_key(s):
        if "Coords" in s: prio = 0
        elif "Learned" in s: prio = 1
        elif "Hybrid" in s: prio = 2
        else: prio = 3
        try: d = int(s.split('(')[1].split('d')[0])
        except: d = 0
        # Push fully connected after sparse within the same arch
        dens_prio = 0 if "sparse" in s else 1 
        return (prio, d, dens_prio)
        
    hue_order = sorted(unique_configs, key=sort_key)
    
    base_colormaps = {
        'Coords': plt.get_cmap("Purples"),
        'Learned': plt.get_cmap("Oranges"),
        'Hybrid': plt.get_cmap("Greens")
    }
    
    palette = {}
    for config in hue_order:
        arch_dims, data_type = config.split('|')
        cmap = next((v for k, v in base_colormaps.items() if k in arch_dims), plt.get_cmap("Greys"))
        
        # Exact Saturation Logic
        palette[config] = cmap(0.4) if data_type == 'sparse' else cmap(0.8)
            
    return hue_order, palette, unique_archs

# --- 4. PLOT 1: ROBUST ENTROPY SENSITIVITY GRID ---
def plot_entropy_sensitivity_grid(df, hue_order, palette, unique_archs):
    strategies = [0, 10, 100, 1280]
    strategy_names = ["Greedy", "BS-10", "BS-100", "BS-1280"]
    
    # Filtering Logic
    exp_counts = df[df['Width'] == 0].groupby('Experiment')['Entropy'].nunique()
    max_count = exp_counts.max()
    threshold = max_count / 2
    valid_experiments = exp_counts[exp_counts >= threshold].index.tolist()
    
    df_filtered = df[df['Experiment'].isin(valid_experiments)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    sns.set_style("whitegrid")
    
    for i, width in enumerate(strategies):
        ax = axes[i]
        subset = df_filtered[df_filtered['Width'] == width].copy()
        
        # AGGREGATE: Group by Model_Config, Distribution, and Entropy -> mean Gap_MoR
        agg_subset = subset.groupby(['Model_Config', 'Distribution', 'Entropy'], as_index=False)['Gap_MoR'].mean()
        
        if agg_subset.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        sns.lineplot(
            data=agg_subset, 
            x='Entropy', 
            y='Gap_MoR',               # Using Gap_MoR
            hue='Model_Config', 
            hue_order=hue_order,
            palette=palette,
            style='Distribution',      # Distribution dictates Line Style
            dashes={'in': (None, None), 'out': (4, 3)}, 
            markers=True, 
            linewidth=2.5, 
            markersize=7,
            ax=ax,
            legend=False
        )
        
        ax.set_title(f"Strategy: {strategy_names[i]}", fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        if i >= 2: ax.set_xlabel("Entropy Coefficient (Log Scale)", fontsize=12)
        else: ax.set_xlabel("")
        if i % 2 == 0: ax.set_ylabel("Optimality Gap MoR (%)", fontsize=12)
        else: ax.set_ylabel("")

    # --- CUSTOM LEGENDS ---
    base_colormaps = {'Coords': plt.get_cmap("Purples"), 'Learned': plt.get_cmap("Oranges"), 'Hybrid': plt.get_cmap("Greens")}
    
    # 1. Architecture & Density
    arch_handles = []
    for arch in unique_archs:
        cmap = next((v for k, v in base_colormaps.items() if k in arch), plt.get_cmap("Greys"))
        arch_handles.append(mpatches.Patch(color=cmap(0.8), label=f"{arch} (Full)"))
        arch_handles.append(mpatches.Patch(color=cmap(0.4), label=f"{arch} (Sparse)"))
        
    leg1 = fig.legend(handles=arch_handles, title="Arch & Density", loc='center left', bbox_to_anchor=(0.86, 0.65), fontsize=10)
    
    # 2. Distribution
    dist_handles = [
        mlines.Line2D([], [], color='black', linestyle='-', label='In-Distribution (ID)'),
        mlines.Line2D([], [], color='black', linestyle='--', label='Out-of-Distribution (OOD)')
    ]
    fig.legend(handles=dist_handles, title="Generalization", loc='center left', bbox_to_anchor=(0.86, 0.40), fontsize=10)
    
    plt.suptitle("Entropy Sensitivity (Aggregated over Experiments)", fontsize=18, fontweight='bold', y=0.98)
    plt.subplots_adjust(right=0.85)
    plt.savefig(OUTPUT_FILE_ENTROPY_GRID, dpi=300, bbox_inches='tight')
    print(f"Saved {OUTPUT_FILE_ENTROPY_GRID}")
    plt.show()

# --- 5. PLOT 2: CLEAN CONFIDENCE CALIBRATION ---
def plot_confidence_calibration(df, hue_order, palette, unique_archs):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    sns.scatterplot(
        data=df,
        x='Avg_Confidence',
        y='Gap_MoR',                   # Using Gap_MoR
        hue='Model_Config',
        hue_order=hue_order,
        palette=palette,
        style='Distribution',          # Distribution dictates Marker Shape
        markers={'in': 'o', 'out': 'X'}, 
        size='Width',
        sizes=(50, 300),
        alpha=0.85,
        edgecolor='black',
        linewidth=0.5
    )
    
    plt.title("Confidence vs. Competence Calibration", fontsize=16, fontweight='bold')
    plt.ylabel("Optimality Gap MoR (%) (Lower is Better)", fontsize=14, fontweight='bold')
    plt.xlabel("Average Model Confidence (Higher is Surer)", fontsize=14, fontweight='bold')
    
    # Build a clean legend manually to avoid Seaborn's messy default scatter legend
    ax = plt.gca()
    ax.get_legend().remove()
    
    # 1. Arch & Density
    base_colormaps = {'Coords': plt.get_cmap("Purples"), 'Learned': plt.get_cmap("Oranges"), 'Hybrid': plt.get_cmap("Greens")}
    arch_handles = []
    for arch in unique_archs:
        cmap = next((v for k, v in base_colormaps.items() if k in arch), plt.get_cmap("Greys"))
        arch_handles.append(mlines.Line2D([], [], color=cmap(0.8), marker='s', linestyle='', label=f"{arch} (Full)"))
        arch_handles.append(mlines.Line2D([], [], color=cmap(0.4), marker='s', linestyle='', label=f"{arch} (Sparse)"))
    leg1 = plt.legend(handles=arch_handles, title="Arch & Density", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.add_artist(leg1)
    
    # 2. Distribution
    dist_handles = [
        mlines.Line2D([], [], color='gray', marker='o', linestyle='', markersize=8, label='In-Dist (ID)'),
        mlines.Line2D([], [], color='gray', marker='X', linestyle='', markersize=8, label='Out-of-Dist (OOD)')
    ]
    leg2 = plt.legend(handles=dist_handles, title="Generalization", bbox_to_anchor=(1.02, 0.6), loc='upper left')
    ax.add_artist(leg2)
    
    # 3. Size (Strategy)
    width_map = {0: 50, 10: 100, 100: 180, 1280: 300}
    size_handles = [mlines.Line2D([], [], color='gray', marker='o', linestyle='', markersize=np.sqrt(s)/1.5, label=str(w)) for w, s in width_map.items()]
    plt.legend(handles=size_handles, labels=["Greedy", "BS-10", "BS-100", "BS-1280"], title="Strategy", bbox_to_anchor=(1.02, 0.4), loc='upper left')
    
    plt.subplots_adjust(right=0.75)
    plt.savefig(OUTPUT_FILE_CONFIDENCE, dpi=300, bbox_inches='tight')
    print(f"Saved {OUTPUT_FILE_CONFIDENCE}")
    plt.show()

# --- 6. EXECUTION ---
if __name__ == "__main__":
    try:
        df = process_data(INPUT_FILE)
        hue_order, palette, unique_archs = get_elegant_palette(df)
        
        plot_entropy_sensitivity_grid(df, hue_order, palette, unique_archs)
        plot_confidence_calibration(df, hue_order, palette, unique_archs)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()