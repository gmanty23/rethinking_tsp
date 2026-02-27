import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.lines as mlines

# --- CONFIGURATION ---
INPUT_FILE = "/home/pfc/gms/code/rethinking_tsp/results/evaluations/01experiment1ONLY56.csv"
OUTPUT_FILE_ENTROPY_GRID = "visualizations/csv_eval/TSP20_EXP1/eval_analysis_entropy_sensitivity_colored_ONLY56.png"
OUTPUT_FILE_CONFIDENCE = "visualizations/csv_eval/TSP20_EXP1/eval_analysis_confidence_calibration_colored_ONLY56.png"

# --- DATA PROCESSING ---
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    
    def parse_row(row):
        name = row['Model_Name']
        if name == "LKH_Baseline":
            return "LKH", "baseline", 0.0, "LKH"
        
        parts = name.split('_')
        exp_name = parts[0]
        
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
        arch_base = ftype.capitalize()
        arch_dims_label = f"{arch_base} ({row['Feature_Dims']}d)"
        
        return exp_name, ftype, entropy, arch_dims_label

    df[['Experiment', 'Feature_Type', 'Entropy', 'Arch_Dims']] = df.apply(
        lambda row: pd.Series(parse_row(row)), axis=1
    )
    
    # Filter out LKH and Blank models
    df = df[~df['Model_Name'].isin(['LKH_Baseline']) & (df['Feature_Type'] != 'blank')].copy()
    
    return df

# --- COLOR PALETTE GENERATION ---
def get_elegant_palette(df):
    """
    Generates the 'Swapped Elegant' palette:
    - Coords: Purples
    - Learned: Oranges
    - Hybrid: Greens
    """
    unique_labels = sorted(df['Arch_Dims'].unique())
    
    def sort_key(s):
        if "Coords" in s: prio = 0
        elif "Learned" in s: prio = 1
        elif "Hybrid" in s: prio = 2
        else: prio = 3
        try: d = int(s.split('(')[1].split('d')[0])
        except: d = 0
        return (prio, d)
        
    hue_order = sorted(unique_labels, key=sort_key)
    
    palette = {}
    groups = {'Coords': [], 'Learned': [], 'Hybrid': []}
    for label in hue_order:
        for key in groups:
            if key in label: groups[key].append(label)
    
    if groups['Coords']:
        colors = plt.get_cmap("Purples")(np.linspace(0.5, 0.9, len(groups['Coords'])))
        for i, l in enumerate(groups['Coords']): palette[l] = colors[i]
    if groups['Learned']:
        colors = plt.get_cmap("Oranges")(np.linspace(0.5, 0.9, len(groups['Learned'])))
        for i, l in enumerate(groups['Learned']): palette[l] = colors[i]
    if groups['Hybrid']:
        colors = plt.get_cmap("Greens")(np.linspace(0.5, 0.9, len(groups['Hybrid'])))
        for i, l in enumerate(groups['Hybrid']): palette[l] = colors[i]
            
    return hue_order, palette

# --- PLOT 1: ROBUST ENTROPY SENSITIVITY GRID ---
def plot_entropy_sensitivity_grid(df, hue_order, palette):
    strategies = [0, 10, 100, 1280]
    strategy_names = ["Greedy (0)", "BS-10", "BS-100", "BS-1280"]
    
    # --- FILTERING LOGIC ---
    # 1. Count how many entropy values each experiment has
    # We look at just ONE strategy (e.g. Width=0) to count unique entropies per experiment
    exp_counts = df[df['Width'] == 0].groupby('Experiment')['Entropy'].nunique()
    max_count = exp_counts.max()
    threshold = max_count / 2
    
    valid_experiments = exp_counts[exp_counts >= threshold].index.tolist()
    
    print(f"Filtering Experiments for Sensitivity Plot:")
    print(f"Max Entropies found: {max_count}. Threshold: {threshold}")
    print(f"Keeping {len(valid_experiments)} experiments. Dropping {len(exp_counts) - len(valid_experiments)}.")
    
    # Apply Filter
    df_filtered = df[df['Experiment'].isin(valid_experiments)].copy()

    # Create Grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    sns.set_style("whitegrid")
    
    for i, width in enumerate(strategies):
        ax = axes[i]
        
        # 1. Select data for this strategy
        subset = df_filtered[df_filtered['Width'] == width].copy()
        
        # 2. AGGREGATE: Group by Arch+Dims and Entropy, compute MEAN Gap
        # This merges multiple experiments (e.g., 05Conf and 06Conf) into one smooth line
        agg_subset = subset.groupby(['Arch_Dims', 'Entropy'], as_index=False)['Gap_Percent'].mean()
        
        if agg_subset.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        sns.lineplot(
            data=agg_subset, 
            x='Entropy', 
            y='Gap_Percent', 
            hue='Arch_Dims', 
            hue_order=hue_order,
            palette=palette,
            style='Arch_Dims', 
            markers=True, dashes=False, linewidth=2.5, markersize=8,
            ax=ax,
            legend=False
        )
        
        ax.set_title(f"Strategy: {strategy_names[i]}", fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        if i >= 2: ax.set_xlabel("Entropy Coefficient (Log Scale)", fontsize=12)
        else: ax.set_xlabel("")
        if i % 2 == 0: ax.set_ylabel("Mean Optimality Gap (%)", fontsize=12)
        else: ax.set_ylabel("")

    # Legend
    dummy_lines = []
    dummy_labels = []
    for label in hue_order:
        dummy_lines.append(mlines.Line2D([], [], color=palette[label], marker='o', linestyle='-', linewidth=2, markersize=8))
        dummy_labels.append(label)
        
    fig.legend(dummy_lines, dummy_labels, title="Model Architecture (Aggregated)", 
               loc='center left', bbox_to_anchor=(0.86, 0.5), fontsize=11, title_fontsize=12)
    
    plt.suptitle(f"Entropy Sensitivity (Aggregated over Experiments with >{int(threshold)} entropy points)", fontsize=18, fontweight='bold', y=0.98)
    plt.subplots_adjust(right=0.85)
    plt.savefig(OUTPUT_FILE_ENTROPY_GRID, dpi=300, bbox_inches='tight')
    print(f"Saved {OUTPUT_FILE_ENTROPY_GRID}")
    plt.show()

# --- PLOT 2: CLEAN CONFIDENCE CALIBRATION ---
def plot_confidence_calibration(df, hue_order, palette):
    plt.figure(figsize=(11, 7))
    sns.set_style("whitegrid")
    
    sns.scatterplot(
        data=df,
        x='Avg_Confidence',
        y='Gap_Percent',
        hue='Arch_Dims',
        hue_order=hue_order,
        palette=palette,
        size='Width',
        sizes=(50, 250),
        alpha=0.8,
        edgecolor='black'
    )
    
    plt.title("Confidence vs. Competence Calibration", fontsize=16, fontweight='bold')
    plt.ylabel("Optimality Gap (%) (Lower is Better)", fontsize=14, fontweight='bold')
    plt.xlabel("Average Model Confidence (Higher is Surer)", fontsize=14, fontweight='bold')
    
    plt.legend(title="Model & Strategy", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_CONFIDENCE, dpi=300, bbox_inches='tight')
    print(f"Saved {OUTPUT_FILE_CONFIDENCE}")
    plt.show()

# --- EXECUTION ---
if __name__ == "__main__":
    try:
        df = process_data(INPUT_FILE)
        hue_order, palette = get_elegant_palette(df)
        
        plot_entropy_sensitivity_grid(df, hue_order, palette)
        plot_confidence_calibration(df, hue_order, palette)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()