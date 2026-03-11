import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.lines as mlines

# --- CONFIGURATION ---
INPUT_FILE = "results/evaluations/03experiment1_compknn_TSP20.csv"
OUTPUT_FILE_ENTROPY_GRID = "visualizations/csv_eval/TSP20_EXP3/eval_analysis_entropy_sensitivity_TSP20_CONNECTIONS.png"
OUTPUT_FILE_CONFIDENCE = "visualizations/csv_eval/TSP20_EXP3/eval_analysis_confidence_calibration_TSP20_CONNECTIONS.png"

# --- DATA PROCESSING ---
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    
    def parse_row(row):
        name = row['Model_Name']
        if name == "LKH_Baseline":
            return "LKH", "baseline", 0.0, "LKH", "unknown"
        
        parts = name.split('_')
        exp_name = parts[0]
        
        # Extract new variables using entropy index
        ent_idx = next((i for i, p in enumerate(parts) if p.startswith('ent')), -1)
        if ent_idx != -1 and len(parts) > ent_idx + 1:
            connection = parts[ent_idx + 1]
            ftype = parts[ent_idx - 1].capitalize()
            msg_passing = "_".join(parts[1:ent_idx-1])
            try:
                entropy = float(parts[ent_idx].replace('ent', ''))
            except:
                entropy = 0.0
        else:
            connection = "Unknown"
            ftype = "Unknown"
            msg_passing = "Unknown"
            entropy = 0.0
        
        # Create "Architecture (Nd) Connection" Label
        dims = row.get('Feature_Dims', 0)
        arch_group = f"{ftype} ({dims}d) {connection}"
        
        return exp_name, ftype, entropy, arch_group, msg_passing

    df[['Experiment', 'Feature_Type', 'Entropy', 'Arch_Group', 'Msg_Passing']] = df.apply(
        lambda row: pd.Series(parse_row(row)), axis=1
    )
    
    # Filter out LKH and Blank models
    df = df[~df['Model_Name'].isin(['LKH_Baseline']) & (df['Feature_Type'] != 'blank')].copy()
    
    return df

# --- COLOR PALETTE GENERATION ---
def get_elegant_palette(df):
    """
    Generates the 'Swapped Elegant' palette:
    - Coords: Purples | Learned: Oranges | Hybrid: Greens
    - fullyConnected (0.9) | random20 (0.55) | knn20 (0.2)
    """
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

# --- PLOT 1: ROBUST ENTROPY SENSITIVITY GRID ---
def plot_entropy_sensitivity_grid(df, hue_order, palette):
    strategies = [0, 10, 100, 1280]
    strategy_names = ["Greedy (0)", "BS-10", "BS-100", "BS-1280"]
    
    # --- FILTERING LOGIC ---
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
        
        # 2. AGGREGATE: Group by Arch+Connection, Msg_Passing, and Entropy
        agg_subset = subset.groupby(['Arch_Group', 'Msg_Passing', 'Entropy'], as_index=False)['Gap_MoR'].mean()
        
        if agg_subset.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        sns.lineplot(
            data=agg_subset, 
            x='Entropy', 
            y='Gap_MoR', 
            hue='Arch_Group', 
            hue_order=hue_order,
            palette=palette,
            style='Msg_Passing', # ADDED: Map style to Message Passing
            markers=True, dashes=False, linewidth=2.5, markersize=8,
            ax=ax,
            legend=False
        )
        
        ax.set_title(f"Strategy: {strategy_names[i]}", fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        if i >= 2: ax.set_xlabel("Entropy Coefficient (Log Scale)", fontsize=12)
        else: ax.set_xlabel("")
        if i % 2 == 0: ax.set_ylabel("Mean Optimality Gap MoR (%)", fontsize=12) # UPDATED LABEL
        else: ax.set_ylabel("")

    # Legend
    dummy_lines = []
    dummy_labels = []
    for label in hue_order:
        dummy_lines.append(mlines.Line2D([], [], color=palette[label], marker='o', linestyle='-', linewidth=2, markersize=8))
        dummy_labels.append(label)
        
    leg1 = fig.legend(dummy_lines, dummy_labels, title="Model Architecture (Aggregated)", 
               loc='center left', bbox_to_anchor=(0.86, 0.65), fontsize=11, title_fontsize=12)
    
    # ADDED: Secondary Legend for Message Passing
    msg_lines = []
    msg_labels = []
    existing_msgs = sorted(df_filtered['Msg_Passing'].unique())
    for msg in existing_msgs:
        # Default Seaborn markers: 'o', 'X', 's', 'P', 'D', etc. 
        # We just use black proxy lines so the user knows what shapes mean
        msg_lines.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='', markersize=8))
        msg_labels.append(msg)
    
    fig.legend(msg_lines, msg_labels, title="Message Passing", 
               loc='center left', bbox_to_anchor=(0.86, 0.35), fontsize=11, title_fontsize=12)
    
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
        y='Gap_MoR', # UPDATED
        hue='Arch_Group',
        hue_order=hue_order,
        palette=palette,
        style='Msg_Passing', # ADDED: Map style to Message Passing
        size='Width',
        sizes=(50, 250),
        alpha=0.8,
        edgecolor='black'
    )
    
    plt.title("Confidence vs. Competence Calibration", fontsize=16, fontweight='bold')
    plt.ylabel("Optimality Gap MoR (%) (Lower is Better)", fontsize=14, fontweight='bold') # UPDATED LABEL
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