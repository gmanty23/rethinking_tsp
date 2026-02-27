import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = "/home/pfc/gms/code/rethinking_tsp/results/evaluations/01experiment1NoBlank_TSP2020.csv"
OUTPUT_FILE_BOX = "visualizations/csv_eval/TSP20_EXP1/eval_analysis_robustness_box_colored.png"


# --- DATA PROCESSING ---
# --- DATA PROCESSING ---
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    
    def parse_row(row):
        name = row['Model_Name']
        
        # --- THE FIX IS HERE ---
        if name == "LKH_Baseline":
            # Must return exactly 2 values to match ['Feature_Type', 'Arch_Dims']
            return "baseline", "LKH"
        
        parts = name.split('_')
        ftype = "unknown"
        for t in ['blank', 'coords', 'hybrid', 'learned']:
            if t in parts:
                ftype = t
                break
        
        # Create "Architecture (Nd)" Label
        arch_base = ftype.capitalize()
        arch_dims_label = f"{arch_base} ({row['Feature_Dims']}d)"
        
        return ftype, arch_dims_label

    # Now the assignment works because parse_row always returns 2 values
    df[['Feature_Type', 'Arch_Dims']] = df.apply(
        lambda row: pd.Series(parse_row(row)), axis=1
    )
    
    # Filter out LKH and Blank models
    df = df[~df['Model_Name'].isin(['LKH_Baseline']) & (df['Feature_Type'] != 'blank')].copy()
    
    return df

# --- PALETTE GENERATION (Same as before) ---
def get_elegant_palette(df):
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

# --- PLOT: ROBUSTNESS BOX PLOT ---
def plot_robustness_box(df, hue_order, palette):
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Filter for BS-100 only (Standard Strategy) to isolate entropy effects
    subset = df[df['Width'] == 100].copy()
    
    # Box Plot showing the distribution of gaps across ALL entropies
    sns.boxplot(
        data=subset,
        x='Arch_Dims',
        y='Gap_Percent',
        order=hue_order,
        palette=palette,
        linewidth=1.5,
        width=0.6,
        fliersize=0 # Hide outliers in boxplot, we show them in stripplot
    )
    
    # Strip Plot (Dots) to show the actual entropy runs
    sns.stripplot(
        data=subset,
        x='Arch_Dims',
        y='Gap_Percent',
        order=hue_order,
        color='black',
        size=6,
        alpha=0.6,
        jitter=True
    )
    
    plt.title("Robustness Analysis: Performance Variation across Entropy Coefficients\n(Strategy: BS-100, Dots = Different Entropies)", fontsize=14, fontweight='bold')
    plt.ylabel("Optimality Gap (%) (Lower & More Compact is Better)", fontsize=12, fontweight='bold')
    plt.xlabel("Model Architecture", fontsize=12, fontweight='bold')
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_BOX, dpi=300)
    print(f"Saved {OUTPUT_FILE_BOX}")
    plt.show()

if __name__ == "__main__":
    df = process_data(INPUT_FILE)
    hue_order, palette = get_elegant_palette(df)
    plot_robustness_box(df, hue_order, palette)