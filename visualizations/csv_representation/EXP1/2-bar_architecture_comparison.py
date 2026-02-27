import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. CONFIGURATION ---
INPUT_FILE = "/home/pfc/gms/code/rethinking_tsp/results/evaluations/01experiment1ONLY56.csv"
OUTPUT_FILE = "visualizations/csv_eval/TSP20_EXP1/bar_robustness_ONLY56.png"

# --- 2. DATA PROCESSING ---
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Get LKH Time
    try:
        lkh_row = df[df['Model_Name'] == 'LKH_Baseline']
        if not lkh_row.empty:
            lkh_time = lkh_row.iloc[0]['Time_Per_Solution']
        else:
            lkh_time = 0.0037
    except:
        lkh_time = 0.0037
        
    # Filter out LKH and Blank
    df = df[~df['Model_Name'].str.contains('LKH')].copy()
    df = df[~df['Model_Name'].str.contains('blank')].copy()

    # Parse Architecture and Dims
    def parse_arch_dims(row):
        name = row['Model_Name']
        parts = name.split('_')
        
        # Determine Architecture
        arch = 'Unknown'
        if 'coords' in parts: arch = 'Coords'
        elif 'learned' in parts: arch = 'Learned'
        elif 'hybrid' in parts: arch = 'Hybrid'
        
        # Get Dimensions
        dims = row['Feature_Dims']
        
        return f"{arch} ({dims}d)"

    df['Arch_Dims'] = df.apply(parse_arch_dims, axis=1)
    
    # Calculate Speedup
    df['Time_Per_Solution'] = df['Time_Per_Solution'].replace(0, 1e-9)
    df['Speedup'] = lkh_time / df['Time_Per_Solution']
    
    # Create Strategy Labels
    speedup_map = df.groupby('Width')['Speedup'].mean().to_dict()
    
    def get_label(width):
        speedup = speedup_map.get(width, 1)
        if speedup >= 10:
            val_str = f"{int(speedup)}x"
        elif speedup >= 1:
            val_str = f"{speedup:.1f}x"
        else:
            val_str = f"{speedup:.2f}x"
            
        if width == 0: return f"Greedy\n({val_str} Faster)"
        return f"Beam-{width}\n({val_str} Faster)"

    df['Strategy_Label'] = df['Width'].apply(get_label)
    
    # Sort X-Axis by Width (0 -> 1280)
    width_sort = {0: 0, 10: 1, 100: 2, 1280: 3}
    df['Sort_Index'] = df['Width'].map(width_sort)
    df = df.sort_values(by='Sort_Index')
    
    return df

# --- 3. PLOTTING ---
def plot_robustness_bar(df):
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    # Define Sort Order for the Legend/Hue
    unique_labels = sorted(df['Arch_Dims'].unique())
    
    # Sort Logic: Coords -> Learned -> Hybrid, then by Dims
    def sort_key(s):
        # Priority: Coords=0, Learned=1, Hybrid=2
        if "Coords" in s: prio = 0
        elif "Learned" in s: prio = 1
        elif "Hybrid" in s: prio = 2
        else: prio = 3
        
        # Dimensions
        try:
            d = int(s.split('(')[1].split('d')[0])
        except:
            d = 0
        return (prio, d)
        
    hue_order = sorted(unique_labels, key=sort_key)
    
    # --- SWAPPED ELEGANT PALETTE GENERATION ---
    palette = {}

    # Group labels by base architecture
    groups = {'Coords': [], 'Learned': [], 'Hybrid': []}
    for label in hue_order:
        for key in groups:
            if key in label:
                groups[key].append(label)
    
    # Assign Gradients - SWAPPED: Hybrid=Green, Coords=Purple
    
    # Coords = Purples (Was Greens)
    for i, label in enumerate(groups['Coords']):
        palette[label] = plt.get_cmap("Purples")(np.linspace(0.5, 0.9, len(groups['Coords'])))[i]
        
    # Learned = Oranges (Unchanged)
    for i, label in enumerate(groups['Learned']):
        palette[label] = plt.get_cmap("Oranges")(np.linspace(0.5, 0.9, len(groups['Learned'])))[i]
        
    # Hybrid = Greens (Was Purples)
    for i, label in enumerate(groups['Hybrid']):
        palette[label] = plt.get_cmap("Greens")(np.linspace(0.5, 0.9, len(groups['Hybrid'])))[i]
    
    ax = sns.barplot(
        data=df,
        x='Strategy_Label',
        y='Gap_Percent',
        hue='Arch_Dims',
        hue_order=hue_order,
        palette=palette,
        ci='sd',          # Standard Deviation Error Bars
        capsize=0.1,      # Caps on error bars
        errwidth=1.5,     # Thicker error bars
        edgecolor='black',
        alpha=0.95        # Slight transparency
    )
    
    # Title and Labels
    plt.title("Robustness & Efficiency Summary\n(Gap vs. Speedup relative to LKH)", fontsize=16, fontweight='bold')
    plt.ylabel("Optimality Gap (%)", fontsize=14, fontweight='bold')
    plt.xlabel("Decoding Strategy & Speedup Factor", fontsize=14, fontweight='bold')
    
    # Legend
    plt.legend(title="Model (Arch + Dims)", title_fontsize='12', fontsize='11', 
               loc='upper right', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Chart saved to {OUTPUT_FILE}")
    plt.show()

# --- EXECUTION ---
if __name__ == "__main__":
    df = process_data(INPUT_FILE)
    plot_robustness_bar(df)