import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches

# --- 1. CONFIGURATION ---
INPUT_FILE = "/home/pfc/gms/code/rethinking_tsp/results/evaluations/02experiment1_TSP50.csv"
OUTPUT_FILE = "visualizations/csv_eval/TSP50_EXP1/eval_bar_TSP50.png"

# --- 2. DATA PROCESSING ---
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Get LKH Time
    try:
        lkh_row = df[df['Model_Name'] == 'LKH_Baseline']
        if not lkh_row.empty:
            lkh_time = lkh_row.iloc[0]['Time_Per_Solution']
        else:
            lkh_time = 0.0201 # Default adjusted to ~20ms
    except:
        lkh_time = 0.0201
        
    # Filter out LKH and Blank
    df = df[~df['Model_Name'].str.contains('LKH')].copy()
    df = df[~df['Model_Name'].str.contains('blank')].copy()

    def parse_row(row):
        name = row['Model_Name']
        parts = name.split('_')
        
        # Data Type (Density)
        data_type = "fully_connected" if "fully_connected" in name else "sparse"
        
        # Determine Architecture
        arch = 'Unknown'
        if 'coords' in parts: arch = 'Coords'
        elif 'learned' in parts: arch = 'Learned'
        elif 'hybrid' in parts: arch = 'Hybrid'
        
        # Get Dimensions
        dims = row['Feature_Dims']
        arch_dims = f"{arch} ({dims}d)"
        
        # Distribution
        dist = row.get('Distribution', 'in')
        
        return arch_dims, data_type, dist

    df[['Arch_Dims', 'Data_Type', 'Distribution']] = df.apply(
        lambda row: pd.Series(parse_row(row)), axis=1
    )
    
    # Calculate Speedup
    df['Time_Per_Solution'] = df['Time_Per_Solution'].replace(0, 1e-9)
    df['Speedup'] = lkh_time / df['Time_Per_Solution']
    
    # Create Strategy Labels for X-Axis
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
    
    # Combined Hue Grouping
    df['Hue_Group'] = df['Arch_Dims'] + "|" + df['Data_Type'] + "|" + df['Distribution']
    
    return df

# --- 3. PLOTTING ---
def plot_robustness_bar(df):
    plt.figure(figsize=(16, 9))
    sns.set_style("whitegrid")
    
    # --- HUE SORTING ---
    unique_hues = sorted(df['Hue_Group'].unique())
    
    def sort_key(s):
        parts = s.split('|')
        arch_dims = parts[0]
        dtype = parts[1]
        dist = parts[2]
        
        # Priority: Coords=0, Learned=1, Hybrid=2
        if "Coords" in arch_dims: prio = 0
        elif "Learned" in arch_dims: prio = 1
        elif "Hybrid" in arch_dims: prio = 2
        else: prio = 3
        
        try: d = int(arch_dims.split('(')[1].split('d')[0])
        except: d = 0
            
        dens_prio = 0 if dtype == 'sparse' else 1
        dist_prio = 0 if dist == 'in' else 1
        
        return (prio, d, dens_prio, dist_prio)
        
    hue_order = sorted(unique_hues, key=sort_key)
    
    # --- BAR PLOT ---
    ax = sns.barplot(
        data=df,
        x='Strategy_Label',
        y='Gap_MoR',       # Targeting Gap_MoR as requested
        hue='Hue_Group',
        hue_order=hue_order,
        ci='sd',
        capsize=0.05,
        errwidth=1.0,
        edgecolor='black'
    )
    
    # --- APPLY COLORS, SHADES & STYLES ---
    base_colormaps = {
        'Coords': plt.get_cmap("Purples"),
        'Learned': plt.get_cmap("Oranges"),
        'Hybrid': plt.get_cmap("Greens")
    }

    num_x_cats = len(df['Strategy_Label'].unique())
    
    for i, patch in enumerate(ax.patches):
        hue_idx = i // num_x_cats
        
        if hue_idx >= len(hue_order) or patch.get_height() == 0 or pd.isna(patch.get_height()):
            continue
            
        config = hue_order[hue_idx]
        arch_dims, data_type, dist = config.split('|')
        
        # Color Map
        cmap = next((v for k, v in base_colormaps.items() if k in arch_dims), plt.get_cmap("Greys"))
        
        # Saturation
        color = cmap(0.4) if data_type == 'sparse' else cmap(0.8)
        patch.set_facecolor(color)
        
        # Distribution Hatching
        if dist == 'out':
            patch.set_hatch('///')
        else:
            patch.set_hatch('')

    # --- AXES & LABELS ---
    plt.title("Robustness & Efficiency Summary\n(Gap vs. Speedup relative to LKH)", fontsize=16, fontweight='bold')
    plt.ylabel("Optimality Gap MoR (%)", fontsize=14, fontweight='bold')
    plt.xlabel("Decoding Strategy & Speedup Factor", fontsize=14, fontweight='bold')
    
    # Remove default combined legend
    ax.get_legend().remove()
    
    # --- CUSTOM LEGENDS ---
    unique_archs = sorted(df['Arch_Dims'].unique(), key=lambda s: sort_key(s + "|sparse|in"))
    
    arch_handles = []
    for arch in unique_archs:
        cmap = next((v for k, v in base_colormaps.items() if k in arch), plt.get_cmap("Greys"))
        arch_handles.append(mpatches.Patch(color=cmap(0.8), label=f"{arch} (Full)"))
        arch_handles.append(mpatches.Patch(color=cmap(0.4), label=f"{arch} (Sparse)"))
        
    leg1 = plt.legend(handles=arch_handles, title="Architecture & Density", loc='upper left', bbox_to_anchor=(1.02, 1.0))
    plt.gca().add_artist(leg1)

    in_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='In-Distribution (ID)')
    out_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Out-of-Distribution (OOD)')
    plt.legend(handles=[in_patch, out_patch], title="Generalization", loc='upper left', bbox_to_anchor=(1.02, 0.45))

    plt.subplots_adjust(right=0.82)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {OUTPUT_FILE}")
    plt.show()

# --- EXECUTION ---
if __name__ == "__main__":
    df = process_data(INPUT_FILE)
    plot_robustness_bar(df)