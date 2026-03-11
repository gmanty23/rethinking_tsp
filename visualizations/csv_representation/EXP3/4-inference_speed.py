import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# --- CONFIGURATION ---
INPUT_FILE = "results/evaluations/03experiment1_compknn_TSP20.csv"
OUTPUT_FILE = "visualizations/csv_eval/TSP20_EXP3/eval_inference_speedup_TSP20_CONNECTIONS.png"

# --- DATA PROCESSING ---
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # 1. Extract LKH Time for Speedup Calculation
    try:
        lkh_row = df[df['Model_Name'].str.contains('LKH')].iloc[0]
        lkh_time = lkh_row['Time_Per_Solution']
    except:
        lkh_time = 0.0201 # Fallback default
        
    def parse_row(row):
        name = row['Model_Name']
        if "LKH" in name:
            # FIXED: Now strictly returns 4 elements to match the dataframe assignment
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
    
    # Filter out LKH and Blank models
    df = df[~df['Model_Name'].isin(['LKH_Baseline']) & (df['Feature_Type'] != 'blank')].copy()
    
    # Calculate Speedup
    df['Time_Per_Solution'] = df['Time_Per_Solution'].replace(0, 1e-9) # Prevent Div/0
    df['Speedup'] = lkh_time / df['Time_Per_Solution']
    
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

# --- PLOTTING ---
def plot_speedup(df, hue_order, palette):
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(
        data=df,
        x='Strategy',
        y='Speedup',
        hue='Arch_Group',
        hue_order=hue_order,
        palette=palette,
        ci='sd',
        capsize=0.05,
        errwidth=1.2,
        errcolor='red',
        edgecolor='black',
        linewidth=0.5,
        width=0.85
    )
    
    # Log scale is vital because Greedy might be 500x and Beam-1280 might be 0.1x
    ax.set_yscale("log")
    
    # Draw LKH Baseline at exactly 1.0
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2.5, zorder=0)
    
    # Formatting Y-Axis ticks to show "X" multiplier easily
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}x'))
    
    # --- AXES & LABELS ---
    plt.title("Inference Speedup vs. LKH Baseline\n(Evaluating Sparse Topological Scaling)", fontsize=16, fontweight='bold')
    plt.ylabel("Speedup Multiplier (Log Scale)", fontsize=14, fontweight='bold')
    plt.xlabel("Decoding Strategy (Beam Width)", fontsize=14, fontweight='bold')
    
    # Remove default combined legend
    ax.get_legend().remove()
    
    # Architecture Legend
    arch_handles = []
    for arch in hue_order:
        arch_handles.append(mpatches.Patch(facecolor=palette[arch], edgecolor='black', linewidth=0.5, label=arch))
    
    # Add LKH to Legend
    arch_handles.append(mlines.Line2D([], [], color='red', linestyle='--', linewidth=2.5, label='LKH (1x Speed)'))
        
    plt.legend(handles=arch_handles, title="Topological Backbone", loc='upper left', bbox_to_anchor=(1.02, 1.0))
    
    plt.subplots_adjust(right=0.75)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {OUTPUT_FILE}")
    plt.show()

# --- EXECUTION ---
if __name__ == "__main__":
    df = process_data(INPUT_FILE)
    hue_order, palette = get_elegant_palette(df)
    plot_speedup(df, hue_order, palette)