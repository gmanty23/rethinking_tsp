import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches

# --- 1. CONFIGURATION ---
INPUT_FILE = "results/evaluations/08gnn_disconnected_no.csv"
OUTPUT_FILE = "visualizations/csv_eval/MLPEXP/eval_bar_TSP20_none.png"


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
        
        # Find the index of the entropy coefficient to dynamically parse
        ent_idx = next((i for i, p in enumerate(parts) if p.startswith('ent')), -1)
        
        if ent_idx != -1 and len(parts) > ent_idx + 1:
            connection = parts[ent_idx + 1] # e.g., fullyConnected, random20, knn20
            feature_type = parts[ent_idx - 1].capitalize()
            msg_passing = "_".join(parts[1:ent_idx-1]) # forward, concat_dual, etc.
        else:
            connection = "Unknown"
            feature_type = "Unknown"
            msg_passing = "Unknown"
            
        # New Arch Grouping: Feature_Type + Dims + Connection
        dims = row.get('Feature_Dims', 0)
        arch_group = f"{feature_type} {connection}"
        
        return arch_group, msg_passing

    df[['Arch_Group', 'Msg_Passing']] = df.apply(
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
    df['Hue_Group'] = df['Arch_Group'] + "|" + df['Msg_Passing']
    
    return df

# --- 3. PLOTTING ---
def plot_robustness_bar(df):
    plt.figure(figsize=(16, 9))
    sns.set_style("whitegrid")
    
    unique_hues = sorted(df['Hue_Group'].unique())
    msg_prio_map = {'forward': 0, 'backward': 1, 'concat_dual': 2, 'gated_dual': 3, 'sum_dual': 4}
    
    def arch_sort_key(arch_group):
        # Priority: Coords=0, Learned=1, Hybrid=2
        if "Coords" in arch_group: prio = 0
        elif "Learned" in arch_group: prio = 1
        elif "Hybrid" in arch_group: prio = 2
        else: prio = 3
        
        # Dimensions
        try: d = int(arch_group.split('(')[1].split('d')[0])
        except: d = 0
            
        # Connection Priority
        if 'fullyConnected' in arch_group: c_prio = 0
        elif 'random20' in arch_group: c_prio = 1
        elif 'knn20' in arch_group: c_prio = 2
        else: c_prio = 3
            
        return (prio, d, c_prio)

    def hue_sort_key(s):
        arch_group, msg = s.split('|')
        prio, d, c_prio = arch_sort_key(arch_group)
        msg_prio = msg_prio_map.get(msg, 5)
        return (prio, d, c_prio, msg_prio)
        
    hue_order = sorted(unique_hues, key=hue_sort_key)
    
    # --- BAR PLOT ---
    ax = sns.barplot(
        data=df,
        x='Strategy_Label',
        y='Gap_MoR',
        hue='Hue_Group',
        hue_order=hue_order,
        ci='sd',
        capsize=0.05,
        errcolor='red',     # <--- High contrast color for error bars
        errwidth=1.2,       # <--- Slightly thicker so the red pops
        edgecolor='black',
        linewidth=0.5,      # Thinner bar borders
        width=0.92          # Wider bars
    )
    
    # --- APPLY COLORS & SHADES ---
    base_colormaps = {
        'Coords': plt.get_cmap("Purples"),
        'Learned': plt.get_cmap("Oranges"),
        'Hybrid': plt.get_cmap("Greens")
    }

    # Structurally distinct hatching dictionary
    hatch_map = {
        'forward': '',         # Solid fill
        'backward': '...',     # Dots
        'concat_dual': 'xxx',  # Crosses
        'gated_dual': 'OOO',   # Large circles
        'sum_dual': '***'      # Stars
    }

    num_x_cats = len(df['Strategy_Label'].unique())
    
    for i, patch in enumerate(ax.patches):
        hue_idx = i // num_x_cats
        
        if hue_idx >= len(hue_order) or patch.get_height() == 0 or pd.isna(patch.get_height()):
            continue
            
        config = hue_order[hue_idx]
        arch_group, msg = config.split('|')
        
        # 1. Get Base Color
        cmap = next((v for k, v in base_colormaps.items() if k in arch_group), plt.get_cmap("Greys"))
        
        # 2. Assign 3 distinct shades based on Connection type
        if 'fullyConnected' in arch_group:
            alpha = 0.9  # Dark
        elif 'random20' in arch_group:
            alpha = 0.55 # Medium
        elif 'knn20' in arch_group:
            alpha = 0.2  # Light
        else:
            alpha = 0.5
            
        patch.set_facecolor(cmap(alpha))
        patch.set_hatch(hatch_map.get(msg, ''))
        patch.set_linewidth(0.5)

    # --- AXES & LABELS ---
    plt.title("TSP20\n(Windy TSP Optimality Gap vs. LKH Speedup)", fontsize=16, fontweight='bold')
    plt.ylabel("Optimality Gap (%)", fontsize=14, fontweight='bold')
    plt.xlabel("Decoding Strategy & Speedup Factor", fontsize=14, fontweight='bold')
    
    # Remove default combined legend
    ax.get_legend().remove()
    
    # --- CUSTOM LEGENDS ---
    unique_archs = sorted(df['Arch_Group'].unique(), key=arch_sort_key)
    
    # 1. Architecture Backbone Legend
    arch_handles = []
    for arch in unique_archs:
        cmap = next((v for k, v in base_colormaps.items() if k in arch), plt.get_cmap("Greys"))
        
        if 'fullyConnected' in arch:
            alpha = 0.9
        elif 'random20' in arch:
            alpha = 0.55
        elif 'knn20' in arch:
            alpha = 0.2
        else:
            alpha = 0.5
            
        arch_handles.append(mpatches.Patch(
            facecolor=cmap(alpha), 
            edgecolor='black', 
            linewidth=0.5,
            label=arch
        ))
        
    leg1 = plt.legend(handles=arch_handles, title="Backbone (Features, Dims, Topology)", loc='upper left', bbox_to_anchor=(1.02, 1.0))
    plt.gca().add_artist(leg1)

    # 2. Message Passing Legend
    msg_handles = []
    existing_msgs = sorted(df['Msg_Passing'].unique(), key=lambda m: msg_prio_map.get(m, 5))
    
    for msg in existing_msgs:
        label_name = msg.replace('_', ' ').title()
        msg_handles.append(
            mpatches.Patch(
                facecolor='white', 
                edgecolor='black', 
                hatch=hatch_map.get(msg, ''), 
                linewidth=0.5,
                label=label_name
            )
        )
        
    leg2 = plt.legend(handles=msg_handles, title="Message Passing Scheme", loc='upper left', bbox_to_anchor=(1.02, 0.45))

    plt.subplots_adjust(right=0.72)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', bbox_extra_artists=(leg1, leg2))
    print(f"Chart saved to {OUTPUT_FILE}")
    plt.show()

# --- EXECUTION ---
if __name__ == "__main__":
    df = process_data(INPUT_FILE)
    plot_robustness_bar(df)