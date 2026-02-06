import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. CONFIGURATION ---
OUTPUT_FILE = "eval_pareto_frontier_TSP50_exp1.png"

# --- 2. DATA PROCESSING ---
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Get LKH Time for Speedup Calculation
    try:
        lkh_time = df[df['Model_Name'] == 'LKH_Baseline']['Time_Per_Solution'].iloc[0]
    except IndexError:
        # Fallback if LKH missing, use an estimated baseline or raise error
        print("Warning: LKH Baseline not found for speedup calc. Using 0.0037s")
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
        
        return exp_name, ftype, entropy, f"{ftype.capitalize()} ({row['Feature_Dims']}d)"

    # Apply Parsing
    df[['Experiment', 'Feature_Type', 'Entropy', 'Model_Category']] = df.apply(
        lambda row: pd.Series(parse_row(row)), axis=1
    )
    
    # Create Speedup Column (Higher is Better)
    # Speedup = Baseline / Model
    df['Speedup_Factor'] = lkh_time / df['Time_Per_Solution']
    
    # Create unique ID for lines
    df['Model_ID'] = df['Experiment'] + "_" + df['Model_Category'] + "_" + df['Entropy'].astype(str)
    
    return df

# --- 3. PLOTTING ---
def plot_pareto(df):
    neural_df = df[df['Model_Name'] != 'LKH_Baseline'].copy()
    
    # Setup Figure with extra space on right for legends
    plt.figure(figsize=(16, 10)) # Wider figure
    sns.set_style("whitegrid")

    # --- MAPPINGS ---
    unique_cats = sorted(neural_df['Model_Category'].unique())
    palette = sns.color_palette("bright", len(unique_cats))
    color_map = dict(zip(unique_cats, palette))

    unique_ents = sorted(neural_df['Entropy'].unique())
    markers = ['^', 's', 'D', 'p', 'o', 'H', '8']
    marker_map = {ent: markers[i % len(markers)] for i, ent in enumerate(unique_ents)}

    width_map = {0: 50, 10: 100, 100: 180, 1280: 300}
    
    # --- DRAW TRAJECTORIES ---
    print("Drawing trajectories...")
    for model_id in neural_df['Model_ID'].unique():
        subset = neural_df[neural_df['Model_ID'] == model_id].sort_values(by='Width')
        
        # Color
        cat = subset['Model_Category'].iloc[0]
        col = color_map.get(cat, 'gray')
        
        # Plot Speedup (X) vs Gap (Y)
        plt.plot(subset['Speedup_Factor'], subset['Gap_Percent'], 
                 color=col, alpha=0.4, linewidth=1.5, zorder=1)

    # --- DRAW POINTS ---
    print("Drawing points...")
    for idx, row in neural_df.iterrows():
        cat = row['Model_Category']
        ent = row['Entropy']
        width = row['Width']
        
        plt.scatter(
            row['Speedup_Factor'], 
            row['Gap_Percent'], 
            c=[color_map[cat]], 
            marker=marker_map.get(ent, 'o'), 
            s=width_map.get(width, 100),
            edgecolor='white', linewidth=0.5, alpha=0.9, zorder=2
        )

    # --- DRAW LKH REFERENCE ---
    # LKH Speedup is exactly 1.0
    plt.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, label='LKH Baseline Speed', zorder=0)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3, zorder=0) # Zero gap line
    
    # Mark LKH point
    plt.scatter(1.0, 0, color='black', marker='*', s=500, label='LKH Solver', zorder=3)
    plt.text(1.1, 0.5, "LKH Baseline (20.3ms)", verticalalignment='bottom', fontsize=10, fontweight='bold')

    # --- AXES & LABELS ---
    plt.xscale('log')
    
    # Invert X axis? 
    # Usually: Left=Slow, Right=Fast. 
    # Current: Speedup 0.1 (Left), Speedup 100 (Right). This is intuitive. 
    # Ideal corner is BOTTOM-RIGHT.
    
    plt.xlabel("Speedup Factor relative to LKH (Log Scale)", fontsize=12, fontweight='bold')
    plt.ylabel("Optimality Gap (%)", fontsize=12, fontweight='bold')
    plt.title("Pareto Frontier: Windy TSP Efficiency vs Optimality for TSP50", fontsize=16)

    # Custom Ticks for Speedup intuition
    ticks = [0.1, 0.5, 1, 10, 50, 100]
    plt.xticks(ticks, [f"{t}x" for t in ticks])

    # --- LEGENDS OUTSIDE ---
    # We place legends to the right of the plot (bbox_to_anchor > 1)
    
    # 1. Architecture (Color)
    leg_lines = [plt.Line2D([0], [0], color=color_map[c], marker='o', linestyle='', markersize=10) for c in unique_cats]
    leg1 = plt.legend(leg_lines, unique_cats, title="Architecture", loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    plt.gca().add_artist(leg1)

    # 2. Entropy (Shape)
    shape_lines = [plt.Line2D([0], [0], color='gray', marker=marker_map[e], linestyle='', markersize=10) for e in unique_ents]
    leg2 = plt.legend(shape_lines, [f"Ent={e}" for e in unique_ents], title="Entropy", loc='upper left', bbox_to_anchor=(1.02, 0.75), borderaxespad=0.)
    plt.gca().add_artist(leg2)
    
    # 3. Strategy (Size)
    size_lines = [plt.Line2D([0], [0], color='gray', marker='o', linestyle='', markersize=np.sqrt(s)/1.5) for s in width_map.values()]
    size_labels = ["Greedy", "BS-10", "BS-100", "BS-1280"]
    leg3 = plt.legend(size_lines, size_labels, title="Strategy", loc='upper left', bbox_to_anchor=(1.02, 0.50), borderaxespad=0.)

    # Adjust layout to make room for legends
    plt.subplots_adjust(right=0.8)

    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Saved to {OUTPUT_FILE}")
    plt.show()

# --- RUN ---
if __name__ == "__main__":
    df = process_data('results/evaluations/02experiment1_TSP50.csv')
    plot_pareto(df)