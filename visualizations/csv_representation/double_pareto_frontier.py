import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. CONFIGURATION ---
OUTPUT_FILE = "eval_pareto_frontier_comparison_TSP50.png"
CSV_FILE = 'results/evaluations/02experiment1_TSP50.csv' # Ensure this matches your path

# --- 2. DATA PROCESSING ---
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Get LKH Time for Speedup Calculation
    try:
        lkh_row = df[df['Model_Name'] == 'LKH_Baseline'].iloc[0]
        lkh_time = lkh_row['Time_Per_Solution']
    except IndexError:
        print("Warning: LKH Baseline not found. Using default 0.0037s")
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
    
    # Speedup Calculation
    df['Speedup_Factor'] = lkh_time / df['Time_Per_Solution']
    
    # Create unique ID for lines
    df['Model_ID'] = df['Experiment'] + "_" + df['Model_Category'] + "_" + df['Entropy'].astype(str)
    
    return df

# --- 3. PLOTTING SUBROUTINE ---
def draw_pareto_subplot(ax, df, gap_column, title, color_map, marker_map, width_map):
    neural_df = df[df['Model_Name'] != 'LKH_Baseline'].copy()
    
    # --- DRAW TRAJECTORIES ---
    for model_id in neural_df['Model_ID'].unique():
        subset = neural_df[neural_df['Model_ID'] == model_id].sort_values(by='Width')
        cat = subset['Model_Category'].iloc[0]
        col = color_map.get(cat, 'gray')
        
        # Plot Speedup (X) vs Gap (Y)
        ax.plot(subset['Speedup_Factor'], subset[gap_column], 
                color=col, alpha=0.4, linewidth=1.5, zorder=1)

    # --- DRAW POINTS ---
    for idx, row in neural_df.iterrows():
        cat = row['Model_Category']
        ent = row['Entropy']
        width = row['Width']
        
        ax.scatter(
            row['Speedup_Factor'], 
            row[gap_column], 
            c=[color_map[cat]], 
            marker=marker_map.get(ent, 'o'), 
            s=width_map.get(width, 100),
            edgecolor='white', linewidth=0.5, alpha=0.9, zorder=2
        )

    # --- DRAW LKH REFERENCE ---
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, zorder=0)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3, zorder=0)
    ax.scatter(1.0, 0, color='black', marker='*', s=400, zorder=3, label='LKH Baseline')
    
    # Axes Labels and Scaling
    ax.set_xscale('log')
    ax.set_xlabel("Speedup Factor (Log Scale)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Optimality Gap (%)", fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Ticks
    ticks = [0.1, 1, 10, 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}x" for t in ticks])
    
    # Highlight the "Collapse" Zone (Only on the Reality plot usually, but consistent axis helps)
    # We set a fixed Y-limit so both plots are comparable
    # You might want to adjust this based on your max gap (e.g., 150%)
    ax.set_ylim(-5, 160) 
    ax.grid(True, which="both", ls="-", alpha=0.2)

# --- 4. MAIN PLOTTING FUNCTION ---
def plot_comparison(df):
    # Setup Figure: 2 Subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    sns.set_style("whitegrid")
    
    neural_df = df[df['Model_Name'] != 'LKH_Baseline']

    # --- MAPPINGS ---
    unique_cats = sorted(neural_df['Model_Category'].unique())
    palette = sns.color_palette("bright", len(unique_cats))
    color_map = dict(zip(unique_cats, palette))

    unique_ents = sorted(neural_df['Entropy'].unique())
    markers = ['^', 's', 'D', 'p', 'o', 'H', '8']
    marker_map = {ent: markers[i % len(markers)] for i, ent in enumerate(unique_ents)}

    width_map = {0: 60, 10: 120, 100: 200, 1280: 350} # Slightly larger for visibility

    # --- PLOT 1: THE ILLUSION (Ratio of Means) ---
    draw_pareto_subplot(
        ax1, df, 
        gap_column='Gap_RoM', 
        title="1. THE ILLUSION (Training Metric)\nGap = Ratio of Means",
        color_map=color_map, marker_map=marker_map, width_map=width_map
    )
    # Add text annotation
    ax1.text(0.05, 0.95, "Appears Competitive\n(Learned $\\approx$ Hybrid)", 
             transform=ax1.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # --- PLOT 2: THE REALITY (Mean of Ratios) ---
    draw_pareto_subplot(
        ax2, df, 
        gap_column='Gap_MoR', 
        title="2. THE REALITY (Eval Metric)\nGap = Mean of Ratios", 
        color_map=color_map, marker_map=marker_map, width_map=width_map
    )
    # Add text annotation
    ax2.text(0.05, 0.95, "Massive Collapse\n(Learned Overfitting)", 
             transform=ax2.transAxes, fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

    # --- LEGENDS (Shared, placed on the right) ---
    # 1. Architecture (Color)
    leg_lines = [plt.Line2D([0], [0], color=color_map[c], marker='o', linestyle='', markersize=10) for c in unique_cats]
    leg1 = fig.legend(leg_lines, unique_cats, title="Architecture", loc='upper right', bbox_to_anchor=(0.98, 0.85), borderaxespad=0.)

    # 2. Entropy (Shape)
    shape_lines = [plt.Line2D([0], [0], color='gray', marker=marker_map[e], linestyle='', markersize=10) for e in unique_ents]
    leg2 = fig.legend(shape_lines, [f"Ent={e}" for e in unique_ents], title="Entropy", loc='upper right', bbox_to_anchor=(0.98, 0.65), borderaxespad=0.)
    
    # 3. Strategy (Size)
    size_lines = [plt.Line2D([0], [0], color='gray', marker='o', linestyle='', markersize=np.sqrt(s)/1.5) for s in width_map.values()]
    size_labels = ["Greedy", "BS-10", "BS-100", "BS-1280"]
    leg3 = fig.legend(size_lines, size_labels, title="Strategy", loc='upper right', bbox_to_anchor=(0.98, 0.45), borderaxespad=0.)

    # Adjust layout to fit legends
    plt.subplots_adjust(right=0.85, wspace=0.15)

    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Saved to {OUTPUT_FILE}")
    plt.show()

# --- RUN ---
if __name__ == "__main__":
    # Ensure you are pointing to the CSV that has the new columns!
    df = process_data(CSV_FILE)
    plot_comparison(df)