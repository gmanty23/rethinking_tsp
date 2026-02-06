import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. CONFIGURATION ---
INPUT_FILE = "results/evaluations/01experiment1NoBlank.csv"
OUTPUT_FILE = "eval_pareto_frontier_TSP20_peak_performance.png"

# --- 2. DATA PROCESSING ---
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Get LKH Time
    try:
        lkh_time = df[df['Model_Name'] == 'LKH_Baseline']['Time_Per_Solution'].iloc[0]
    except IndexError:
        print("Warning: LKH Baseline not found. Using 0.0037s")
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
    
    # Calculate Speedup
    df['Speedup_Factor'] = lkh_time / df['Time_Per_Solution']
    
    return df

def get_peak_performance(df):
    """
    Selects the single best run (lowest Gap) for each Architecture + Strategy combination,
    discarding suboptimal entropy settings.
    """
    neural_df = df[df['Model_Name'] != 'LKH_Baseline'].copy()
    
    # Find the index of the minimum Gap for each Category + Width
    # This automatically picks the best Entropy for that specific configuration
    best_indices = neural_df.groupby(['Model_Category', 'Width'])['Gap_Percent'].idxmin()
    
    peak_df = neural_df.loc[best_indices].copy()
    
    return peak_df.sort_values(by=['Model_Category', 'Width'])

# --- 3. PLOTTING ---
def plot_peak_pareto(df, peak_df):
    plt.figure(figsize=(16, 10))
    sns.set_style("whitegrid")

    # --- MAPPINGS ---
    unique_cats = sorted(peak_df['Model_Category'].unique())
    palette = sns.color_palette("bright", len(unique_cats))
    color_map = dict(zip(unique_cats, palette))

    # Size mapping
    width_map = {0: 100, 10: 180, 100: 280, 1280: 450}
    
    # Markers for entropy (to show WHICH entropy was selected as best)
    unique_ents = sorted(df['Entropy'].unique())
    markers = ['^', 's', 'D', 'p', 'o', 'H', '8']
    marker_map = {ent: markers[i % len(markers)] for i, ent in enumerate(unique_ents)}

    # --- DRAW TRAJECTORIES ---
    print("Drawing best-case trajectories...")
    for cat in unique_cats:
        subset = peak_df[peak_df['Model_Category'] == cat].sort_values(by='Width')
        col = color_map.get(cat, 'gray')
        
        plt.plot(subset['Speedup_Factor'], subset['Gap_Percent'], 
                 color=col, alpha=0.6, linewidth=2.5, zorder=1)

    # --- DRAW POINTS ---
    print("Drawing points...")
    for idx, row in peak_df.iterrows():
        cat = row['Model_Category']
        ent = row['Entropy']
        width = row['Width']
        
        plt.scatter(
            row['Speedup_Factor'], 
            row['Gap_Percent'], 
            c=[color_map[cat]], 
            marker=marker_map.get(ent, 'o'), 
            s=width_map.get(width, 100),
            edgecolor='white', linewidth=1.5, alpha=1.0, zorder=2
        )

    # --- DRAW LKH REFERENCE ---
    plt.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, label='LKH Baseline Speed', zorder=0)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3, zorder=0)
    plt.scatter(1.0, 0, color='black', marker='*', s=500, label='LKH Solver', zorder=4)
    plt.text(1.1, 0.5, "LKH Baseline (3.7ms)", verticalalignment='bottom', fontsize=10, fontweight='bold')

    # --- AXES & LABELS ---
    plt.xscale('log')
    plt.xlabel("Speedup Factor vs LKH (Log Scale)", fontsize=12, fontweight='bold')
    plt.ylabel("Best Optimality Gap (%)", fontsize=12, fontweight='bold')
    plt.title("Pareto Frontier: Peak Performance by Architecture for TSP20\n(Best Entropy Selected for Each Strategy)", fontsize=16)

    # Custom Ticks
    ticks = [0.1, 0.5, 1, 10, 50, 100]
    plt.xticks(ticks, [f"{t}x" for t in ticks])

    # --- LEGENDS ---
    # 1. Architecture
    leg_lines = [plt.Line2D([0], [0], color=color_map[c], marker='o', linestyle='', markersize=10) for c in unique_cats]
    leg1 = plt.legend(leg_lines, unique_cats, title="Architecture", loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    plt.gca().add_artist(leg1)

    # 2. Strategy
    size_lines = [plt.Line2D([0], [0], color='gray', marker='o', linestyle='', markersize=np.sqrt(s)/1.5) for s in width_map.values()]
    size_labels = ["Greedy", "BS-10", "BS-100", "BS-1280"]
    leg2 = plt.legend(size_lines, size_labels, title="Strategy", loc='upper left', bbox_to_anchor=(1.02, 0.75), borderaxespad=0.)
    plt.gca().add_artist(leg2)

    # 3. Selected Entropy (Informational)
    # This shows which entropy usually "wins"
    shape_lines = [plt.Line2D([0], [0], color='gray', marker=marker_map[e], linestyle='', markersize=10) for e in unique_ents if e in peak_df['Entropy'].values]
    existing_ents = sorted([e for e in unique_ents if e in peak_df['Entropy'].values])
    leg3 = plt.legend(shape_lines, [f"Ent={e}" for e in existing_ents], title="Winner Entropy", loc='upper left', bbox_to_anchor=(1.02, 0.50), borderaxespad=0.)

    plt.subplots_adjust(right=0.8)
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Saved to {OUTPUT_FILE}")
    plt.show()

# --- EXECUTION ---
if __name__ == "__main__":
    df = process_data(INPUT_FILE)
    peak_df = get_peak_performance(df)
    
    # Preview the winners
    print("\nPeak Performance Winners:")
    print(peak_df[['Model_Category', 'Width', 'Entropy', 'Gap_Percent', 'Speedup_Factor']].head(10))
    
    plot_peak_pareto(df, peak_df)