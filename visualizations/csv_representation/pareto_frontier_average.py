import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. CONFIGURATION ---
INPUT_FILE = "results/evaluations/01experiment1NoBlank.csv"
OUTPUT_FILE = "eval_pareto_frontier_TSP20_averaged_with_errorbars.png"

# --- 2. DATA PROCESSING ---
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Get LKH Time for Speedup Calculation
    # We assume LKH is stable, taking the first one found
    try:
        lkh_time = df[df['Model_Name'] == 'LKH_Baseline']['Time_Per_Solution'].iloc[0]
    except IndexError:
        print("Warning: LKH Baseline not found. Using 0.0037s")
        lkh_time = 0.0037

    def parse_row(row):
        name = row['Model_Name']
        if name == "LKH_Baseline":
            return "LKH", "baseline", "LKH"
        
        parts = name.split('_')
        exp_name = parts[0]
        
        # Feature Type
        ftype = "unknown"
        for t in ['blank', 'coords', 'hybrid', 'learned']:
            if t in parts:
                ftype = t
                break
        
        return exp_name, ftype, f"{ftype.capitalize()} ({row['Feature_Dims']}d)"

    # Apply Parsing
    df[['Experiment', 'Feature_Type', 'Model_Category']] = df.apply(
        lambda row: pd.Series(parse_row(row)), axis=1
    )
    
    # Calculate Speedup Factor per instance
    df['Speedup_Factor'] = lkh_time / df['Time_Per_Solution']
    
    return df

def aggregate_data(df):
    # Filter out LKH for aggregation (we treat it separately)
    neural_df = df[df['Model_Name'] != 'LKH_Baseline'].copy()
    
    # Group by Architecture (Category) and Strategy (Width)
    # We average over 'Entropy' (and any other variations like dates)
    grouped = neural_df.groupby(['Model_Category', 'Width']).agg({
        'Gap_Percent': ['mean', 'std', 'count'],
        'Speedup_Factor': ['mean', 'std'],
        'Time_Per_Solution': 'mean'
    }).reset_index()
    
    # Flatten MultiIndex columns
    grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns.values]
    
    # Rename for clarity
    grouped.rename(columns={
        'Gap_Percent_mean': 'Gap_Mean',
        'Gap_Percent_std': 'Gap_Std',
        'Speedup_Factor_mean': 'Speedup_Mean',
        'Speedup_Factor_std': 'Speedup_Std'
    }, inplace=True)
    
    return grouped

# --- 3. PLOTTING ---
def plot_averaged_pareto(df, grouped_df):
    plt.figure(figsize=(16, 10))
    sns.set_style("whitegrid")

    # --- MAPPINGS ---
    unique_cats = sorted(grouped_df['Model_Category'].unique())
    palette = sns.color_palette("bright", len(unique_cats))
    color_map = dict(zip(unique_cats, palette))

    # Size mapping for Width
    width_map = {0: 80, 10: 150, 100: 250, 1280: 400}

    # --- DRAW TRAJECTORIES (Lines connecting Means) ---
    print("Drawing trajectories...")
    for cat in unique_cats:
        subset = grouped_df[grouped_df['Model_Category'] == cat].sort_values(by='Width')
        col = color_map.get(cat, 'gray')
        
        plt.plot(subset['Speedup_Mean'], subset['Gap_Mean'], 
                 color=col, alpha=0.6, linewidth=2, zorder=1)

    # --- DRAW ERROR BARS & POINTS ---
    print("Drawing points with error bars...")
    for idx, row in grouped_df.iterrows():
        cat = row['Model_Category']
        width = row['Width']
        
        # Error Bar
        plt.errorbar(
            x=row['Speedup_Mean'], 
            y=row['Gap_Mean'],
            yerr=row['Gap_Std'], 
            fmt='none', # Don't draw the point again, just the bar
            ecolor=color_map[cat], 
            elinewidth=1.5, 
            capsize=4, 
            alpha=0.6,
            zorder=2
        )
        
        # The Mean Point
        plt.scatter(
            row['Speedup_Mean'], 
            row['Gap_Mean'], 
            c=[color_map[cat]], 
            marker='o', # Standard circle for means
            s=width_map.get(width, 100),
            edgecolor='white', linewidth=1.0, alpha=1.0, zorder=3
        )

    # --- DRAW LKH REFERENCE ---
    plt.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, label='LKH Baseline Speed', zorder=0)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3, zorder=0)
    
    # Mark LKH
    plt.scatter(1.0, 0, color='black', marker='*', s=500, label='LKH Solver', zorder=4)
    plt.text(1.1, 0.5, "LKH Baseline (3.7ms)", verticalalignment='bottom', fontsize=10, fontweight='bold')

    # --- AXES & LABELS ---
    plt.xscale('log')
    plt.xlabel("Mean Speedup Factor vs LKH (Log Scale)", fontsize=12, fontweight='bold')
    plt.ylabel("Mean Optimality Gap (%) ± Std Dev", fontsize=12, fontweight='bold')
    plt.title("Pareto Frontier: Average Performance by Architecture for TSP20\n(Aggregated over Entropy Coeffs)", fontsize=16)

    # Custom Ticks
    ticks = [0.1, 0.5, 1, 10, 50, 100]
    plt.xticks(ticks, [f"{t}x" for t in ticks])

    # --- LEGENDS ---
    # 1. Architecture (Color)
    leg_lines = [plt.Line2D([0], [0], color=color_map[c], marker='o', linestyle='', markersize=10) for c in unique_cats]
    leg1 = plt.legend(leg_lines, unique_cats, title="Architecture", loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    plt.gca().add_artist(leg1)

    # 2. Strategy (Size)
    size_lines = [plt.Line2D([0], [0], color='gray', marker='o', linestyle='', markersize=np.sqrt(s)/1.5) for s in width_map.values()]
    size_labels = ["Greedy", "BS-10", "BS-100", "BS-1280"]
    leg2 = plt.legend(size_lines, size_labels, title="Strategy", loc='upper left', bbox_to_anchor=(1.02, 0.75), borderaxespad=0.)

    plt.subplots_adjust(right=0.8)
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Saved to {OUTPUT_FILE}")
    plt.show()

# --- EXECUTION ---
if __name__ == "__main__":
    df = process_data(INPUT_FILE)
    grouped_df = aggregate_data(df)
    
    # Print a preview of the aggregated data for the user
    print("\nAggregated Data Preview:")
    print(grouped_df[['Model_Category', 'Width', 'Gap_Mean', 'Gap_Std', 'Speedup_Mean']].head(10))
    
    plot_averaged_pareto(df, grouped_df)