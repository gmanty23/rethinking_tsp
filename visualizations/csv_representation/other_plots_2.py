import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
INPUT_FILE = "results/evaluations/01experiment1NoBlank.csv"
OUTPUT_BAR = "eval_analysis_feature_impact_bar.png"
OUTPUT_BOX = "eval_analysis_robustness_box.png"

# --- DATA PROCESSING ---
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    
    def parse_row(row):
        name = row['Model_Name']
        if name == "LKH_Baseline":
            return "LKH", "baseline", "LKH"
        
        parts = name.split('_')
        ftype = "unknown"
        for t in ['blank', 'coords', 'hybrid', 'learned']:
            if t in parts:
                ftype = t
                break
        
        # Order for plotting: Blank -> Coords -> Learned -> Hybrid
        order_map = {'blank': 0, 'coords': 1, 'learned': 2, 'hybrid': 3}
        sort_order = order_map.get(ftype, 99)
        
        return ftype, f"{ftype.capitalize()} ({row['Feature_Dims']}d)", sort_order

    df[['Feature_Type', 'Category', 'Sort_Order']] = df.apply(
        lambda row: pd.Series(parse_row(row)), axis=1
    )
    
    # Filter out LKH
    return df[df['Model_Name'] != 'LKH_Baseline'].copy()

# --- PLOT 1: FEATURE IMPACT BAR CHART ---
def plot_feature_impact(df):
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Filter: We only want to show Greedy vs BS-100 (The "Sweet Spot")
    # And we want the BEST entropy for each to represent "Peak Potential"
    subset = df[df['Width'].isin([0, 100])].copy()
    
    # Group by Category + Width, keep the Minimum Gap (Best Entropy)
    best_subset = subset.loc[subset.groupby(['Category', 'Width'])['Gap_Percent'].idxmin()]
    
    # Map Width to nice names
    best_subset['Strategy_Label'] = best_subset['Width'].map({0: 'Greedy (Fast)', 100: 'BS-100 (Balanced)'})
    
    # Sort
    best_subset = best_subset.sort_values(by=['Sort_Order', 'Width'])
    
    # Bar Plot
    ax = sns.barplot(
        data=best_subset,
        x='Category',
        y='Gap_Percent',
        hue='Strategy_Label',
        palette="viridis",
        edgecolor='black',
        alpha=0.9
    )
    
    # Annotate values on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', 
                   xytext=(0, 8), 
                   textcoords='offset points',
                   fontsize=10, fontweight='bold')

    plt.title("Feature Importance: Impact of Input Data on Solution Quality\n(Comparing Best Config per Architecture)", fontsize=14, fontweight='bold')
    plt.ylabel("Optimality Gap (%) (Lower is Better)", fontsize=12)
    plt.xlabel("Model Architecture", fontsize=12)
    plt.legend(title="Decoding Strategy", loc='upper right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_BAR, dpi=300)
    print(f"Saved {OUTPUT_BAR}")
    plt.show()

# --- PLOT 2: ROBUSTNESS BOX PLOT ---
def plot_robustness(df):
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Filter: Fix Strategy to BS-100 to isolate the effect of Architecture/Entropy
    subset = df[df['Width'] == 100].copy()
    subset = subset.sort_values(by='Sort_Order')

    # Box Plot (Shows distribution)
    sns.boxplot(
        data=subset,
        x='Category',
        y='Gap_Percent',
        hue='Category', # Coloring by category looks nice
        palette="pastel",
        dodge=False,
        showfliers=False # Hide outliers, we show them with stripplot
    )
    
    # Strip Plot (Shows individual entropy points)
    sns.stripplot(
        data=subset,
        x='Category',
        y='Gap_Percent',
        color='black',
        alpha=0.6,
        jitter=True,
        size=6
    )

    plt.title("Architecture Robustness: Performance Spread across Hyperparameters\n(Fixed Strategy: BS-100, Varying Entropy)", fontsize=14, fontweight='bold')
    plt.ylabel("Optimality Gap (%)", fontsize=12)
    plt.xlabel("Model Architecture", fontsize=12)
    
    # Remove legend as X-axis explains it
    plt.legend([],[], frameon=False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_BOX, dpi=300)
    print(f"Saved {OUTPUT_BOX}")
    plt.show()

# --- RUN ---
if __name__ == "__main__":
    df = process_data(INPUT_FILE)
    plot_feature_impact(df)
    plot_robustness(df)