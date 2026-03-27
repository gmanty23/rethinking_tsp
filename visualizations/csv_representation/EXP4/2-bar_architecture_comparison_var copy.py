import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# --- Configuration ---
INPUT_FILE = "results/evaluations/06experiment6_neighbors_coords_learned_sparse.csv"
OUTPUT_FILE = "visualizations/csv_eval/coords_learned_sparse.png"

# Selection Variable: Choose from ['greedy', 10, 100, 1280]
SELECTED_WIDTH = 10

def plot_windy_ablation_single():
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)

    # 2. Extract Metadata
    def parse_metadata(name):
        if not isinstance(name, str) or "tsp" not in name:
            return None
        
        # Updated regex to capture 4 groups: msg_passing, embedding, sparsification, neighbors
        match = re.search(r'tsp\d+_([a-z]+)_([a-z]+)_ent[\d.]+_([a-z]+)_n([\d.]+)_', name)
        
        if match:
            return {
                'Msg_Passing': match.group(1),    # backward, forward, dual
                'Embedding': match.group(2),      # coords, learned
                'Sparsification': match.group(3), # knn, random
                'Neighbors': float(match.group(4))
            }
        return None

    meta = df['Model_Name'].apply(parse_metadata)
    df_neural = df[meta.notnull()].copy()
    
    # Expand metadata into 4 columns now
    meta_df = pd.DataFrame(meta.dropna().tolist(), index=df_neural.index)
    df_neural[['Msg_Passing', 'Embedding', 'Sparsification', 'Neighbors']] = meta_df

    # 3. Filter for Selected Strategy
    if SELECTED_WIDTH == 'greedy':
        target_df = df_neural[df_neural['Strategy'] == 'greedy']
        title_strat = "Greedy Decoding"
    else:
        target_df = df_neural[(df_neural['Strategy'] == 'bs') & (df_neural['Width'] == int(SELECTED_WIDTH))]
        title_strat = f"Beam Search (Width {SELECTED_WIDTH})"

    if target_df.empty:
        print(f"No data found for the selected width: {SELECTED_WIDTH}")
        return

    # 4. Plot Setup
    neighbor_vals = sorted(target_df['Neighbors'].unique())
    emb_types = ['coords', 'learned']
    spar_types = ['random', 'knn']
    msg_types = ['backward', 'forward', 'dual']

    # Color Mapping: Greens for Coords, Blues for Learned
    colors = {
        ('coords', 'knn'): '#006400',    # Dark Green
        ('coords', 'random'): '#90EE90',  # Light Green
        ('learned', 'knn'): '#00008B',   # Dark Blue
        ('learned', 'random'): '#ADD8E6'  # Light Blue
    }
    
    hatches = {'backward': '', 'forward': '//', 'dual': '..'} 

    fig, ax = plt.subplots(figsize=(16, 8)) # Made figure slightly wider to fit the legend
    
    # Adjust bar width and offsets for 12 bars (2 emb * 2 spar * 3 msg)
    bar_width = 0.07 
    x = np.arange(len(neighbor_vals))
    offsets = np.linspace(-bar_width*5.5, bar_width*5.5, 12)

    bar_count = 0
    for emb in emb_types:
        for sparse in spar_types:
            for msg in msg_types:
                # Filter down to the specific combination of the 3 categorical variables
                sub_df = target_df[(target_df['Embedding'] == emb) & 
                                   (target_df['Sparsification'] == sparse) & 
                                   (target_df['Msg_Passing'] == msg)]
                
                heights = []
                yerrs = []
                
                for n in neighbor_vals:
                    n_df = sub_df[sub_df['Neighbors'] == n]
                    if not n_df.empty:
                        heights.append(n_df['Gap_MoR'].values[0])
                        yerrs.append(n_df['Gap_STDoR'].values[0])
                    else:
                        heights.append(0.0)
                        yerrs.append(0.0)
                
                label = f"{emb.capitalize()} | {sparse.upper()} | {msg.capitalize()} MP"
                
                ax.bar(x + offsets[bar_count], heights, width=bar_width, 
                       color=colors[(emb, sparse)], hatch=hatches[msg], 
                       edgecolor='black', alpha=0.9, label=label,
                       yerr=yerrs, capsize=3, error_kw={'elinewidth': 1.0, 'alpha': 0.8})
                
                bar_count += 1

    # 5. Styling
    ax.set_title(f"Ablation Analysis: Optimality Gap (MoR) with Variance\nStrategy: {title_strat}", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Gap MoR (%) $\pm$ StdDev", fontsize=12)
    ax.set_xlabel("Neighborhood Sparsification Factor (n)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in neighbor_vals])
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Move legend slightly further out so it doesn't overlap the plot
    ax.legend(title="Embedding | Sparsification | Message Passing", 
              bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')

    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Isolated plot saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    plot_windy_ablation_single()