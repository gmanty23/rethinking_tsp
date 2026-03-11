import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# --- Configuration ---
INPUT_FILE = "results/evaluations/04experiment1_compknn_TSP20_neighbors.csv"
OUTPUT_FILE = "visualizations/csv_eval/TSP20_EXP4/eval_bar2_TSP20_10.png"

# Selection Variable: Choose from ['greedy', 10, 100, 1280]
# Set to 'greedy' for Greedy decoding, or an integer for Beam Search width.
SELECTED_WIDTH = 10

def plot_windy_ablation_single():
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)

    # 2. Extract Metadata
    def parse_metadata(name):
        if not isinstance(name, str) or "tsp20" not in name:
            return None
        match = re.search(r'tsp20_([a-z]+)_hybrid_ent[\d.]+_([a-z]+)_n([\d.]+)_', name)
        if match:
            return {
                'Msg_Passing': match.group(1),
                'Sparsification': match.group(2),
                'Neighbors': float(match.group(3))
            }
        return None

    meta = df['Model_Name'].apply(parse_metadata)
    df_neural = df[meta.notnull()].copy()
    df_neural[['Msg_Passing', 'Sparsification', 'Neighbors']] = pd.DataFrame(meta.dropna().tolist(), index=df_neural.index)

    # 3. Filter for Selected Strategy
    if SELECTED_WIDTH == 'greedy':
        target_df = df_neural[df_neural['Strategy'] == 'greedy']
        title_strat = "Greedy Decoding"
    else:
        target_df = df_neural[(df_neural['Strategy'] == 'bs') & (df_neural['Width'] == SELECTED_WIDTH)]
        title_strat = f"Beam Search (Width {SELECTED_WIDTH})"

    if target_df.empty:
        print(f"No data found for the selected width: {SELECTED_WIDTH}")
        return

    # 4. Plot Setup
    neighbor_vals = sorted(target_df['Neighbors'].unique())
    spar_types = ['random', 'knn']
    msg_types = ['backward', 'forward', 'dual']

    colors = {'knn': '#006400', 'random': '#90EE90'}  # Dark vs Light Green
    hatches = {'backward': '', 'forward': '//', 'dual': '..'} 

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_width = 0.14
    x = np.arange(len(neighbor_vals))
    offsets = np.linspace(-bar_width*2.5, bar_width*2.5, 6)

    bar_count = 0
    for sparse in spar_types:
        for msg in msg_types:
            sub_df = target_df[(target_df['Sparsification'] == sparse) & (target_df['Msg_Passing'] == msg)]
            
            heights = [sub_df[sub_df['Neighbors'] == n]['Gap_MoR'].values[0] 
                       if n in sub_df['Neighbors'].values else 0 for n in neighbor_vals]
            
            label = f"{sparse.upper()} + {msg.capitalize()} MP"
            ax.bar(x + offsets[bar_count], heights, width=bar_width, 
                   color=colors[sparse], hatch=hatches[msg], 
                   edgecolor='black', alpha=0.9, label=label)
            bar_count += 1

    # 5. Styling
    ax.set_title(f"Ablation Analysis: Optimality Gap (MoR)\nStrategy: {title_strat}", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Gap MoR (%)", fontsize=12)
    ax.set_xlabel("Neighborhood Sparsification Factor (n)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in neighbor_vals])
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(title="Architecture & Message Passing", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Isolated plot saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    plot_windy_ablation_single()