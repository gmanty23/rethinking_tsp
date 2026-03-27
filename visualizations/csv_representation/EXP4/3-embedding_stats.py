import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --- Configuration ---
INPUT_FILE = "results/evaluations/04experiment1_compknn_TSP20_neighbors.csv"
OUT_VAR = "visualizations/csv_eval/TSP20_EXP4/scatter_variance_vs_gap_TODOS.png"
OUT_DIRICHLET = "visualizations/csv_eval/TSP20_EXP4/scatter_dirichlet_vs_gap_TODOS.png"

def generate_scatter_plots():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)

    # 1. Parsing Metadata from Model_Name
    def parse_metadata(name):
        if not isinstance(name, str) or "tsp20" not in name: return None
        # Extract: {msg}_{feature_dims}_{ent}_{spar}_{n}_{date}
        match = re.search(r'tsp20_([a-z]+)_hybrid_ent[\d.]+_([a-z]+)_n([\d.]+)_', name)
        if match:
            return {
                'Msg_Passing': match.group(1), 
                'Sparsification': match.group(2), 
                'Neighbors': float(match.group(3))
            }
        return None

    meta = df['Model_Name'].apply(parse_metadata)
    df = df[meta.notnull()].copy()
    df[['Msg_Passing', 'Sparsification', 'Neighbors']] = pd.DataFrame(meta.dropna().tolist(), index=df.index)

    # 2. Define Mappings (Ensuring all are defined in scope)
    # Size based on BS Width
    size_map = {0: 60, 10: 150, 100: 400, 1280: 900}
    
    # Color based on Architecture
    color_map = {'knn': '#006400', 'random': '#90EE90'} # Dark vs Light Green
    
    # Hatch based on Message Passing (This was missing in your previous run)
    hatch_map = {'backward': '', 'forward': '///', 'dual': '...'} 
    
    # Shape based on Neighbors (n)
    unique_neighbors = sorted(df['Neighbors'].unique())
    all_markers = ['o', 's', '^', 'D', 'p', 'X', '*', 'H']
    marker_map = {n: all_markers[i % len(all_markers)] for i, n in enumerate(unique_neighbors)}

    # 3. Plotting Logic
    def create_plot(x_col, y_col, out_path, title):
        fig, ax = plt.subplots(figsize=(13, 9))
        
        # Iterate and plot each point to apply unique marker/hatch combinations
        for _, row in df.iterrows():
            ax.scatter(
                row[x_col], row[y_col],
                s=size_map.get(row['Width'], 60),
                c=color_map.get(row['Sparsification'], 'gray'),
                marker=marker_map.get(row['Neighbors'], 'o'),
                hatch=hatch_map.get(row['Msg_Passing'], ''),
                edgecolor='black',
                linewidths=0.8,
                alpha=0.8
            )

        ax.set_xlabel(x_col.replace('_', ' '), fontsize=12, fontweight='bold')
        ax.set_ylabel(y_col.replace('_', ' '), fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold', pad=25)
        ax.grid(True, linestyle='--', alpha=0.3)

        # 4. Corrected Legend Construction
        legend_elements = [
            # Architecture (Colors)
            Line2D([0], [0], marker='o', color='w', label='Arch: KNN', 
                   markerfacecolor=color_map['knn'], markersize=12),
            Line2D([0], [0], marker='o', color='w', label='Arch: Random', 
                   markerfacecolor=color_map['random'], markersize=12),
            
            Patch(facecolor='none', edgecolor='none', label=''), # Spacer
            
            # Message Passing (Hatches using Patch)
            Patch(facecolor='gray', edgecolor='black', label='MP: Backward (Solid)'),
            Patch(facecolor='gray', edgecolor='black', hatch='///', label='MP: Forward (Slashed)'),
            Patch(facecolor='gray', edgecolor='black', hatch='...', label='MP: Dual (Dotted)'),
            
            Patch(facecolor='none', edgecolor='none', label=''), # Spacer
            
            # Strategy (Size)
            Line2D([0], [0], marker='o', color='w', label='Dec: Greedy', 
                   markerfacecolor='gray', markersize=np.sqrt(60)),
            Line2D([0], [0], marker='o', color='w', label='Dec: BS-1280', 
                   markerfacecolor='gray', markersize=np.sqrt(900))
        ]

        # Neighbors (Shapes)
        shape_elements = [
            Line2D([0], [0], marker=marker_map[n], color='w', label=f'n={n}', 
                   markerfacecolor='gray', markersize=10, markeredgecolor='black') 
            for n in unique_neighbors
        ]

        leg1 = ax.legend(handles=legend_elements, title="Model & Strategy Config", 
                         loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
        leg2 = ax.legend(handles=shape_elements, title="Sparsity (Neighbors)", 
                         loc='lower left', bbox_to_anchor=(1.02, 0), frameon=True)
        ax.add_artist(leg1)
        
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Plot saved successfully: {out_path}")

    # 5. Generate both representation analysis graphs
    create_plot('Avg_Embedding_Variance', 'Gap_MoR', OUT_VAR, 
                "Representation Analysis: Embedding Variance vs. Optimality Gap")
    
    create_plot('Avg_Dirichlet_Energy', 'Gap_MoR', OUT_DIRICHLET, 
                "Spectral Analysis: Dirichlet Energy vs. Optimality Gap")

if __name__ == "__main__":
    generate_scatter_plots()