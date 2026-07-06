import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Load the data
df = pd.read_csv('results/nab_ablation_full_time.csv')

# Get the baseline time
lkh_time = df[df['Model_Name'] == 'LKH_Baseline']['Time_Per_Solution'].values[0]

# Filter specifically for the 'greedy' strategy
df_greedy = df[df['Strategy'] == 'greedy'].copy()

# Function to extract model, NAB strategy, and neighbors from the Model_Name string
def extract_model_strat_neigh(name):
    if name == 'LKH_Baseline':
        return None, None, None
    # match model: everything between V3_ and _original
    model_match = re.search(r'V3_(.*?)_original', name)
    strat_match = re.search(r'nab-(.*?)(_|$)', name)
    neigh_match = re.search(r'_neighbors(\d+)_', name)
    
    if model_match and strat_match and neigh_match:
        # Removing '-standard' from the extracted name
        model = model_match.group(1).replace('-standard', '') 
        strat = strat_match.group(1)
        neigh = neigh_match.group(1)
        return model, strat, neigh
    return None, None, None

df_greedy['Model'], df_greedy['Strat'], df_greedy['Neighbors'] = zip(*df_greedy['Model_Name'].apply(extract_model_strat_neigh))
df_greedy = df_greedy.dropna(subset=['Model', 'Strat', 'Neighbors'])

# Average Time_Per_Solution if there are duplicate variations
df_grouped = df_greedy.groupby(['Model', 'Strat', 'Neighbors'])['Time_Per_Solution'].mean().reset_index()

# Calculate SPEEDUP: How many times faster it is vs baseline (LKH Time / Model Time)
df_grouped['Multiplier'] = lkh_time / df_grouped['Time_Per_Solution']

# Sorting for plotting logically
df_grouped = df_grouped.sort_values(by=['Model', 'Strat', 'Neighbors'])

# Plotting Configuration
fig, ax = plt.subplots(figsize=(10, 12))

models = df_grouped['Model'].unique()
strats = ['none', 'encoder', 'decoder', 'both']

# Group colors by model
base_colors = {
    'aafm': 'tab:blue',
    'edge_gat': 'tab:orange',
    'gat': 'tab:green',
    'gnn-deep': 'tab:red',
    'gnn': 'tab:purple',
    'mlp': 'tab:brown'
}

# Distinguish NAB strategies using color opacity/shading
alphas = {'none': 0.3, 'encoder': 0.5, 'decoder': 0.7, 'both': 1.0}

y_positions = []
y_labels = []

current_y = 0
for model in models:
    for strat in strats:
        # Filter for current model and strategy
        rows = df_grouped[(df_grouped['Model'] == model) & (df_grouped['Strat'] == strat)]
        
        for _, row in rows.iterrows():
            val = row['Time_Per_Solution']
            mult = row['Multiplier']
            neigh = row['Neighbors']
            
            color = base_colors.get(model, 'gray')
            alpha = alphas.get(strat, 1.0)
            
            # Plot horizontal bar with log scale on X axis
            bar = ax.barh(current_y, val, color=color, alpha=alpha, edgecolor='black', log=True)
            
            # Switched back to 1 decimal place (.1f) since the numbers are now large multipliers (e.g. 50.1x)
            ax.text(val * 1.05, current_y, f"{mult:.1f}x", va='center', fontsize=9)
            
            y_positions.append(current_y)
            # Add neighbors count to the label
            y_labels.append(f"{model} + {strat} (NN: {neigh})")
            current_y += 1
            
    current_y += 1 # Adds space visually distinguishing between different model families

ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels)
ax.set_xlabel('Time Per Solution (Seconds) - Log Scale', fontsize=11)
ax.set_title('Inference Time Compared to LKH Baseline (Speedup Multiplier)', fontsize=14)

# Remove unnecessary plot borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('visualizations/csv_eval/EXP5_RIVAL/nab_horizontal_block_graph_speedup.png')