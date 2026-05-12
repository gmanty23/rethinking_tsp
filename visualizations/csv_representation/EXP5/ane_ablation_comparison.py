import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# Create output directory
out_dir = 'visualizations/csv_eval/EXP5_RIVAL'
os.makedirs(out_dir, exist_ok=True)

# Macro: Set to True to plot the bs10 and bs100 bars next to the greedy bars
PLOT_BS = False  

# Load dataframe
df = pd.read_csv('results/ane_ablation_full.csv')

# Focus on greedy to find the best epoch per model
df_greedy = df[df['Strategy'] == 'greedy'].copy()

# Extract Base Model string (without the epoch information)
df_greedy['Base_Model'] = df_greedy['Model_Name'].apply(lambda x: x.split('_epoch')[0] if '_epoch' in x else x)

# Find the row index with the minimum Gap_MoR for each Base_Model (Best performing epoch)
best_idx = df_greedy.groupby('Base_Model')['Gap_MoR'].idxmin()
best_epochs = df_greedy.loc[best_idx].copy()

# Identify core model (without 'wind') to assign color families
def get_core_model(name):
    if name.endswith('_wind'): return name[:-5]
    return name

best_epochs['Core_Model'] = best_epochs['Base_Model'].apply(get_core_model)
best_epochs['Is_Wind'] = best_epochs['Base_Model'].apply(lambda x: x.endswith('_wind'))

# Sort for better visualization (keeps core models grouped together)
best_epochs = best_epochs.sort_values(['Core_Model', 'Is_Wind'])

core_models = best_epochs['Core_Model'].unique()

# Use tab20 colormap: it naturally has paired colors (light/dark of the same hue)
cmap = plt.get_cmap('tab20')
core_color_idx = {model: i for i, model in enumerate(core_models)}

labels = []
greedy_vals = []
bs10_vals = []
bs100_vals = []
colors = []

for _, row in best_epochs.iterrows():
    model_name = row['Model_Name'] # The specific name containing the best epoch
    core_name = row['Core_Model']
    is_wind = row['Is_Wind']
    
    labels.append(model_name)
    greedy_vals.append(row['Gap_MoR'])
    
    # Map colors: standard is even index (darker tone), wind is odd index (lighter tone)
    c_idx = (core_color_idx[core_name] * 2) + (1 if is_wind else 0)
    colors.append(cmap(c_idx % 20))
    
    if PLOT_BS:
        # Fetch beam search width 10 results for this specific model and epoch
        bs10 = df[(df['Model_Name'] == model_name) & (df['Strategy'] == 'bs') & (df['Width'].astype(str) == '10')]
        bs10_vals.append(bs10['Gap_MoR'].values[0] if not bs10.empty else 0)
        
        # Fetch beam search width 100 results for this specific model and epoch
        bs100 = df[(df['Model_Name'] == model_name) & (df['Strategy'] == 'bs') & (df['Width'].astype(str) == '100')]
        bs100_vals.append(bs100['Gap_MoR'].values[0] if not bs100.empty else 0)

# ----------------- PLOTTING -----------------
plt.figure(figsize=(18, 8))
x = np.arange(len(labels))

if not PLOT_BS:
    bars = plt.bar(x, greedy_vals, color=colors, edgecolor='black')
    # Write the exact value above each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(greedy_vals)*0.02), f'{yval:.2f}', 
                 ha='center', va='bottom', fontsize=10, rotation=45)
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=10)

else:
    width = 0.25
    # Plot Greedy (Solid fill, no hatch)
    bars1 = plt.bar(x - width, greedy_vals, width, color=colors, edgecolor='black', alpha=1.0, label='Greedy')
    
    # Plot BS 10 (Slightly faded, single hatch)
    bars2 = plt.bar(x, bs10_vals, width, color=colors, edgecolor='black', alpha=0.6, hatch='//', label='BS 10')
    
    # Plot BS 100 (Most faded, double cross hatch)
    bars3 = plt.bar(x + width, bs100_vals, width, color=colors, edgecolor='black', alpha=0.3, hatch='xx', label='BS 100')
    
    # Write values for all grouped bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                plt.text(bar.get_x() + bar.get_width()/2, yval + (max(greedy_vals)*0.02), f'{yval:.2f}', 
                         ha='center', va='bottom', fontsize=8, rotation=90)
                
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=10)
    
    # Custom legend so we only show shading keys, not the color mapping keys
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', alpha=1.0, label='Greedy'),
        Patch(facecolor='gray', edgecolor='black', alpha=0.6, hatch='//', label='Beam Search (Width=10)'),
        Patch(facecolor='gray', edgecolor='black', alpha=0.3, hatch='xx', label='Beam Search (Width=100)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

plt.ylabel('Gap_MoR', fontsize=12)
plt.title('Best Epoch (Min Gap_MoR) Comparison Across Models', fontsize=14)

# Adjust ylim to ensure the label texts don't get cut off at the top
max_val = max(greedy_vals + bs10_vals + bs100_vals) if PLOT_BS else max(greedy_vals)
plt.ylim(0, max_val * 1.2)
plt.tight_layout()

output_path = f'{out_dir}/best_epochs_comparison.png'
plt.savefig(output_path)
print(f"Generated chart at {output_path}")