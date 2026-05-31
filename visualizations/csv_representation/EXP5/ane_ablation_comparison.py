import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from matplotlib.patches import Patch

# Create output directory
out_dir = 'visualizations/csv_eval/EXP5_RIVAL'
os.makedirs(out_dir, exist_ok=True)

# Macros for plotting options
PLOT_BS = False   # Set to True to plot the bs10 and bs100 bars
PLOT_STD = False   # Set to True to include standard deviation error bars

# Load dataframe
df = pd.read_csv('results/ane_ablation_full_2.csv')

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

# Use tab20 colormap: we will only use the even indices (darker tones) for colors
cmap = plt.get_cmap('tab20')
core_color_idx = {model: i for i, model in enumerate(core_models)}

labels = []
colors = []
hatches = []

# Means
greedy_vals = []
bs10_vals = []
bs100_vals = []

# Standard Deviations
greedy_std = []
bs10_std = []
bs100_std = []

for _, row in best_epochs.iterrows():
    model_name = row['Model_Name'] # The specific name containing the best epoch
    core_name = row['Core_Model']
    is_wind = row['Is_Wind']
    
    labels.append(model_name)
    greedy_vals.append(row['Gap_MoR'])
    greedy_std.append(row['Gap_STDoR'])
    
    # Map colors: Force both base and wind to use the exact SAME color (even index)
    c_idx = core_color_idx[core_name] * 2 
    colors.append(cmap(c_idx % 20))
    
    # Assign shading/hatching for WIND models
    if is_wind:
        hatches.append('////')
    else:
        hatches.append('')
    
    if PLOT_BS:
        # Fetch beam search width 10 results
        bs10 = df[(df['Model_Name'] == model_name) & (df['Strategy'] == 'bs') & (df['Width'].astype(str) == '10')]
        bs10_vals.append(bs10['Gap_MoR'].values[0] if not bs10.empty else 0)
        bs10_std.append(bs10['Gap_STDoR'].values[0] if not bs10.empty else 0)
        
        # Fetch beam search width 100 results
        bs100 = df[(df['Model_Name'] == model_name) & (df['Strategy'] == 'bs') & (df['Width'].astype(str) == '100')]
        bs100_vals.append(bs100['Gap_MoR'].values[0] if not bs100.empty else 0)
        bs100_std.append(bs100['Gap_STDoR'].values[0] if not bs100.empty else 0)

# ----------------- PLOTTING -----------------
plt.figure(figsize=(18, 8))
x = np.arange(len(labels))

# Set up error bar parameters based on the macro
greedy_yerr = greedy_std if PLOT_STD else None
bs10_yerr = bs10_std if PLOT_STD else None
bs100_yerr = bs100_std if PLOT_STD else None
cap_size = 4 if PLOT_STD else 0

if not PLOT_BS:
    bars = plt.bar(x, greedy_vals, color=colors, edgecolor='black', yerr=greedy_yerr, capsize=cap_size)
    
    # Apply hatching
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
        
    # Write the exact value above each bar (and above the error bar if it exists)
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        std_offset = greedy_std[i] if PLOT_STD else 0
        plt.text(bar.get_x() + bar.get_width()/2, yval + std_offset + (max(greedy_vals)*0.02), f'{yval:.2f}', 
                 ha='center', va='bottom', fontsize=10, rotation=45)
                 
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=10)
    
    # Add a legend just to explain the WIND shading
    legend_elements = [Patch(facecolor='white', edgecolor='black', hatch='////', label='WIND Model')]
    plt.legend(handles=legend_elements, loc='upper right')

else:
    width = 0.25
    # Plot Greedy
    bars1 = plt.bar(x - width, greedy_vals, width, color=colors, edgecolor='black', alpha=1.0, 
                    yerr=greedy_yerr, capsize=cap_size, label='Greedy')
    for bar, h in zip(bars1, hatches):
        bar.set_hatch(h)
    
    # Plot BS 10
    bars2 = plt.bar(x, bs10_vals, width, color=colors, edgecolor='black', alpha=0.6, 
                    yerr=bs10_yerr, capsize=cap_size, label='BS 10')
    for bar, h in zip(bars2, hatches):
        bar.set_hatch(h + '//') 
    
    # Plot BS 100
    bars3 = plt.bar(x + width, bs100_vals, width, color=colors, edgecolor='black', alpha=0.3, 
                    yerr=bs100_yerr, capsize=cap_size, label='BS 100')
    for bar, h in zip(bars3, hatches):
        bar.set_hatch(h + 'xx')
    
    # Write values for all grouped bars
    for bars, stds in zip([bars1, bars2, bars3], [greedy_std, bs10_std, bs100_std]):
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            if yval > 0:
                std_offset = stds[i] if PLOT_STD else 0
                plt.text(bar.get_x() + bar.get_width()/2, yval + std_offset + (max(greedy_vals)*0.02), f'{yval:.2f}', 
                         ha='center', va='bottom', fontsize=8, rotation=90)
                
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=10)
    
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', alpha=1.0, label='Greedy'),
        Patch(facecolor='gray', edgecolor='black', alpha=0.6, hatch='//', label='Beam Search (Width=10)'),
        Patch(facecolor='gray', edgecolor='black', alpha=0.3, hatch='xx', label='Beam Search (Width=100)'),
        Patch(facecolor='white', edgecolor='black', hatch='////', label='WIND Model')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

plt.ylabel('Gap_MoR', fontsize=12)
plt.title('Best Epoch (Min Gap_MoR) Comparison Across Models', fontsize=14)

# Calculate the maximum y-value needed for the chart limits based on standard deviations
if PLOT_BS:
    max_greedy = max([v + (s if PLOT_STD else 0) for v, s in zip(greedy_vals, greedy_std)]) if greedy_vals else 0
    max_bs10 = max([v + (s if PLOT_STD else 0) for v, s in zip(bs10_vals, bs10_std)]) if bs10_vals else 0
    max_bs100 = max([v + (s if PLOT_STD else 0) for v, s in zip(bs100_vals, bs100_std)]) if bs100_vals else 0
    max_val = max(max_greedy, max_bs10, max_bs100)
else:
    max_val = max([v + (s if PLOT_STD else 0) for v, s in zip(greedy_vals, greedy_std)]) if greedy_vals else 0

plt.ylim(0, max_val * 1.2)
plt.tight_layout()

output_path = f'{out_dir}/best_epochs_comparison.png'
plt.savefig(output_path)
print(f"Generated chart at {output_path}")