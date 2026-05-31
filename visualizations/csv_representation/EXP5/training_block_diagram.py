import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Load the dataframe
df = pd.read_csv('results/ane_ablation_full_2.csv')

# Output directory
out_dir = 'visualizations/csv_eval/EXP5_RIVAL'
os.makedirs(out_dir, exist_ok=True)

# Macro for enabling standard deviation error bars
PLOT_STD = False

# Filter for the 'greedy' strategy
df_greedy = df[df['Strategy'] == 'greedy'].copy()

# Extract base model name and epoch
# This splits the name at '_epoch' and assigns the base name and epoch number
df_greedy['Base_Model'] = df_greedy['Model_Name'].apply(lambda x: x.split('_epoch')[0] if '_epoch' in x else x)
df_greedy['Epoch'] = df_greedy['Model_Name'].apply(lambda x: int(x.split('_epoch')[1]) if '_epoch' in x else 0)

# Group by base model name
grouped = df_greedy.groupby('Base_Model')

for base_name, group in grouped:
    # Sort by epoch to ensure correct order on x-axis
    group = group.sort_values('Epoch')
    
    plt.figure(figsize=(10, 6))
    
    # Extract data for plotting
    epochs = group['Epoch'].astype(str)
    gap_mor = group['Gap_MoR']
    gap_std = group['Gap_STDoR']
    
    # Plot block diagrams (bars) with or without std based on the macro
    if PLOT_STD:
        bars = plt.bar(epochs, gap_mor, yerr=gap_std, capsize=5, color='skyblue', edgecolor='black')
    else:
        bars = plt.bar(epochs, gap_mor, color='skyblue', edgecolor='black')
        
    # Write the exact number above the block diagram bars
    for bar in bars:
        yval = bar.get_height()
        # Dynamically adjust text position based on whether error bars are present
        offset = 0.5 if not PLOT_STD else max(gap_std) * 0.1
        plt.text(bar.get_x() + bar.get_width() / 2, yval + offset, 
                 f'{yval:.2f}', ha='center', va='bottom', fontsize=9)
        
    plt.xlabel('Epoch Number', fontsize=12)
    plt.ylabel('Gap_MoR', fontsize=12)
    plt.title(f'Gap_MoR vs Epoch for {base_name}', fontsize=14)
    
    # Extend Y-axis slightly so labels don't get cut off
    plt.ylim(0, max(gap_mor) * 1.2 + (max(gap_std) if PLOT_STD else 0)) 
    
    plt.tight_layout()
    
    # Save the figure as a png
    plt.savefig(f'{out_dir}/{base_name}.png')
    plt.close()

print(f"Successfully generated {len(grouped)} block diagrams in {out_dir}")