import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths
input_path = 'results/evaluations/09final_comparison.csv'
output_dir = 'visualizations/csv_eval/TSP100_RIVAL_PAPER_EXP'
output_path = os.path.join(output_dir, 'pareto_rival_paper.png')

def main():
    # 1. Load the data
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Could not find the input file at {input_path}")
        return

    # 2. Setup the plot
    plt.figure(figsize=(12, 7))
    
    # Get unique models and create a color palette
    models = df['Model'].unique()
    palette = sns.color_palette("husl", len(models))
    
    # 3. Plot points and lines for each model
    for i, model_name in enumerate(models):
        # Filter data for the specific model
        model_df = df[df['Model'] == model_name].copy()
        
        # Sort by Time to ensure the line connects points progressively from left to right
        model_df = model_df.sort_values('Time Per Graph (s)')
        
        color = palette[i]
        
        # Emphasize the LKH Baseline with a star marker, use standard dots for the rest
        is_baseline = 'LKH' in str(model_name).upper()
        marker = '*' if is_baseline else 'o'
        markersize = 14 if is_baseline else 7
        linewidth = 2.5 if len(model_df) > 1 else 0 # Only draw lines if there's >1 strategy
        
        # Plot the line (if multiple strategies) and points
        plt.plot(model_df['Time Per Graph (s)'], model_df['Opt Gap (%)'], 
                 marker=marker, color=color, markersize=markersize,
                 linewidth=linewidth, label=model_name, alpha=0.85)
                 
        # Annotate the model name directly on the graph 
        # (Placed near the last/right-most point of its curve)
        last_point = model_df.iloc[-1]
        plt.annotate(model_name, 
                     (last_point['Time Per Graph (s)'], last_point['Opt Gap (%)']),
                     textcoords="offset points", 
                     xytext=(10, 0), # 10 points to the right
                     ha='left', va='center',
                     fontsize=9, color=color, weight='bold')

    # 4. Set Log scale for X-axis and expand the right margin
    plt.xscale('log')
    plt.xlim(right=1)  # <--- ADDED THIS LINE to expand margin up to 10
    
    # 5. Formatting the Graph
    plt.title('Pareto Frontier: Optimality Gap vs. Inference Time', fontsize=14, pad=15)
    plt.xlabel('Time Per Graph (s) [Log Scale]', fontsize=12)
    plt.ylabel('Optimality Gap (%)', fontsize=12)
    
    # Add grid lines (major and minor for the log scale)
    plt.grid(True, which="major", ls="-", alpha=0.6)
    plt.grid(True, which="minor", ls=":", alpha=0.3)
    
    # Add legend (placed outside the plot to avoid overlapping lines)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 6. Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Pareto frontier graph successfully saved to: {output_path}")

if __name__ == "__main__":
    main()