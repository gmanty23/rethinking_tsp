import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths
input_path = 'results/mlp100.csv'
output_dir = 'visualizations/csv_eval/TSP100_RIVAL_PAPER_EXP'
# Adding .png to the requested output path name for the image format
output_path = os.path.join(output_dir, 'long_hybrid_training.png') 

def main():
    # 1. Load the data
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Could not find the input file at {input_path}")
        return

    # 2. Filter out non-numeric Model_Names (e.g., 'LKH_Baseline')
    # This isolates just the epochs
    df = df[df['Model_Name'].astype(str).str.isnumeric()].copy()
    
    # Convert 'Model_Name' to integer so it sorts properly (100 before 1000)
    df['Epoch'] = df['Model_Name'].astype(int)

    # 3. Create a combined category for Decoding Style
    def determine_style(row):
        if row['Strategy'] == 'greedy':
            return 'greedy'
        elif row['Strategy'] == 'bs':
            return f"bs_{row['Width']}"
        return row['Strategy']

    df['Decoding_Style'] = df.apply(determine_style, axis=1)

    # 4. Filter to ensure we strictly have the 4 requested styles 
    # (greedy, bs_10, bs_100, bs_1280)
    target_styles = ['greedy', 'bs_10', 'bs_100', 'bs_1280']
    df = df[df['Decoding_Style'].isin(target_styles)]

    # Sort the dataframe by the Epoch numerically
    df = df.sort_values('Epoch')

    # 5. Create the Plot
    plt.figure(figsize=(12, 6))
    
    sns.barplot(
        data=df,
        x='Epoch',
        y='Gap_MoR',
        hue='Decoding_Style',
        hue_order=target_styles,  # Ensures bars are always in this exact order
        palette='Set2'            # Color palette for the bars
    )

    # 6. Formatting
    plt.title('Gap MoR vs Epoch by Decoding Style')
    plt.xlabel('Epoch Number')
    plt.ylabel('Gap MoR')
    
    # Place legend neatly outside the plot if it gets crowded, or just use default
    plt.legend(title='Decoding Style')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 7. Save the output
    os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists
    plt.savefig(output_path, dpi=300)
    print(f"Bar diagram successfully saved to: {output_path}")

if __name__ == "__main__":
    main()