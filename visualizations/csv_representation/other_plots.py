import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = "results/evaluations/01experiment1NoBlank.csv"
OUTPUT_FILE_ENTROPY = "eval_analysis_entropy_sensitivity.png"
OUTPUT_FILE_SCALING = "eval_analysis_beam_scaling.png"
OUTPUT_FILE_CONFIDENCE = "eval_analysis_confidence_calibration.png"

# --- DATA PROCESSING ---
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    
    def parse_row(row):
        name = row['Model_Name']
        if name == "LKH_Baseline":
            return "LKH", "baseline", "LKH"
        
        parts = name.split('_')
        exp_name = parts[0]
        
        ftype = "unknown"
        for t in ['blank', 'coords', 'hybrid', 'learned']:
            if t in parts:
                ftype = t
                break
        
        # Entropy
        entropy = 0.0
        for p in parts:
            if p.startswith('ent'):
                try:
                    entropy = float(p.replace('ent', ''))
                except:
                    pass
        
        return exp_name, ftype, entropy, f"{ftype.capitalize()} ({row['Feature_Dims']}d)"

    df[['Experiment', 'Feature_Type', 'Entropy', 'Model_Category']] = df.apply(
        lambda row: pd.Series(parse_row(row)), axis=1
    )
    
    # Filter out LKH and Blank for these specific neural analyses
    neural_df = df[~df['Model_Name'].isin(['LKH_Baseline']) & (df['Feature_Type'] != 'blank')].copy()
    return neural_df

def plot_entropy_sensitivity(df):
    """Plot 1: How does Entropy Coeff affect performance?"""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # We focus on BS-100 as the representative strategy
    subset = df[df['Width'] == 100].sort_values(by='Entropy')
    
    sns.lineplot(
        data=subset, 
        x='Entropy', 
        y='Gap_Percent', 
        hue='Model_Category', 
        style='Model_Category',
        markers=True, dashes=False, linewidth=2.5, markersize=9
    )
    
    plt.xscale('log')
    plt.title("Entropy Sensitivity (at Beam Width 100)", fontsize=14, fontweight='bold')
    plt.ylabel("Optimality Gap (%)", fontsize=12)
    plt.xlabel("Entropy Coefficient (Log Scale)", fontsize=12)
    plt.legend(title="Architecture", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_ENTROPY, dpi=300)
    print(f"Saved {OUTPUT_FILE_ENTROPY}")
    plt.show()

def plot_beam_scaling(df):
    """Plot 2: The Diminishing Returns of Search"""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Average over entropies to get the 'general behavior' of the architecture
    sns.lineplot(
        data=df, 
        x='Width', 
        y='Gap_Percent', 
        hue='Model_Category',
        style='Model_Category',
        markers=True, dashes=False, linewidth=2.5, markersize=9,
        err_style='band' # Shows the spread due to entropy differences
    )
    
    # Fix X-axis to categorical-like log behavior
    plt.xscale('symlog', linthresh=10) # Allows 0 to exist comfortably near 10
    plt.xticks([0, 10, 100, 1280], ["Greedy (0)", "BS-10", "BS-100", "BS-1280"])
    
    plt.title("Beam Search Scaling Laws: Improvement vs Cost", fontsize=14, fontweight='bold')
    plt.ylabel("Optimality Gap (%)", fontsize=12)
    plt.xlabel("Decoding Strategy (Beam Width)", fontsize=12)
    plt.legend(title="Architecture", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_SCALING, dpi=300)
    print(f"Saved {OUTPUT_FILE_SCALING}")
    plt.show()

def plot_confidence_calibration(df):
    """Plot 3: Does the model know when it's right?"""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Scatter plot of Confidence vs Gap
    sns.scatterplot(
        data=df,
        x='Avg_Confidence',
        y='Gap_Percent',
        hue='Model_Category',
        size='Width',
        sizes=(50, 200),
        alpha=0.7
    )
    
    plt.title("Confidence vs. Competence Calibration", fontsize=14, fontweight='bold')
    plt.ylabel("Optimality Gap (%) (Lower is Better)", fontsize=12)
    plt.xlabel("Average Model Confidence (Higher is Surer)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Invert X axis? No, usually 0->1 left to right.
    # Ideal quadrant: Bottom-Right (High Confidence, Low Gap)
    # Dangerous quadrant: Top-Right (High Confidence, High Gap -> Hallucination)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_CONFIDENCE, dpi=300)
    print(f"Saved {OUTPUT_FILE_CONFIDENCE}")
    plt.show()

# --- EXECUTION ---
if __name__ == "__main__":
    neural_df = process_data(INPUT_FILE)
    
    plot_entropy_sensitivity(neural_df)
    plot_beam_scaling(neural_df)
    plot_confidence_calibration(neural_df)