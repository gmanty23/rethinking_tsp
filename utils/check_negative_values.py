#!/usr/bin/env python

import pandas as pd
import os

def identify_sub_zero_variance_models(csv_path):
    """
    Scans the evaluation CSV to identify configurations where the 
    standard deviation of the gap reaches below 0 (meaning it beats LKH on some instances).
    """
    print(f"Analyzing ablation variance from: {csv_path}\n")
    
    if not os.path.exists(csv_path):
        print(f"[!] Error: File not found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        
        # Ensure columns exist and are numeric
        for col in ['Gap_MoR', 'Gap_STDoR']:
            if col not in df.columns:
                print(f"[!] Error: '{col}' column missing from the CSV.")
                return
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate the 1-Sigma Lower Bound
        df['Lower_Bound (MoR - STD)'] = df['Gap_MoR'] - df['Gap_STDoR']
        
        # Filter for models where the lower bound is below zero
        sub_zero_df = df[(df['Lower_Bound (MoR - STD)'] < 0.0) & (df['Model_Name'] != 'LKH_Baseline')].copy()
        
        if sub_zero_df.empty:
            print("[-] No configurations had a standard deviation dropping below 0.0.")
        else:
            print(f"[+] Found {len(sub_zero_df)} configuration(s) with sub-zero variance bounds!")
            print("    This means these models beat LKH on a significant chunk of the dataset.")
            print("-" * 110)
            
            # Sort by how deep into the negative the bound goes
            sub_zero_df = sub_zero_df.sort_values(by='Lower_Bound (MoR - STD)', ascending=True)
            
            # Format the output for readability
            cols_to_show = [
                'Model_Name', 'Strategy', 'Width', 
                'Gap_MoR', 'Gap_STDoR', 'Lower_Bound (MoR - STD)'
            ]
            
            # Print the dataframe
            print(sub_zero_df[cols_to_show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            
            print("-" * 110)
            print("\nResearch Implication:")
            print("You should isolate the specific graphs where these top models beat LKH.")
            print("It is highly likely these graphs feature extreme wind asymmetries (high delta C_ij).")
            
    except Exception as e:
        print(f"[!] An unexpected error occurred during analysis: {e}")

if __name__ == "__main__":
    CSV_FILE = "results/evaluations/05experiment5_neighbors2.csv"
    identify_sub_zero_variance_models(CSV_FILE)