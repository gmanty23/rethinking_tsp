import pandas as pd
import os
import sys

# --- CONFIGURATION ---
# Replace these with your actual file paths
SOURCE_CSV = "/home/pfc/gms/code/rethinking_tsp/results/experiment1_compknn_TSP20_energy.csv"  # The master CSV containing the new metric
TARGET_CSV = "/home/pfc/gms/code/rethinking_tsp/results/evaluations/03experiment1_compknn_TSP20.csv"  # The filtered CSV to update
OUTPUT_CSV = None  # Set to a string to save as a new file, or None to overwrite TARGET_CSV

def copy_column_by_order(source_path, target_path, output_path=None):
    """
    Extracts the very last column from the source CSV and copies it directly 
    to the target CSV, strictly preserving the row order without key matching.
    """
    print(f"[*] Loading Source CSV: {source_path}")
    print(f"[*] Loading Target CSV: {target_path}")

    if not os.path.exists(source_path) or not os.path.exists(target_path):
        print("[!] Error: One or both input paths do not exist. Please check your CONFIGURATION variables.")
        sys.exit(1)

    df_source = pd.read_csv(source_path)
    df_target = pd.read_csv(target_path)

    # 1. Identify the last column in the source CSV
    last_col_name = df_source.columns[-1]
    print(f"[*] Extracting metric: '{last_col_name}'")

    # 2. Safety check for length mismatch
    if len(df_source) != len(df_target):
        print(f"[!] WARNING: Row counts differ (Source: {len(df_source)}, Target: {len(df_target)}).")
        print("    The script will strictly copy values top-to-bottom. If the source is longer, ")
        print("    extra values will be discarded. If the target is longer, it will have NaNs.")

    # Prevent duplicating the column if the script is run twice
    if last_col_name in df_target.columns:
        print(f"[*] Note: Column '{last_col_name}' already exists in target. Overwriting...")
        df_target = df_target.drop(columns=[last_col_name])

    # 3. Direct Ordered Injection
    # Using .reset_index(drop=True) guarantees that Pandas ignores any internal indices 
    # and simply assigns the values from top to bottom (Row 0 -> Row 0, Row 1 -> Row 1, etc.)
    df_target[last_col_name] = df_source.iloc[:, -1].reset_index(drop=True)

    # 4. Save the result
    save_path = output_path if output_path else target_path
    df_target.to_csv(save_path, index=False)
    print(f"[+] Direct ordered copy complete. Data saved to: {save_path}")

# --- EXECUTION ---
if __name__ == "__main__":
    copy_column_by_order(SOURCE_CSV, TARGET_CSV, OUTPUT_CSV)