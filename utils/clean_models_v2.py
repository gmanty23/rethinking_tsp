# Explanation of the script:
# This script is designed to clean up a folder containing model checkpoint files named in the format "epoch-N.pt". 
# It will keep only the specified epochs and delete the rest, while providing a summary of the actions taken and the space saved.
# If SEARCH_SUBFOLDERS is True, it will also scan and clean all nested directories inside the target folder.

import os

def get_readable_size(size_bytes):
    """Converts bytes to a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def clean_model_folder(folder_path, keep_numbers, recursive=False):
    # Construct the exact filenames to keep based on the 'epoch-n.pt' pattern
    keep_filenames = {f"epoch-{str(n)}.pt" for n in keep_numbers}
    
    total_space_saved = 0
    files_deleted_count = 0
    
    if not os.path.exists(folder_path):
        print(f"Error: Path '{folder_path}' not found.")
        return

    print(f"--- Analysis of: {folder_path} ---")
    print(f"Targeting pattern: epoch-N.pt")
    print(f"Searching Subfolders: {recursive}")
    print(f"Keeping: {', '.join(keep_filenames)}\n")
    
    # Gather all files based on the recursive flag
    files_to_check = []
    if recursive:
        # os.walk travels down through every subfolder
        for root, _, files in os.walk(folder_path):
            for f in files:
                files_to_check.append((root, f))
    else:
        # os.listdir only grabs what is directly inside the top folder
        for f in os.listdir(folder_path):
            files_to_check.append((folder_path, f))

    for directory, filename in files_to_check:
        # Only look at files that match the naming convention
        if filename.startswith("epoch-") and filename.endswith(".pt"):
            file_path = os.path.join(directory, filename)
            
            # Ensure we are looking at a file and not a weirdly named directory
            if not os.path.isfile(file_path):
                continue
            
            if filename in keep_filenames:
                print(f"[KEEPING]  {file_path}")
            else:
                file_size = os.path.getsize(file_path)
                total_space_saved += file_size
                files_deleted_count += 1
                
                try:
                    # --- ACTION ZONE ---
                    os.remove(file_path) # <--- UNCOMMENT TO ACTUALLY DELETE
                    print(f"[DELETING] {file_path} ({get_readable_size(file_size)})")
                except Exception as e:
                    print(f"[ERROR]    Could not delete {file_path}: {e}")

    print("-" * 40)
    # Visual reminder about dry run
    action_taken = "removed" if False else "flagged for deletion (DRY RUN)" 
    
    print(f"Cleanup Finished!")
    print(f"Files {action_taken}: {files_deleted_count}")
    print(f"Total space saved: {get_readable_size(total_space_saved)}")

# --- CONFIGURATION ---
TARGET_FOLDER = "outputs/FINAL_PAPER/V3-TSP_SIZES_ablation/windy_tsp_50-50/NAB-CLIPPED-V3_TSP_SIZES_mlp-standard_original_hybrid_nab-none_tsp50_ent0.05_neighbors100_strat-cost_weighted_percentage_dir-dual_layers-3_emb-original_feat-hybrid_20260712T205758" 
MODELS_TO_KEEP = [0, 24, 49, 74, 94, 99] # The script will look for epoch-99.pt, etc.
SEARCH_SUBFOLDERS = False # Set to True to clean all nested folders, False for just the root folder

if __name__ == "__main__":
    clean_model_folder(TARGET_FOLDER, MODELS_TO_KEEP, recursive=SEARCH_SUBFOLDERS)