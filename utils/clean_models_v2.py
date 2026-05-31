import os

def get_readable_size(size_bytes):
    """Converts bytes to a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def clean_model_folder(folder_path, keep_numbers):
    # Construct the exact filenames to keep based on the 'epoch-n.pt' pattern
    keep_filenames = {f"epoch-{str(n)}.pt" for n in keep_numbers}
    
    total_space_saved = 0
    files_deleted_count = 0
    
    if not os.path.exists(folder_path):
        print(f"Error: Path '{folder_path}' not found.")
        return

    print(f"--- Analysis of: {folder_path} ---")
    print(f"Targeting pattern: epoch-N.pt")
    print(f"Keeping: {', '.join(keep_filenames)}\n")
    
    for filename in os.listdir(folder_path):
        # Only look at files that match the naming convention
        if filename.startswith("epoch-") and filename.endswith(".pt"):
            file_path = os.path.join(folder_path, filename)
            
            if filename in keep_filenames:
                print(f"[KEEPING]  {filename}")
            else:
                file_size = os.path.getsize(file_path)
                total_space_saved += file_size
                files_deleted_count += 1
                
                try:
                    # --- ACTION ZONE ---
                    os.remove(file_path) # <--- UNCOMMENT TO ACTUALLY DELETE
                    print(f"[DELETING] {filename} ({get_readable_size(file_size)})")
                except Exception as e:
                    print(f"[ERROR]    Could not delete {filename}: {e}")

    print("-" * 40)
    # Visual reminder about dry run
    action_taken = "removed" if False else "flagged for deletion (DRY RUN)" 
    
    print(f"Cleanup Finished!")
    print(f"Files {action_taken}: {files_deleted_count}")
    print(f"Total space saved: {get_readable_size(total_space_saved)}")

# --- CONFIGURATION ---
TARGET_FOLDER = "outputs/windy_tsp_100-100/resume_original_wind_20260513T083258" 
MODELS_TO_KEEP = [599, 699, 799, 899, 968, 999] #492, 499 ,599, 699, 799, 883, 899] # The script will look for epoch-99.pt, etc.

if __name__ == "__main__":
    clean_model_folder(TARGET_FOLDER, MODELS_TO_KEEP)