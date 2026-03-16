import os
import re
import math
from pathlib import Path
from tqdm import tqdm

def natural_sort_key(s):
    """Sorts strings containing numbers in human-readable order."""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', str(s))]

def get_size_format(b, factor=1024, suffix="B"):
    """Scales bytes to its proper format (e.g., KB, MB, GB)."""
    for unit in ["", "K", "M", "G", "T", "P"]:
        if b < factor:
            return f"{b:.2f}{unit}{suffix}"
        b /= factor

def prune_checkpoints(root_path, keep_first=5, keep_last=5, dry_run=True):
    root = Path(root_path)
    if not root.exists():
        print(f"Error: Path {root_path} not found.")
        return

    # Statistics for the final report
    stats = {"folders_processed": 0, "files_deleted": 0, "bytes_saved": 0}
    
    # We first collect all subdirectories to initialize the progress bar accurately
    all_subdirs = [Path(d[0]) for d in os.walk(root)]
    
    print(f"\n{'[DRY RUN]' if dry_run else '[EXECUTION MODE]'} Initializing Pruning...")
    print(f"Target: {root.resolve()}\n")

    # Wrap the directory iterator with tqdm for visual feedback
    for subdir in tqdm(all_subdirs, desc="Processing Directories", unit="dir"):
        # List all files in the current subdir
        try:
            files = [f for f in os.listdir(subdir) if os.path.isfile(subdir / f)]
        except PermissionError:
            continue

        # Target only .pt files (keeping args.json and logs safe)
        checkpoint_files = [f for f in files if f.endswith('.pt')]
        
        if len(checkpoint_files) <= (keep_first + keep_last):
            continue

        # Sort naturally (epoch-1 before epoch-10)
        checkpoint_files.sort(key=natural_sort_key)
        
        # Determine files to keep and files to delete
        to_keep = set(checkpoint_files[:keep_first] + checkpoint_files[-keep_last:])
        to_delete = [f for f in checkpoint_files if f not in to_keep]

        stats["folders_processed"] += 1

        for file_name in to_delete:
            file_path = subdir / file_name
            file_size = file_path.stat().st_size
            
            stats["files_deleted"] += 1
            stats["bytes_saved"] += file_size

            if not dry_run:
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"\nError deleting {file_name}: {e}")
    
    # Final Summary Report
    print(f"\n{'-'*40}")
    print(f"Cleanup Summary ({'DRY RUN' if dry_run else 'COMPLETE'}):")
    print(f"  Directories Pruned: {stats['folders_processed']}")
    print(f"  Files {'to be' if dry_run else ''} deleted: {stats['files_deleted']}")
    print(f"  Estimated Space {'to be' if dry_run else ''} saved: {get_size_format(stats['bytes_saved'])}")
    print(f"{'-'*40}\n")

if __name__ == "__main__":
    # CONFIGURATION
    OUTPUTS_DIR = "outputs" 
    KEEP_START = 5
    KEEP_END = 5
    IS_DRY_RUN = False  # Toggle to False to execute deletion

    prune_checkpoints(OUTPUTS_DIR, KEEP_START, KEEP_END, IS_DRY_RUN)