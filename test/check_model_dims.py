#!/usr/bin/env python

import os
import torch
import argparse
import sys

def check_dims(root_dir):
    """
    Walks through the directory, finds only 'epoch-99.pt' files, 
    and reports the input dimension size.
    """
    print(f"\nScanning directory: {root_dir}")
    print("="*100)
    print(f"{'Model Path (Relative)':<80} | {'Dim'}")
    print("="*100)
    
    # Counter
    found_count = 0
    
    # Sort directories to make output deterministic and grouped by experiment
    for root, dirs, files in os.walk(root_dir):
        dirs.sort() 
        files.sort()
        
        for file in files:
            # STRICT FILTER: Only check epoch-99.pt
            if file == "epoch-99.pt":
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, root_dir)
                
                try:
                    # Load with weights_only=False to fix the PyTorch 2.6 security error
                    checkpoint = torch.load(full_path, map_location='cpu', weights_only=False)
                    
                    # Extract State Dict
                    state_dict = checkpoint.get('model', checkpoint)
                    
                    # Check for the embedding layer
                    if 'init_embed.weight' in state_dict:
                        # Shape is [Hidden_Dim, Input_Dim]
                        weight_shape = state_dict['init_embed.weight'].shape
                        input_dim = weight_shape[1]
                        
                        # Print row with JUST the number
                        print(f"{rel_path:<80} | {input_dim}")
                        found_count += 1
                        
                    else:
                        print(f"{rel_path:<80} | ???")

                except Exception as e:
                    print(f"{rel_path:<80} | Error")

    print("="*100)
    print(f"Scan Complete. Found {found_count} models.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check input dimensions of trained models.")
    parser.add_argument("dir", nargs='?', default="outputs", help="Root directory to search (default: outputs)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.dir):
        print(f"Error: Directory '{args.dir}' not found.")
        sys.exit(1)
        
    check_dims(args.dir)