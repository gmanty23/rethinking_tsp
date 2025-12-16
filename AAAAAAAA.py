import json

# Map the filenames to the keys you want in the final JSON
file_map = {
    "coords": "exp1_edges&blank_exp1_coords_valavgreward.json",
    "blank":  "exp1_edges&blank_exp1_blank_valavgreward.json",
    "hybrid": "exp1_edges&blank_exp1_hybrid_valavgreward.json",
    "learned": "exp1_edges&blank_exp1_learned_valavgreward.json"
}

combined_data = {}

try:
    for key, filename in file_map.items():
        with open(filename, 'r') as f:
            data = json.load(f)
            combined_data[key] = data
            print(f"Successfully loaded {key} from {filename}")

    # Write the combined result to a new file
    output_filename = "combined_valavgreward.json"
    with open(output_filename, 'w') as f_out:
        json.dump(combined_data, f_out, indent=None) # indent=None keeps file size smaller
    
    print(f"\nSuccess! Merged data saved to: {output_filename}")

except FileNotFoundError as e:
    print(f"Error: Could not find file. Make sure all JSON files are in this folder.\n{e}")
except Exception as e:
    print(f"An error occurred: {e}")