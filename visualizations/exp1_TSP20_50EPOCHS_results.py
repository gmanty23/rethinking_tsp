# Two methods for visualizing results from experiment 1 (TSP20, 50 epochs)

# 1. Tensorboard (Interactive Dashboard)
#    Run the following command in terminal:
#       tensorboard --logdir logs/windy_tsp_20-20
#    Then open the provided URL in a web browser.

# 2. Matplotlib (Static Plots)
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalars(log_dir, scalar_tag='val1/avg_reward'):
    """
    Reads TensorBoard event files and extracts specific scalar data.
    """
    # Find event file
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        print(f"Warning: No event files found in {log_dir}")
        return [], []

    # Use the most recent file if multiple exist
    event_file = max(event_files, key=os.path.getctime)
    print(f"  Loading: {os.path.basename(event_file)}")
    
    # Load the event file
    try:
        ea = EventAccumulator(event_file)
        ea.Reload()
    except Exception as e:
        print(f"  Error loading event file: {e}")
        return [], []

    # Check available tags
    # TensorBoard stores tags in a dictionary structure inside the reservoir
    # We need to robustly check if our tag exists
    tags = ea.Tags()['scalars']
    
    if scalar_tag not in tags:
        print(f"  Warning: Tag '{scalar_tag}' not found.")
        # print(f"  Available tags: {tags}") # Uncomment to debug
        return [], []

    try:
        events = ea.Scalars(scalar_tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    except KeyError:
        print(f"  Error: Key {scalar_tag} found in tags list but failed to retrieve.")
        return [], []

def plot_results():
    # Setup styling
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    # Define the experiments we want to compare
    # We look for folders in logs/windy_tsp_20-20/ that contain these keywords
    experiments = {
        'coords': {'label': 'Baseline (Coords Only)', 'color': 'red', 'style': '--'},
        'learned': {'label': 'Topological (Learned Only)', 'color': 'blue', 'style': '-.'},
        'hybrid': {'label': 'Hybrid (Proposed)', 'color': 'green', 'style': '-'}
    }

    base_log_dir = "logs/windy_tsp_20-20"
    
    if not os.path.exists(base_log_dir):
        print(f"Error: Log directory '{base_log_dir}' does not exist.")
        return

    # Find the specific run folders (handling the timestamps)
    all_runs = os.listdir(base_log_dir)
    
    found_data = False

    for key, config in experiments.items():
        # Find directory matching "exp1_{key}_"
        # We sort to pick the latest one if you ran it multiple times
        matching_runs = sorted([r for r in all_runs if f"exp1_{key}_" in r])
        
        if not matching_runs:
            print(f"Could not find run for: {key}")
            continue

        latest_run = matching_runs[-1]
        full_path = os.path.join(base_log_dir, latest_run)
        print(f"Processing: {latest_run}")

        # Extract Validation Cost
        steps, costs = extract_scalars(full_path, scalar_tag='val1/avg_reward')
        
        if len(steps) > 0:
            found_data = True
            plt.plot(steps, costs, 
                     label=config['label'], 
                     color=config['color'], 
                     linestyle=config['style'], 
                     linewidth=2)
        else:
            print(f"  No data points found yet for {key}.")

    if not found_data:
        print("\nNo data found to plot yet! (Training might be too early)")
        return

    plt.title("Experiment 1: Information Study (Windy TSP-20)", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Validation Cost (Lower is Better)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    output_file = "experiment_1_results.png"
    plt.savefig(output_file, dpi=300)
    print(f"\n[Success] Plot saved to {output_file}")

if __name__ == "__main__":
    plot_results()