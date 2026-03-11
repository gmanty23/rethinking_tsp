import os
import torch
from torch.utils.data import DataLoader

# Import local utilities from your codebase
from utils.functions import load_model, move_to
from problems.tsp.problem_tsp import TSP  # Or WindyTSP depending on your __init__ routing

def print_single_inference(model_path, dataset_path, neighbors=20, knn_strat='percentage'):
    """
    Loads a sparse model and prints the greedy inference result for a single Windy TSP instance.
    """
    print(f"Loading Model from: {model_path}")
    
    # 1. Load the pre-trained model and its original training arguments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_args = load_model(model_path)
    model.to(device)
    model.eval()
    
    # Set decoding strategy to greedy for deterministic inference
    model.set_decode_type("greedy")
    
    # Determine the feature type dynamically from the training args (e.g., 'hybrid', 'coords')
    feature_type = train_args.get('node_feature_type', 'coords')
    print(f"Model Configuration -> Feature Type: {feature_type} | Message Passing: {train_args.get('gnn_direction_mode', 'forward')}")

    # 2. Instantiate the Dataset
    # CRITICAL: We pass the 'neighbors' argument to enforce the sparse graph topology.
    print(f"Loading single instance from {dataset_path} (Sparsity: k={neighbors})...")
    dataset = model.problem.make_dataset(
        filename=dataset_path, 
        batch_size=1,          # Only need 1 instance
        num_samples=1,         # Only load 1 instance from the file
        neighbors=neighbors, 
        knn_strat=knn_strat, 
        node_feature_type=feature_type,
        supervised=True, 
        nar=False
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(dataloader))
    
    # Move tensors to the appropriate device
    nodes = move_to(batch['nodes'], device)
    graph = move_to(batch['graph'], device)

    # 3. Perform the Forward Pass
    print("\nExecuting Forward Pass...")
    with torch.no_grad():
        # Depending on the architecture, return_pi=True yields the predicted tour
        costs, ll, pi = model(nodes, graph, return_pi=True)
        
    # Extract variables
    tour = pi[0].cpu().numpy()
    cost = costs[0].cpu().item()
    log_likelihood = ll[0].cpu().item()
    
    # Calculate confidence proxy (average probability per node decision)
    confidence = torch.exp(torch.tensor(log_likelihood / len(tour))).item()

    # 4. Output the Results
    print("\n" + "="*50)
    print(" INFERENCE RESULTS (SPARSE MODEL)")
    print("="*50)
    print(f"Predicted Tour ($\pi$):  {tour.tolist()}")
    print(f"Tour Cost:              {cost:.4f}")
    print(f"Sequence Log-Likelihood:{log_likelihood:.4f}")
    print(f"Model Confidence:       {confidence:.2%}")
    print("="*50)

if __name__ == "__main__":
    # --- Configuration ---
    # Replace these paths with your actual model checkpoint and validation data
    MODEL_CHECKPOINT = "/home/pfc/gms/code/rethinking_tsp/outputs/windy_tsp_20-20/008_exp3_graph_sparsification/12KNN_Sparsification/tsp20_forward_hybrid_ent0.05_knn20_20260302T175516"
    DATASET_FILE = "data/windy_tsp/windy_tsp20_val.pkl"
    
    # Adjust K-neighbors based on the sparse model you are testing (e.g., 20)
    K_NEIGHBORS = 20 
    
    print_single_inference(MODEL_CHECKPOINT, DATASET_FILE, neighbors=K_NEIGHBORS)