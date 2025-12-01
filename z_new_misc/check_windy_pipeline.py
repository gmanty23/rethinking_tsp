import torch
import numpy as np
import os
import pickle
from problems.tsp.problem_tsp import TSPDataset, WindyTSP

def test_pipeline():
    print("=== Starting Windy TSP Pipeline Check ===\n")

    # ----------------------------------------------------------------
    # 1. CREATE MOCK DATA
    # ----------------------------------------------------------------
    filename = "debug_windy.pkl"
    print(f"[1] Generating mock data -> {filename}")
    
    # Let's create a specific scenario where we KNOW the answer.
    # Node 0 at (0,0), Node 1 at (1,0). Distance = 1.0.
    # Wind blowing East (1,0).
    # Moving 0->1 is WITH wind (Tailwind).
    # Moving 1->0 is AGAINST wind (Headwind).
    
    mock_instance = {
        'loc': np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), # 3 nodes
        'wind': np.array([1.0, 0.0]), # Wind blowing East (strong!)
        'alpha': 1.0
    }
    
    with open(filename, 'wb') as f:
        pickle.dump([mock_instance], f) # List of 1 instance

    # ----------------------------------------------------------------
    # 2. TEST DATASET LOADER
    # ----------------------------------------------------------------
    print("[2] Initializing TSPDataset (Hybrid Mode)...")
    
    # FIX: Explicitly set batch_size=1 and num_samples=1 to match our mock data
    dataset = TSPDataset(
        filename=filename, 
        node_feature_type='hybrid', 
        batch_size=1, 
        num_samples=1
    )
    
    item = dataset[0] # Get the first (and only) item
    nodes = item['nodes'] # Should be Tensor
    
    print(f"    Node Tensor Shape: {nodes.shape} (Expected: [3, 7])")
    assert nodes.shape == (3, 7), "Shape mismatch!"
    
    # Check Features
    # [x, y, wx, wy, alpha, stat_out, stat_in]
    print(f"    Node 0 Features: {nodes[0].tolist()}")
    
    # Verify Packing
    assert nodes[0, 2].item() == 1.0, "Wind X not packed at index 2"
    assert nodes[0, 3].item() == 0.0, "Wind Y not packed at index 3"
    assert nodes[0, 4].item() == 1.0, "Alpha not packed at index 4"
    assert nodes[0, 5].item() != 0.0, "Stats (Out) appear to be zero/empty"
    
    print("    >> Feature Packing: SUCCESS")

    # ----------------------------------------------------------------
    # 3. TEST PHYSICS ENGINE (get_costs)
    # ----------------------------------------------------------------
    print("\n[3] Testing WindyTSP.get_costs...")
    
    # We will simulate a batch of size 1 containing our mock instance
    batch_nodes = nodes.unsqueeze(0) # (1, 3, 7)
    
    # Create a tour: 0 -> 1 -> 2 -> 0
    pi = torch.tensor([[0, 1, 2]]) # (1, 3)
    
    # Call the static method
    total_cost, _ = WindyTSP.get_costs(batch_nodes, pi)
    
    print(f"    Calculated Total Cost: {total_cost.item():.4f}")
    
    # --- MANUAL VERIFICATION ---
    # Edge 0 (0,0) -> 1 (1,0):
    #   Dist = 1.0
    #   Dir = (1, 0)
    #   Wind = (1, 0)
    #   Proj = 1*1 + 0*0 = 1.0
    #   Exp = exp(-1.0 * 1.0) = exp(-1) ≈ 0.3679
    cost_01 = 1.0 * np.exp(-1.0)
    
    # Edge 1 (1,0) -> 2 (0,1):
    #   Dist = sqrt(2) ≈ 1.414
    #   Dir = (-0.707, 0.707)
    #   Wind = (1, 0)
    #   Proj = -0.707
    #   Exp = exp(-1.0 * -0.707) = exp(0.707) ≈ 2.028
    cost_12 = 1.4142 * np.exp(0.7071)
    
    # Edge 2 (0,1) -> 0 (0,0) (Return trip):
    #   Dist = 1.0
    #   Dir = (0, -1)
    #   Wind = (1, 0)
    #   Proj = 0
    #   Exp = exp(0) = 1.0
    cost_20 = 1.0 * 1.0
    
    expected_sum = cost_01 + cost_12 + cost_20
    print(f"    Manual Expected Cost:  {expected_sum:.4f}")
    
    diff = abs(total_cost.item() - expected_sum)
    if diff < 1e-4:
        print("    >> Physics Calculation: SUCCESS (Matches Manual)")
    else:
        print(f"    >> Physics Calculation: FAILED (Diff {diff})")
        
    # ----------------------------------------------------------------
    # 4. TEST ASYMMETRY
    # ----------------------------------------------------------------
    print("\n[4] Testing Asymmetry (0->1 vs 1->0)...")
    
    # Tour A: 0 -> 1 -> 2
    tour_A = torch.tensor([[0, 1, 2]])
    # Tour B: 0 -> 2 -> 1 (This traverses 1->0 at the end)
    tour_B = torch.tensor([[0, 2, 1]])
    
    cost_A, _ = WindyTSP.get_costs(batch_nodes, tour_A)
    cost_B, _ = WindyTSP.get_costs(batch_nodes, tour_B)
    
    print(f"    Tour A (0->1->2->0) Cost: {cost_A.item():.4f}")
    print(f"    Tour B (0->2->1->0) Cost: {cost_B.item():.4f}")
    
    if abs(cost_A.item() - cost_B.item()) > 1e-4:
        print("    >> Asymmetry Check: SUCCESS (Costs are different)")
    else:
        print("    >> Asymmetry Check: FAILED (Costs are identical - Physics might be broken)")

    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)

if __name__ == "__main__":
    test_pipeline()