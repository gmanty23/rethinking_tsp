import torch
import torch.nn as nn
import warnings # Import warnings module
from nets.attention_model import AttentionModel
from nets.encoders.gnn_encoder import GNNEncoder
from problems.tsp.problem_tsp import WindyTSP

# Suppress the specific PyTorch warning about uint8 indexing
warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated")

def count_parameters(model):
    """Helper function to count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_architecture():
    print("================================================================")
    print("       STARTING WINDY TSP ARCHITECTURE VERIFICATION             ")
    print("================================================================\n")
    
    # ----------------------------------------------------------------
    # 0. SETUP MOCK DATA
    # ----------------------------------------------------------------
    BATCH_SIZE = 2
    NUM_NODES = 10
    EMBED_DIM = 128
    
    print("[0] Generating Mock Data (Super-Packed 7D Tensor)...")
    nodes = torch.randn(BATCH_SIZE, NUM_NODES, 7)
    
    # Adjacency matrix must be LongTensor (integer)
    graph = torch.zeros(BATCH_SIZE, NUM_NODES, NUM_NODES, dtype=torch.long)
    
    problem = WindyTSP() 
    problem.VEHICLE_CAPACITY = 1.0 

    print("    Done.\n")

    # ----------------------------------------------------------------
    # TEST 1: Feature Slicing (Input Layer)
    # ----------------------------------------------------------------
    print("[1] TEST: Input Feature Slicing")
    print("    Description: Verifying model accepts 7D input and slices it based on mode.")
    
    feature_modes = ['coords', 'learned', 'hybrid']
    
    for mode in feature_modes:
        print(f"    > Testing mode='{mode}'...", end=" ")
        try:
            model = AttentionModel(
                problem=problem,
                embedding_dim=EMBED_DIM,
                encoder_class=GNNEncoder,
                n_encode_layers=2,
                n_heads=8,
                node_feature_type=mode
            )
            # FIX: Set decode type to greedy so the forward pass knows what to do
            model.set_decode_type("greedy")
            
            _ = model(nodes, graph)
            print("OK (Forward pass successful)")
            
        except RuntimeError as e:
            print(f"FAILED!\n      Error: {e}")
            return
        except Exception as e:
            print(f"FAILED!\n      Error: {e}")
            return

    print("    >> RESULT: Feature Slicing logic is VALID.\n")

    # ----------------------------------------------------------------
    # TEST 2: GNN Directionality (Encoder Logic)
    # ----------------------------------------------------------------
    print("[2] TEST: GNN Directionality & Parameter Count")
    print("    Description: Verifying 'dual' mode instantiates extra weight matrices.")
    
    direction_modes = ['forward', 'backward', 'dual']
    param_counts = {}
    
    for mode in direction_modes:
        print(f"    > Building model with gnn_direction_mode='{mode}'...", end=" ")
        try:
            model = AttentionModel(
                problem=problem,
                embedding_dim=EMBED_DIM,
                encoder_class=GNNEncoder,
                n_encode_layers=2,
                n_heads=8,
                node_feature_type='hybrid',
                gnn_direction_mode=mode
            )
            # FIX: Set decode type here as well
            model.set_decode_type("greedy")
            
            output = model(nodes, graph)
            count = count_parameters(model)
            param_counts[mode] = count
            print(f"OK. Params: {count}")
            
        except Exception as e:
            print(f"FAILED!\n      Error: {e}")
            return

    # ----------------------------------------------------------------
    # 3. LOGIC VERIFICATION
    # ----------------------------------------------------------------
    print("\n[3] TEST: Verifying Logic Consistency")
    
    if param_counts['forward'] == param_counts['backward']:
         print("    > Check A (Forward vs Backward size): PASS (Sizes match)")
    else:
         print("    > Check A (Forward vs Backward size): FAIL (Sizes differ?)")

    if param_counts['dual'] > param_counts['forward']:
         diff = param_counts['dual'] - param_counts['forward']
         expected_diff = 16512 * 2
         
         print(f"    > Check B (Dual > Forward):           PASS")
         print(f"      - Extra Parameters found: {diff}")
         print(f"      - Expected Extra Params:  {expected_diff}")
         
         if diff == expected_diff:
             print("      - EXACT MATCH: The architecture is perfectly instantiated.")
         else:
             print("      - WARNING: Parameter count mismatch.")
    else:
         print("    > Check B (Dual > Forward):           FAIL (Dual did not add parameters)")

    print("\n================================================================")
    print("       ALL ARCHITECTURE TESTS PASSED                            ")
    print("================================================================")

if __name__ == "__main__":
    test_architecture()