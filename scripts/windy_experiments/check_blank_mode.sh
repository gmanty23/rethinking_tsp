#!/bin/bash

# Define the temporary Python test file
TEST_FILE="test_blank_fix.py"

echo "=========================================================="
echo " Generating and Running Quick Verification for 'Blank' Mode"
echo "=========================================================="

# Create the Python script
cat << 'EOF' > $TEST_FILE
import torch
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from nets.attention_model import AttentionModel
from nets.encoders.gnn_encoder import GNNEncoder
from problems.tsp.problem_tsp import WindyTSP

def test_blank_mode():
    print("\n[Step 1] Initializing Model with node_feature_type='blank'...")
    
    # Mock Problem
    problem = WindyTSP()
    problem.VEHICLE_CAPACITY = 1.0 # Dummy value to prevent potential attribute errors

    try:
        # Initialize Model
        model = AttentionModel(
            problem=problem,
            embedding_dim=128,
            encoder_class=GNNEncoder,
            n_encode_layers=2,
            n_heads=8,
            node_feature_type='blank',  # <--- THIS IS THE CRITICAL TEST
            gnn_direction_mode='forward'
        )
        print("   -> Success: Model initialized without UnboundLocalError.")

    except Exception as e:
        print(f"   -> FAIL: Initialization crashed!\n      Error: {e}")
        sys.exit(1)

    print("\n[Step 2] Running Forward Pass (Fake Data)...")
    try:
        # Mock Data
        BATCH = 2
        N = 10
        # Random inputs (values don't matter for blank mode, just shape)
        nodes = torch.randn(BATCH, N, 2) 
        graph = torch.zeros(BATCH, N, N, dtype=torch.long)
        
        # We must set decode type to avoid errors during beam search setup
        model.set_decode_type("greedy")
        
        # Forward pass
        # Note: We don't pass cost_matrix here just to test basic structural stability
        _ = model(nodes, graph)
        print("   -> Success: Forward pass completed.")

    except Exception as e:
        print(f"   -> FAIL: Forward pass crashed!\n      Error: {e}")
        sys.exit(1)

    print("\n[Result] PASSED. The 'blank' mode fix is working.\n")

if __name__ == "__main__":
    test_blank_mode()
EOF

# Run the python script
python $TEST_FILE

# Capture exit code
STATUS=$?

# Cleanup
rm $TEST_FILE

if [ $STATUS -eq 0 ]; then
    echo "Verification Successful."
else
    echo "Verification Failed."
fi
