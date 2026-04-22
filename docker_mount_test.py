import torch
import torch.nn as nn
import time

def main():
    print(f"--- Environment Check ---")
    print(f"PyTorch Version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Simulate the Windy TSP Data
    batch_size = 32
    num_nodes = 20
    # Create random directed edge costs (Asymmetric: cost[i, j] != cost[j, i])
    print("\nGenerating dummy asymmetric cost matrices...")
    costs = torch.rand(batch_size, num_nodes, num_nodes, device=device)

    # 2. Define a minimal dummy network
    class DummyEncoder(nn.Module):
        def __init__(self, hidden_dim=128):
            super().__init__()
            # A simple linear projection to simulate node embeddings
            self.edge_projector = nn.Linear(num_nodes, hidden_dim)

        def forward(self, x):
            return torch.relu(self.edge_projector(x))

    model = DummyEncoder().to(device)

    # 3. Test modern PyTorch optimization (Triton kernel compilation)
    print("Compiling model via torch.compile() (this may take a moment)...")
    compiled_model = torch.compile(model)

    # 4. Execute the forward pass on the RTX 5080
    print("\nExecuting forward pass...")
    start_time = time.time()
    
    # Run a few warmup iterations (standard practice for compiled kernels)
    for _ in range(5):
        _ = compiled_model(costs)
        
    # Measure the actual run
    torch.cuda.synchronize() # Wait for GPU to finish
    output = compiled_model(costs)
    torch.cuda.synchronize()
    
    end_time = time.time()

    print(f"\n--- Success! ---")
    print(f"Input shape (Batch, Nodes, Nodes): {costs.shape}")
    print(f"Output shape (Batch, Nodes, Embedding): {output.shape}")
    print(f"Execution time (after compilation): {end_time - start_time:.5f} seconds")

if __name__ == "__main__":
    main()