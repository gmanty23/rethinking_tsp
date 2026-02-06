import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

# ==========================================
# 1. PASTE THE NEW FUNCTION HERE TO TEST IT
# ==========================================
def get_wind_knn_graph(nodes, neighbors, knn_strat, cost_matrix):
    """
    Calculates KNN based on Asymmetric Cost Matrix (Wind).
    """
    num_nodes = len(nodes)
    
    # 1. Determine k
    if knn_strat == 'percentage':
        k = int(num_nodes * neighbors)
    else:
        k = neighbors
        
    if k >= num_nodes - 1 or k == -1:
        return np.zeros((num_nodes, num_nodes))

    # 2. Calculate Nearest Neighbors based on COST
    # We use the cost_matrix directly.
    # argsort(axis=1) sorts the destinations for each source node by cost.
    # We take indices 1 to k+1 (skipping index 0 which is self, assuming self-cost is lowest)
    knn_indices = np.argsort(cost_matrix, axis=1)[:, 1:k+1] 

    # 3. Build Adjacency Matrix
    # 1 = No Edge, 0 = Edge Exists
    W = np.ones((num_nodes, num_nodes))
    for i in range(num_nodes):
        W[i, knn_indices[i]] = 0
        
    return W

# ==========================================
# 2. STANDARD KNN (FOR COMPARISON)
# ==========================================
def standard_knn_graph(nodes, neighbors, knn_strat):
    num_nodes = len(nodes)
    if knn_strat == 'percentage':
        k = int(num_nodes * neighbors)
    else:
        k = neighbors
    
    dist = squareform(pdist(nodes, metric='euclidean'))
    knn_indices = np.argsort(dist, axis=1)[:, 1:k+1]

    W = np.ones((num_nodes, num_nodes))
    for i in range(num_nodes):
        W[i, knn_indices[i]] = 0
    return W

# ==========================================
# 3. THE TEST SCENARIO
# ==========================================
def run_test():
    print("--- Setting up Wind Trap Scenario ---")
    
    # Coordinates: Node 0 is at (0,0)
    # Node 1 is CLOSE at (1,0)
    # Node 2 is FAR at (10,0)
    nodes = np.array([
        [0, 0],  # Node 0 (Source)
        [1, 0],  # Node 1 (Close Neighbor)
        [10, 0], # Node 2 (Distant Neighbor)
        [0, 5]   # Node 3 (Dummy)
    ])
    
    # Manually define the Cost Matrix
    # Rows = From, Cols = To
    num_nodes = 4
    costs = np.zeros((num_nodes, num_nodes))
    
    # --- The TRAP ---
    # Cost 0->1 is HIGH (Headwind), even though distance is small (1.0)
    costs[0, 1] = 100.0 
    
    # Cost 0->2 is LOW (Tailwind), even though distance is large (10.0)
    costs[0, 2] = 5.0 
    
    # Fill random costs for others just to be complete
    costs[0, 3] = 50.0
    costs[1, 0] = 50.0
    costs[2, 0] = 50.0
    
    print(f"Distance 0->1: {np.linalg.norm(nodes[0]-nodes[1])} (Close)")
    print(f"Cost     0->1: {costs[0,1]} (Expensive!)\n")
    
    print(f"Distance 0->2: {np.linalg.norm(nodes[0]-nodes[2])} (Far)")
    print(f"Cost     0->2: {costs[0,2]} (Cheap!)\n")

    # ==========================================
    # 4. RUN COMPARISON
    # ==========================================
    k = 1 # We only want the single "best" neighbor
    
    print(f"--- Running KNN (k={k}) ---\n")

    # A. Standard Euclidean KNN
    w_std = standard_knn_graph(nodes, neighbors=k, knn_strat=None)
    neighbor_std = np.where(w_std[0] == 0)[0][0] # Find who Node 0 connected to
    
    print(f"Standard KNN chose neighbor: Node {neighbor_std}")
    if neighbor_std == 1:
        print("-> CORRECT behavior for Standard KNN (Picked closest distance).")
    else:
        print("-> WRONG behavior for Standard KNN.")

    # B. Wind-Aware KNN (Your New Function)
    w_wind = get_wind_knn_graph(nodes, neighbors=k, knn_strat=None, cost_matrix=costs)
    neighbor_wind = np.where(w_wind[0] == 0)[0][0] # Find who Node 0 connected to
    
    print(f"Wind KNN chose neighbor:     Node {neighbor_wind}")
    
    if neighbor_wind == 2:
        print("-> SUCCESS: Wind KNN ignored distance and picked the cheapest cost!")
    elif neighbor_wind == 1:
        print("-> FAILURE: Wind KNN picked the closest node despite the high cost.")
    else:
        print(f"-> FAILURE: Wind KNN picked Node {neighbor_wind} (Unexpected).")

if __name__ == "__main__":
    run_test()