from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search
# Import the generator logic locally
from data.windy_tsp.generate_windy_tsp import generate_windy_instance


def nearest_neighbor_graph(nodes, neighbors, knn_strat):
    """Returns k-Nearest Neighbor graph as a **NEGATIVE** adjacency matrix
    """
    num_nodes = len(nodes)
    
    # 1. Determine number of neighbors k
    if knn_strat in ['percentage', 'random_percentage', 'cost_weighted_percentage']:
        k = int(num_nodes * neighbors)
    else:
        k = int(neighbors)
        
    if k >= num_nodes - 1 or k == -1:
        W = np.zeros((num_nodes, num_nodes))
    else:
        # Compute distance matrix (Euclidean cost for standard TSP)
        W_val = squareform(pdist(nodes, metric='euclidean'))
        W = np.ones((num_nodes, num_nodes))
        
        # 2. Connection Selection Logic
        if knn_strat in ['random', 'random_percentage']:
            # --- Uniform Random Sparsification ---
            for i in range(num_nodes):
                valid_neighbors = np.delete(np.arange(num_nodes), i)
                rand_indices = np.random.choice(valid_neighbors, size=k, replace=False)
                W[i, rand_indices] = 0
                
        elif knn_strat == 'cost_weighted_percentage':
            # --- Cost-Proportional Random Sparsification (Distance-based) ---
            for i in range(num_nodes):
                valid_neighbors = np.delete(np.arange(num_nodes), i)
                distances = W_val[i, valid_neighbors]
                
                # Inverse distance weighting: shorter distance = higher probability
                weights = 1.0 / (distances + 1e-8)
                probs = weights / np.sum(weights)
                
                rand_indices = np.random.choice(valid_neighbors, size=k, replace=False, p=probs)
                W[i, rand_indices] = 0
                
        else:
            # --- BASELINE: Deterministic kNN ---
            # Sort distances to grab the top 'k' closest (skipping self, which is index 0)
            knns = np.argsort(W_val, axis=-1)[:, 1:k+1]
            for idx in range(num_nodes):
                W[idx][knns[idx]] = 0

    # Remove self-connections (1 = disconnected/masked)
    np.fill_diagonal(W, 1)
    return W


def tour_nodes_to_W(tour_nodes):
    """Computes edge adjacency matrix representation of tour
    """
    num_nodes = len(tour_nodes)
    tour_edges = np.zeros((num_nodes, num_nodes))
    for idx in range(len(tour_nodes) - 1):
        i = tour_nodes[idx]
        j = tour_nodes[idx + 1]
        tour_edges[i][j] = 1
        tour_edges[j][i] = 1
    # Add final connection
    tour_edges[j][tour_nodes[0]] = 1
    tour_edges[tour_nodes[0]][j] = 1
    return tour_edges

def get_wind_knn_graph(nodes, neighbors, knn_strat, cost_matrix):
    """
    Calculates Graph Mask based on Asymmetric Cost Matrix (Wind) OR Random Connections.
    Ensures all nodes have a minimum in-degree to maintain reachability.
    """
    num_nodes = len(nodes)
    
    # 1. Determine number of neighbors k
    if knn_strat in ['percentage', 'random_percentage', 'cost_weighted_percentage']:
        k = int(num_nodes * neighbors)
    else:
        k = int(neighbors)
    
    # 2. Guard clause: If k is too high, return fully connected (all zeros)
    if k >= num_nodes - 1 or k == -1:
        return np.zeros((num_nodes, num_nodes))
        
    # 3. Build the Adjacency Matrix
    # Start with all 1s (1 = disconnected/masked in this architecture)
    W = np.ones((num_nodes, num_nodes))
    
    # 4. Connection Selection Logic
    if knn_strat in ['random', 'random_percentage']:
        # --- ABLATION: Uniform Random Sparsification ---
        for i in range(num_nodes):
            valid_neighbors = np.delete(np.arange(num_nodes), i)
            rand_indices = np.random.choice(valid_neighbors, size=k, replace=False)
            W[i, rand_indices] = 0 # 0 = edge exists

    # (AQUÍ): We add 'ane' to your existing cost-weighted logic!
    elif knn_strat in ['cost_weighted_percentage', 'ane']:
        # --- ANE / Cost-Proportional Random Sparsification ---
        for i in range(num_nodes):
            valid_neighbors = np.delete(np.arange(num_nodes), i)
            
            # Extract costs to valid neighbors
            costs_to_neighbors = cost_matrix[i, valid_neighbors]
            
            # Inverse cost weighting: lower cost = higher weight
            # Add epsilon to prevent division by zero
            weights = 1.0 / (costs_to_neighbors + 1e-8) 
            
            # Normalize to create a probability distribution that sums to 1
            probs = weights / np.sum(weights)
            
            # Sample 'k' neighbors without replacement based on their cost probabilities
            rand_indices = np.random.choice(valid_neighbors, size=k, replace=False, p=probs)
            W[i, rand_indices] = 0

    else:
        # --- BASELINE: Cost-aware kNN ---
        knn_indices = np.argsort(cost_matrix, axis=1)[:, 1:k+1]
        for i in range(num_nodes):
            W[i, knn_indices[i]] = 0 
            
    # --- ENFORCE MINIMUM IN-DEGREE ---
    min_in_edges = max(1, int(0.1 * k)) 
    in_degrees = np.sum(W == 0, axis=0)
    isolated_nodes = np.where(in_degrees == 0)[0] 
    
    for j in isolated_nodes:
        valid_origins = np.delete(np.arange(num_nodes), j)

        if knn_strat in ['random', 'random_percentage']:
            # --- Random Reconnection ---
            best_origins = np.random.choice(valid_origins, size=min_in_edges, replace=False)

        # (AQUÍ): We add 'ane' here as well to ensure minimum reachability
        elif knn_strat in ['cost_weighted_percentage', 'ane']:
            # --- ANE / Cost-Proportional Reconnection ---
            costs_from_origins = cost_matrix[valid_origins, j]
            
            weights = 1.0 / (costs_from_origins + 1e-8)
            probs = weights / np.sum(weights)
            
            best_origins = np.random.choice(valid_origins, size=min_in_edges, replace=False, p=probs)

        else:
            # --- Cost-Aware Reconnection ---
            costs_to_j = cost_matrix[:, j].copy()
            costs_to_j[j] = np.inf 
            best_origins = np.argsort(costs_to_j)[:min_in_edges]
            
        # Forcibly open the edges from the selected origins to j
        W[best_origins, j] = 0
            
    return W

class TSP(object):
    """Class representing the Standard Symmetric Travelling Salesman Problem
    """

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        """Returns TSP tour length for given graph nodes and tour permutations
        
        Args:
            dataset: graph nodes (torch.Tensor) [Batch, N, 2]
            pi: node permutations representing tours (torch.Tensor) [Batch, N]

        Returns:
            TSP tour length, None
        """
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour:\n{}\n{}".format(dataset, pi)

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Standard Euclidean Distance (L2 Norm)
        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(nodes, graph, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096, 
                    cost_matrix=None): # 1. ADD THIS ARGUMENT
        """Method to call beam search, given TSP samples and a model
        """

        assert model is not None, "Provide model"

        # 2. PASS IT DOWN TO PRECOMPUTE_FIXED
        fixed = model.precompute_fixed(nodes, graph, cost_matrix=cost_matrix)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            nodes, graph, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class WindyTSP(TSP):
    """
    Class representing the Windy (Asymmetric) TSP.
    Inherits State and Beam Search logic from TSP, but overrides cost calculation.
    """
    NAME = 'windy_tsp'

    @staticmethod
    def get_costs(dataset, pi):
        """
        Calculates tour cost using the Asymmetric Exponential Formula.
        
        Args:
            dataset: Tensor of shape (Batch, N, Features)
                     We expect the first 5 features to be [x, y, wind_x, wind_y, alpha]
            pi: Tensor of shape (Batch, N) containing the tour indices
        """
        # Validity check (inherited logic)
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # 1. Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Check for correct data shape
        if d.size(-1) == 2:
            # Fallback for standard coords being passed to windy problem
            # This happens if the baseline generator makes standard coords
            # We calculate Euclidean cost to prevent crash
            return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

        # 2. Extract Physics Variables (Indices 0 to 4)
        # Even if we added more features at the end (stats), the physics is always at the start.
        coords = d[..., :2]          # (Batch, N, 2)
        wind = d[..., 2:4]           # (Batch, N, 2)
        alpha = d[..., 4:5]          # (Batch, N, 1)

        # 3. Prepare Segments (From -> To)
        # We need the "next" node for every "current" node to form edges
        # Rolling -1 means index i aligns with i+1
        coords_next = torch.roll(coords, -1, dims=1)

        # 4. Calculate Vectors and Distances
        diff = coords_next - coords  # Vector u_ij (Batch, N, 2)
        dist = diff.norm(p=2, dim=2) # Euclidean Distance D_ij (Batch, N)

        # 5. Normalized Direction Unit Vectors
        # Clamp distance to avoid division by zero
        dist_clamped = torch.clamp(dist, min=1e-8)
        u_hat = diff / dist_clamped.unsqueeze(-1)

        # 6. Project Wind onto Direction
        # Dot product: (u_x * w_x) + (u_y * w_y)
        wind_proj = (u_hat * wind).sum(dim=-1) # (Batch, N)

        # 7. Apply Exponential Physics Formula
        # Cost = Dist * exp( -alpha * (wind . direction) )
        exponent = -1.0 * alpha.squeeze(-1) * wind_proj
        multiplier = torch.exp(exponent)

        step_costs = dist * multiplier

        # 8. Sum up costs for the full tour
        total_cost = step_costs.sum(dim=1)

        return total_cost, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        # INJECT THE FLAG: We tell the dataset "This is a Windy Problem"
        kwargs['problem_type'] = 'windy_tsp'
        return TSPDataset(*args, **kwargs)


class TSPSL(TSP):
    """Class representing the Travelling Salesman Problem, trained with Supervised Learning
    """
    NAME = 'tspsl'


class TSPDataset(Dataset):
    
    def __init__(self, filename=None, min_size=20, max_size=50, batch_size=128,
                 num_samples=128000, offset=0, distribution=None, neighbors=20, 
                 knn_strat=None, supervised=False, nar=False, 
                 node_feature_type='coords',
                 problem_type='tsp',
                 node_embedding_type='original',
                 legacy_mode=False):
        """Class representing a PyTorch dataset of TSP instances, which is fed to a dataloader
            Supports both Standard TSP (.txt) and Windy TSD (.pkl)
        """
        super(TSPDataset, self).__init__()

        self.filename = filename
        self.min_size = min_size
        self.max_size = max_size
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.offset = offset
        self.distribution = distribution
        self.neighbors = neighbors
        self.knn_strat = knn_strat
        self.supervised = supervised
        self.nar = nar
        self.node_feature_type = node_feature_type
        self.problem_type = problem_type 
                
        # New flags for Windy TSP
        self.is_windy = False
        self.wind_data = None 
        self.node_embedding_type = node_embedding_type
        self.legacy_mode = legacy_mode #features without normalization


        if filename is not None:
            # === Windy TSP Loading (.pkl) ===
            if filename.endswith('.pkl'):
                self.is_windy = True
                print(f'\nLoading Windy TSP from {filename} with mode {node_feature_type}...')
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    self.wind_data = data[offset:offset+num_samples]
            
            # === Standard TSP Loading (.txt) ===
            else:
                self.nodes_coords = []
                self.tour_nodes = []

                print('\nLoading from {}...'.format(filename))
                for line in tqdm(open(filename, "r").readlines()[offset:offset+num_samples], ascii=True):
                    line = line.split(" ")
                    num_nodes = int(line.index('output')//2)
                    self.nodes_coords.append(
                        [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
                    )

                    if self.supervised:
                        tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
                        self.tour_nodes.append(tour_nodes)

        # === Random Generation ===
        else:
            # Check problem_type OR feature type to decide generation mode
            if self.problem_type == 'windy_tsp' or node_feature_type in ['learned', 'hybrid', 'blank']:
                self.is_windy = True
                print(f'\nGenerating {num_samples} samples of Windy TSP{min_size}-{max_size}...')
                self.wind_data = []
                for _ in tqdm(range(num_samples), ascii=True):
                    num_nodes = np.random.randint(low=min_size, high=max_size+1)
                    
                    instance = generate_windy_instance(num_nodes, alpha=3.0, max_wind=1.0)
                    self.wind_data.append(instance)
            else:
                # Standard TSP Generation
                self.nodes_coords = []
                print('\nGenerating {} samples of TSP{}-{}...'.format(num_samples, min_size, max_size))
                for _ in tqdm(range(num_samples//batch_size), ascii=True):
                    num_nodes = np.random.randint(low=min_size, high=max_size+1)
                    self.nodes_coords += list(np.random.random([batch_size, num_nodes, 2]))
        
        if self.is_windy:
            self.size = len(self.wind_data)
        else:
            self.size = len(self.nodes_coords)
            
        assert self.size % batch_size == 0, \
            "Number of samples ({}) must be divisible by batch size ({})".format(self.size, batch_size)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.is_windy:
            # === Windy TSP Packing ===
            data = self.wind_data[idx]
            loc = data['loc']     # (N, 2)
            wind = data['wind']   # (2,)
            alpha = data['alpha'] # Scalar
            
            num_nodes = len(loc)
            
            # 1. Physics Packing (For Environment/Reward downstream)
            wind_repeated = np.tile(wind, (num_nodes, 1))
            alpha_repeated = np.full((num_nodes, 1), alpha)
            
            # 2. Base Cost Calculation
            diff = loc[None, :, :] - loc[:, None, :] # (N, N, 2)
            dists = np.linalg.norm(diff, axis=-1)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                u = diff / dists[:, :, None]
            u[np.isnan(u)] = 0
            
            wind_proj = np.dot(u, wind) # (N, N)
            costs = dists * np.exp(-1.0 * alpha * wind_proj)
            np.fill_diagonal(costs, 0)
            
            # --- THE FEATURE/REWARD SPLIT ---
            # 2A. The RL Reward (Pure, unscaled exponential physics)
            norm_costs = costs.copy() 

            # --- THE FEATURE/REWARD SPLIT ---
            # 2A. The RL Reward (Pure, unscaled exponential physics)
            norm_costs = costs.copy() 

            if self.legacy_mode:
                # ==========================================
                # LEGACY MODE (For evaluating old baselines)
                # ==========================================
                costs_masked = costs.copy()
                np.fill_diagonal(costs_masked, np.nan) # Ignore self-loops
                
                # Extract Global Stats (Using RAW, UNNORMALIZED costs)
                stat_out_mean = np.nanmean(costs_masked, axis=1, keepdims=True)
                stat_out_std  = np.nanstd(costs_masked, axis=1, keepdims=True)
                stat_out_min  = np.nanmin(costs_masked, axis=1, keepdims=True)
                stat_out_max  = np.nanmax(costs_masked, axis=1, keepdims=True)

                stat_in_mean  = np.nanmean(costs_masked, axis=0, keepdims=True).T
                stat_in_std   = np.nanstd(costs_masked, axis=0, keepdims=True).T
                stat_in_min   = np.nanmin(costs_masked, axis=0, keepdims=True).T
                stat_in_max   = np.nanmax(costs_masked, axis=0, keepdims=True).T

                if self.neighbors is not None:
                    graph_bytes = get_wind_knn_graph(loc, self.neighbors, self.knn_strat, costs)
                else:
                    graph_bytes = np.zeros((num_nodes, num_nodes))

                # Old baseline features (exactly 13 dimensions, no sampled_costs attached)
                nodes_feature = np.concatenate([
                    loc, wind_repeated, alpha_repeated, 
                    stat_out_mean, stat_in_mean, stat_out_std, stat_in_std, 
                    stat_out_min, stat_in_min, stat_out_max, stat_in_max
                ], axis=-1)
                
            else:
                # ==========================================
                # NEW MODE (For ANE and Upgraded Baselines) (Safe Log-Compression + Raw Stats)
                # ==========================================
                # # 1. FIX: Calculate Global Stats using RAW costs!
                # # This restores the massive spikes the MLP and Critic need to learn effectively.
                # raw_costs_masked = costs.copy()
                # np.fill_diagonal(raw_costs_masked, np.nan) # Ignore self-loops for stats

                # stat_out_mean = np.nanmean(raw_costs_masked, axis=1, keepdims=True)
                # stat_out_std  = np.nanstd(raw_costs_masked, axis=1, keepdims=True)
                # stat_out_min  = np.nanmin(raw_costs_masked, axis=1, keepdims=True)
                # stat_out_max  = np.nanmax(raw_costs_masked, axis=1, keepdims=True)

                # stat_in_mean  = np.nanmean(raw_costs_masked, axis=0, keepdims=True).T
                # stat_in_std   = np.nanstd(raw_costs_masked, axis=0, keepdims=True).T
                # stat_in_min   = np.nanmin(raw_costs_masked, axis=0, keepdims=True).T
                # stat_in_max   = np.nanmax(raw_costs_masked, axis=0, keepdims=True).T

                # # 2. FIX: Safe Log-Compression for the Attention Encoders (NO Z-SCORE!)
                # # We take the log to prevent FP16 explosions, but we DO NOT subtract 
                # # the instance mean so the RL Critic doesn't go blind to global difficulty!
                # costs_log = np.log(costs + 1e-8)
                # costs_feature_scaled = costs_log.copy()
                # np.fill_diagonal(costs_feature_scaled, np.nan)


                # 1. Safe Log-Compression (NO Z-SCORE!)
                # We take the log to prevent FP16 explosions in the GAT attention, 
                # but we DO NOT subtract the instance mean so the RL Critic doesn't go blind!
                costs_log = np.log(costs + 1e-8)
                costs_feature_scaled = costs_log.copy()
                np.fill_diagonal(costs_feature_scaled, np.nan)

                # 2. Calculate Global Stats using safely compressed costs!
                stat_out_mean = np.nanmean(costs_feature_scaled, axis=1, keepdims=True)
                stat_out_std  = np.nanstd(costs_feature_scaled, axis=1, keepdims=True)
                stat_out_min  = np.nanmin(costs_feature_scaled, axis=1, keepdims=True)
                stat_out_max  = np.nanmax(costs_feature_scaled, axis=1, keepdims=True)

                stat_in_mean  = np.nanmean(costs_feature_scaled, axis=0, keepdims=True).T
                stat_in_std   = np.nanstd(costs_feature_scaled, axis=0, keepdims=True).T
                stat_in_min   = np.nanmin(costs_feature_scaled, axis=0, keepdims=True).T
                stat_in_max   = np.nanmax(costs_feature_scaled, axis=0, keepdims=True).T

                # 4. GENERATE GRAPH FIRST (Using raw costs for physical accuracy)
                if self.neighbors is not None:
                    graph_bytes = get_wind_knn_graph(loc, self.neighbors, self.knn_strat, costs)
                else:
                    graph_bytes = np.zeros((num_nodes, num_nodes)) 

                # 5. ALWAYS EXTRACT SAMPLED COSTS (Unconditional Fat Vector Generation)
                if isinstance(self.neighbors, float) and self.neighbors < 1.0:
                    k_val = int(num_nodes * self.neighbors)
                else:
                    k_val = int(self.neighbors) if self.neighbors is not None else 20
                    
                sampled_costs = np.zeros((num_nodes, k_val))
                
                mask = (graph_bytes == 0) 
                max_scaled_cost = np.nanmax(costs_feature_scaled) 
                
                for i in range(num_nodes):
                    valid_costs = costs_feature_scaled[i][mask[i]]
                    sorted_costs = np.sort(valid_costs)
                    
                    if len(sorted_costs) >= k_val:
                        sampled_costs[i] = sorted_costs[:k_val]
                    else:
                        sampled_costs[i] = np.pad(sorted_costs, (0, k_val - len(sorted_costs)), constant_values=max_scaled_cost)

                # 6. RETRO-COMPATIBLE SUPER-PACKING
                baseline_features = np.concatenate([
                    loc, wind_repeated, alpha_repeated, 
                    stat_out_mean, stat_in_mean, stat_out_std, stat_in_std, 
                    stat_out_min, stat_in_min, stat_out_max, stat_in_max
                ], axis=-1)

                # ALWAYS append sampled_costs so the dimension is always perfectly 13 + k
                nodes_feature = np.concatenate([baseline_features, sampled_costs], axis=-1)

            return {
                'nodes': torch.FloatTensor(nodes_feature),
                'graph': torch.ByteTensor(graph_bytes),
                'cost_matrix': torch.FloatTensor(norm_costs) # Unscaled for RL Reward
            }











