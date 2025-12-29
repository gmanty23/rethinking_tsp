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
    # If `neighbors` is a percentage, convert to int
    if knn_strat == 'percentage':
        neighbors = int(num_nodes * neighbors)
    
    if neighbors >= num_nodes-1 or neighbors == -1:
        W = np.zeros((num_nodes, num_nodes))
    else:
        # Compute distance matrix
        W_val = squareform(pdist(nodes, metric='euclidean'))
        W = np.ones((num_nodes, num_nodes))
        
        # Determine k-nearest neighbors for each node
        knns = np.argpartition(W_val, kth=neighbors, axis=-1)[:, neighbors::-1]
        # Make connections
        for idx in range(num_nodes):
            W[idx][knns[idx]] = 0
    
    # Remove self-connections
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
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        """Method to call beam search, given TSP samples and a model
        """

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(nodes, graph)

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
                 problem_type = 'tsp'): # New argument
        """Class representing a PyTorch dataset of TSP instances, which is fed to a dataloader
            Supports both Standard TSP (.txt) and Windy TSD (.pkl)

        Args:
            filename: File path to read from (for SL)
            min_size: Minimum TSP size to generate (for RL)
            max_size: Maximum TSP size to generate (for RL)
            batch_size: Batch size for data loading/batching
            num_samples: Total number of samples in dataset
            offset: Offset for loading from file
            distribution: Data distribution for generation (unused)
            neighbors: Number of neighbors for k-NN graph computation
            knn_strat: Strategy for computing k-NN graphs ('percentage'/'standard')
            supervised: Flag to enable supervised learning
            nar: Flag to indicate Non-autoregressive decoding scheme, which uses edge-level groundtruth
            node_feature_type: 'coords', 'learned', or 'hybrid'. 
                               Controls what features are packed for the model.
            problem_type: 'tsp' or 'windy_tsp'. Indicates the problem variant.

        Notes:
            `batch_size` is important to fix across dataset and dataloader,
            as we are dealing with TSP graphs of variable sizes. To enable
            efficient training without DGL/PyG style sparse graph libraries,
            we ensure that each batch contains dense graphs of the same size.
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
        self.problem_type = problem_type # Store it
        
        # New flags for Windy TSP
        self.is_windy = False
        self.wind_data = None 

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
            
            # 1. Physics Packing (For Environment/Reward)
            wind_repeated = np.tile(wind, (num_nodes, 1))
            alpha_repeated = np.full((num_nodes, 1), alpha)
            
            # 2. Statistics Calculation (For Model Features)
            # Calculate diffs
            diff = loc[None, :, :] - loc[:, None, :] # (N, N, 2)
            dists = np.linalg.norm(diff, axis=-1)
            # Unit vectors
            with np.errstate(divide='ignore', invalid='ignore'):
                u = diff / dists[:, :, None]
            u[np.isnan(u)] = 0
            
            # Wind Projection
            wind_proj = np.dot(u, wind) # (N, N)
            
            # Asymmetric Costs
            costs = dists * np.exp(-1.0 * alpha * wind_proj)
            np.fill_diagonal(costs, 0)

            # norm_costs = costs / (costs.max() + 1e-6)
            norm_costs = costs

            # Extract Stats
            # Mask diagonal with NaN to ignore self-loops in stats
            costs_masked = costs.copy()
            np.fill_diagonal(costs_masked, np.nan)

            # OUTGOING Stats (Axis 1 = Rows)
            stat_out_mean = np.nanmean(costs_masked, axis=1, keepdims=True)
            stat_out_std  = np.nanstd(costs_masked, axis=1, keepdims=True)
            stat_out_min  = np.nanmin(costs_masked, axis=1, keepdims=True)
            stat_out_max  = np.nanmax(costs_masked, axis=1, keepdims=True)

            # INCOMING Stats (Axis 0 = Cols)
            stat_in_mean  = np.nanmean(costs_masked, axis=0, keepdims=True).T
            stat_in_std   = np.nanstd(costs_masked, axis=0, keepdims=True).T
            stat_in_min   = np.nanmin(costs_masked, axis=0, keepdims=True).T
            stat_in_max   = np.nanmax(costs_masked, axis=0, keepdims=True).T

            # 3. Super-Packing
            # Structure: [x, y, w_x, w_y, alpha,  mean_out, mean_in, std_out, std_in, min_out, min_in, max_out, max_in]
            # Indices:    0  1   2    3      4       5         6        7        8        9        10       11       12
            nodes_feature = np.concatenate([
                loc,              # 0-1
                wind_repeated,    # 2-3
                alpha_repeated,   # 4
                stat_out_mean,    # 5
                stat_in_mean,     # 6
                stat_out_std,     # 7
                stat_in_std,      # 8
                stat_out_min,     # 9
                stat_in_min,      # 10
                stat_out_max,     # 11
                stat_in_max       # 12
            ], axis=-1)
            
            return {
                'nodes': torch.FloatTensor(nodes_feature),
                'graph': torch.ByteTensor(nearest_neighbor_graph(loc, self.neighbors, self.knn_strat)),
                'cost_matrix': torch.FloatTensor(norm_costs)
            }
            
        else:
            # === Standard TSP Packing ===
            nodes = self.nodes_coords[idx]
            item = {
                'nodes': torch.FloatTensor(nodes),
                'graph': torch.ByteTensor(nearest_neighbor_graph(nodes, self.neighbors, self.knn_strat))
            }
            if self.supervised:
                tour_nodes = self.tour_nodes[idx]
                item['tour_nodes'] = torch.LongTensor(tour_nodes)
                if self.nar:
                    item['tour_edges'] = torch.LongTensor(tour_nodes_to_W(tour_nodes))

            return item





















