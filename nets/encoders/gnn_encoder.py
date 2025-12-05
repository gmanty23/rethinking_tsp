import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GNNLayer(nn.Module):
    """Configurable GNN Layer

    Implements the Gated Graph ConvNet layer:
        h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
        sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
        e_ij = ReLU ( A*h_i + B*h_j + C*e_ij ),
        where Aggr. is an aggregation function: sum/mean/max.

    References:
        - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs. In International Conference on Learning Representations, 2018.
        - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.
    """

    def __init__(self, hidden_dim, aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True, gnn_direction_mode='forward'):
        """
        Args:
            hidden_dim: Hidden dimension size (int)
            aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
            norm: Feature normalization scheme ("layer"/"batch"/None)
            learn_norm: Whether the normalizer has learnable affine parameters (True/False)
            track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
            gated: Whether to use edge gating (True/False)
            gnn_direction_mode: 'forward' (standard, in-only), 'backward' (out-only), or 'dual' (bi-directional)

        """
        super(GNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.norm = norm
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        self.gnn_direction_mode = gnn_direction_mode

        assert self.gated, "Use gating with GCN, pass the `--gated` flag"
        
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # For Dual mode, we need a separate weight matrix for Out-Aggregation
        if self.gnn_direction_mode == 'dual':
            self.V_out = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm_h = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)

        self.norm_e = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)
        
    def forward(self, h, e, graph):
        """
        Args:
            h: Input node features (B x V x H)
            e: Input edge features (B x V x V x H)
            graph: Graph adjacency matrices (B x V x V)
                   0 indicates connection j->i, 1 indicates no connection.
        Returns: 
            Updated node and edge features
        """
        batch_size, num_nodes, hidden_dim = h.shape
        h_in = h
        e_in = e

        # Linear transformations for node update
        Uh = self.U(h)  # B x V x H
        # Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H

        # Linear transformations for edge update and gating
        Ah = self.A(h)  # B x V x H
        Bh = self.B(h)  # B x V x H
        Ce = self.C(e)  # B x V x V x H

        # Update edge features and compute edge gates
        e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
        gates = torch.sigmoid(e)  # B x V x V x H

        # ===== Directional Aggregation Logic =====

        # Helper to prepare Vh for aggregation: (B, V, V, H)
        # Represents feature of 'j' available at 'i'
        def prepare_Vh(linear_layer, h_input):
            return linear_layer(h_input).unsqueeze(1).expand(-1, num_nodes, -1, -1)

        if self.gnn_direction_mode == 'forward':
            # Standard: Aggregate from j to i (Incoming)
            # graph[b,i,j]=0 means edge j->i exists.
            Vh = prepare_Vh(self.V, h)
            aggr = self.aggregate(Vh, graph, gates)
            
        elif self.gnn_direction_mode == 'backward':
            # Out-Only: Aggregate from j where i->j exists
            # We transpose the graph so graph_T[b,i,j] = graph[b,j,i].
            # If graph[b,j,i]=0 (edge i->j exists), then graph_T[b,i,j]=0.
            # Effectively, we sum up features of our Successors.
            Vh = prepare_Vh(self.V, h)
            aggr = self.aggregate(Vh, graph.transpose(1, 2), gates.transpose(1, 2))
            
        elif self.gnn_direction_mode == 'dual':
            # Bi-directional: Sum of Incoming (V) and Outgoing (V_out)
            
            # 1. Incoming (Standard)
            Vh_in = prepare_Vh(self.V, h)
            aggr_in = self.aggregate(Vh_in, graph, gates)
            
            # 2. Outgoing (Transposed)
            # Note: We use the separate weight matrix V_out
            Vh_out = prepare_Vh(self.V_out, h)
            aggr_out = self.aggregate(Vh_out, graph.transpose(1, 2), gates.transpose(1, 2))
            
            aggr = aggr_in + aggr_out
            
        else:
            raise ValueError(f"Unknown gnn_direction_mode: {self.gnn_direction_mode}")


        # Update node features
        h = Uh + aggr  # B x V x H

        # Normalize node features
        h = self.norm_h(
            h.view(batch_size*num_nodes, hidden_dim)
        ).view(batch_size, num_nodes, hidden_dim) if self.norm_h else h
        
        # Normalize edge features
        e = self.norm_e(
            e.view(batch_size*num_nodes*num_nodes, hidden_dim)
        ).view(batch_size, num_nodes, num_nodes, hidden_dim) if self.norm_e else e

        # Apply non-linearity
        h = F.relu(h)
        e = F.relu(e)

        # Make residual connection
        h = h_in + h
        e = e_in + e

        return h, e

    def aggregate(self, Vh, graph, gates):
        """
        Args:
            Vh: Neighborhood features (B x V x V x H)
            graph: Graph adjacency matrices (B x V x V)
            gates: Edge gates (B x V x V x H)
        Returns:
            Aggregated neighborhood features (B x V x H)
        """
        # Perform feature-wise gating mechanism
        Vh = gates * Vh  # B x V x V x H
        
        # Enforce graph structure through masking
        # graph has 1s where there are NO edges, so we zero those out
        Vh[graph.unsqueeze(-1).expand_as(Vh)] = 0
        
        if self.aggregation == "mean":
            return torch.sum(Vh, dim=2) / torch.sum(1-graph, dim=2).unsqueeze(-1).type_as(Vh)
        
        elif self.aggregation == "max":
            return torch.max(Vh, dim=2)[0]
        
        else:
            return torch.sum(Vh, dim=2)
        

class GNNEncoder(nn.Module):
    """Configurable GNN Encoder
    """
    
    def __init__(self, n_layers, hidden_dim, aggregation="sum", norm="layer", 
                 learn_norm=True, track_norm=False, gated=True, gnn_direction_mode = 'forward', *args, **kwargs):
        super(GNNEncoder, self).__init__()

        self.init_embed_edges = nn.Embedding(2, hidden_dim)

        self.layers = nn.ModuleList([
            GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated, gnn_direction_mode)
                for _ in range(n_layers)
        ])

    def forward(self, x, graph):
        """
        Args:
            x: Input node features (B x V x H)
            graph: Graph adjacency matrices (B x V x V)
        Returns: 
            Updated node features (B x V x H)
        """
        # Embed edge features
        e = self.init_embed_edges(graph.type(torch.long))

        for layer in self.layers:
            x, e = layer(x, e, graph)

        return x
