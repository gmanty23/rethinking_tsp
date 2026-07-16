"""
nets/encoders/gnn_encoder.py

Directional & Bias-Injected GNN Encoder.

BASE IMPLEMENTATION:
- Standard Gated Graph ConvNet (Bresson et al. 2018).

CONTRIBUTIONS (Windy/Asymmetric TSP Extensions):
- Added `gnn_direction_mode` to allow the GNN to perform asymmetric aggregation 
  ('forward' for incoming edges, 'backward' for outgoing, 'dual' for bidirectional fusion).
- Replaced binary edge embeddings with continuous linear projections to natively 
  handle explicit Cost Matrices and NAB Biases.
- Edge gating incorporates the edge costs and directional biases directly into the aggregation mechanism.
- Implemented `deep_nab` injection to ensure wind biases survive deep into the network.
"""

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

        # For Dual mode, we need a separate weight matrix for Out-Aggregation ---- CONCATENATION AND LINEAR PROJECTION MODE
        if self.gnn_direction_mode == 'dual':
            self.V_out = nn.Linear(hidden_dim, hidden_dim, bias=True)
            # UPDATE: Projection layer to mix concatenated inputs (2 * hidden -> hidden)
            self.project_dual = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)



        self.norm_h = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)

        self.norm_e = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)
        
    def forward(self, h, e, graph, deep_nab = None):
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
        e_tmp = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
        
        # === UPDATE DEEP NAB INJECTION ===
        if deep_nab is not None:
            # Inject the structural wind bias directly into the gating calculation
            e_tmp = e_tmp + deep_nab
            
        e = e_tmp # Reassign to 'e' so your normalization and residual code below works seamlessly
        gates = torch.sigmoid(e)  # B x V x V x H

        # =====UPPDATE: Directional Aggregation Logic =====

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
            # In Asymmetric TSP, the cost from A->B is not the cost from B->A.
            # Here we calculate two separate aggregations:
            # 1. Incoming (costs to reach node i)
            # 2. Outgoing (costs to leave node i)
            # We then fuse them. (NOTE: A simple sum() caused cancellation of wind 
            # costs during experimentation. A learned projection/fusion works best).
            
            # 1. Incoming (Standard V_h over standard graph)
            Vh_in = prepare_Vh(self.V, h)
            aggr_in = self.aggregate(Vh_in, graph, gates)
            
            # 2. Outgoing (Transposed V_out over transposed graph)
            Vh_out = prepare_Vh(self.V_out, h)
            aggr_out = self.aggregate(Vh_out, graph.transpose(1, 2), gates.transpose(1, 2))
            
            # 3. Concatenate and Project 
            # Stacks features side-by-side [Batch, Nodes, 2*Hidden] and linearly 
            # projects back to [Batch, Nodes, Hidden]
            aggr_cat = torch.cat([aggr_in, aggr_out], dim=-1)
            aggr = self.project_dual(aggr_cat)

            # Original Simple Sum Version (no gating)
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
        Vh[graph.unsqueeze(-1).expand_as(Vh).bool()] = 0
        
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
                 learn_norm=True, track_norm=False, gated=True, gnn_direction_mode = 'forward', gnn_deep_bias=False, *args, **kwargs):
        super(GNNEncoder, self).__init__()

        # --- Base Implementation (Legacy Support) ---
        # Used for standard Euclidean TSP where the graph is purely binary (edge exists: 0 or 1).
        self.init_embed_edges = nn.Embedding(2, hidden_dim)
        
        # --- UPDATE: Continuous Edge Projection ---
        # Used for Windy TSP. Projects the continuous scalar values of the Cost Matrix 
        # (or the generated NAB bias) directly into the hidden dimension.
        self.init_lin_edges = nn.Linear(1, hidden_dim)

        # GNN Layer Configurations
        self.gnn_deep_bias = gnn_deep_bias

        self.layers = nn.ModuleList([
            GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated, gnn_direction_mode)
            for _ in range(n_layers)
        ])


    def forward(self, x, graph, cost_matrix=None, nab_bias=None, **kwargs):
        """
        Args:
            x: Input node features (B x V x H)
            graph: Graph adjacency matrices (B x V x V)
        Returns: 
            Updated node features (B x V x H)
        """

        # 1. Choose Layer 0 initialization (MUST use elif to prevent overwriting!)
        if nab_bias is not None:
            # --- The NAB-GNN Ablation ---
            e = self.init_lin_edges(nab_bias.unsqueeze(-1))
        elif cost_matrix is not None: 
            # --- Windy TSP Path ---
            # This prevents the GNN's internal Sigmoid gates from instantly saturating.
            safe_costs = torch.log(cost_matrix + 1e-8)
            e = self.init_lin_edges(safe_costs.unsqueeze(-1))
        else:
            # --- Standard TSP Path (Legacy) ---
            e = self.init_embed_edges(graph.type(torch.long))

        # 2. Project Deep NAB if the ablation flag is turned on
        # UPDATE: Deep NAB Injection
        # Standard GNNs suffer from "oversmoothing", where edge information 
        # degrades in deeper layers. If `gnn_deep_bias` is True, we project the 
        # NAB matrix and explicitly re-inject it into the gating mechanism of 
        # EVERY layer, enforcing structural asymmetry throughout the network.
        deep_nab = None
        if self.gnn_deep_bias and nab_bias is not None:
            deep_nab = self.init_lin_edges(nab_bias.unsqueeze(-1))

        # 3. Pass through GNN Layers
        for layer in self.layers:
            # We must explicitly pass the deep_nab tensor into the layer!
            x, e = layer(x, e, graph, deep_nab=deep_nab)

        return x