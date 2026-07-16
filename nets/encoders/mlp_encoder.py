"""
nets/encoders/mlp_encoder.py

Baseline Encoders for NCO.

BASE IMPLEMENTATION:
- Standard Multi-Layer Perceptron (MLP) baseline (Kool et al. 2019).

CONTRIBUTIONS:
- Modified the forward signatures to gracefully accept `cost_matrix` and `nab_bias` 
  to ensure compatibility with the Windy TSP training pipeline.
- Added `IdentityEncoder` for ablation studies testing pure decoder performance.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MLPLayer(nn.Module):
    """Simple MLP layer with ReLU activation
    """

    def __init__(self, hidden_dim, norm="layer", learn_norm=True, track_norm=False):
        """
        Args:
            hidden_dim: Hidden dimension size (int)
            norm: Feature normalization scheme ("layer"/"batch"/None)
            learn_norm: Whether the normalizer has learnable affine parameters (True/False)
            track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
        """
        super(MLPLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.norm = norm
        self.learn_norm = learn_norm

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)

    def forward(self, x):
        batch_size, num_nodes, hidden_dim = x.shape
        x_in = x

        # Linear transformation
        x = self.U(x)

        # Normalize features
        x = self.norm(
            x.view(batch_size*num_nodes, hidden_dim)
        ).view(batch_size, num_nodes, hidden_dim) if self.norm else x

        # Apply non-linearity
        x = F.relu(x)

        # Make residual connection
        x = x_in + x

        return x


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder with ReLU activation, independent of graph structure.
    """
    def __init__(self, n_layers, hidden_dim, norm="layer",
                 learn_norm=True, track_norm=False, *args, **kwargs):
        super(MLPEncoder, self).__init__()
        self.layers = nn.ModuleList(
            MLPLayer(hidden_dim, norm, learn_norm, track_norm) for _ in range(n_layers)
        )

    def forward(self, x, graph=None, cost_matrix=None, **kwargs):
        """
        Args:
            x: Input node features (B x V x H)
            graph: Ignored in MLP
            cost_matrix: Ignored in MLP, accepted for compatibility
            nab_bias: Ignored in MLP, accepted for compatibility
        Returns:
            Updated node features (B x V x H)
        """
        for layer in self.layers:
            x = layer(x)

        return x
    


class IdentityEncoder(nn.Module):
    """
    UPDATE 

    Identity (Dummy) Encoder.
    
    This encoder skips deep feature processing entirely, passing the linearly 
    projected node inputs straight to the Decoder. 
    
    RATIONALE: Used heavily in ablation studies to isolate the performance of 
    the Decoder (and its injected NAB biases) without the compounding effects 
    of deep GNN/Transformer spatial aggregation.
    """
    def __init__(self, *args, **kwargs):
        super(IdentityEncoder, self).__init__()

    def forward(self, x, graph=None, cost_matrix=None, **kwargs):
        # Simply return the input node embeddings (x) without any transformation
        return x

