"""
nets/encoders/aafm_encoder.py

UPDATE: The whole file 

This file replaces the standard GAT/Transformer encoder with a novel architecture 
designed specifically for asymmetric routing (Windy TSP). Instead of relying on 
standard dot-product multi-head attention, it uses the explicitly calculated 
Neural Adaptive Bias (NAB) matrix as the primary routing mechanism.
"""

import torch
import torch.nn.functional as F
from torch import nn
import math

# Reuse your existing modular blocks
from nets.encoders.gat_encoder import SkipConnection, Normalization, PositionWiseFeedforward

class AdaptationAttentionFreeModule(nn.Module):
    """
    Implements Equation 10 from the RRNCO paper, rewritten using 
    a 4D Softmax to guarantee FP16 numerical stability while 
    maintaining 100% mathematical equivalence to the original formula.
    """
    def __init__(self, embed_dim):
        super(AdaptationAttentionFreeModule, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, A_bias):
        # x: (Batch, Nodes, Dim)
        # A_bias (The NAB matrix): (Batch, Nodes, Nodes)
        
        # 1. Project Q, K, V
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        
        # 2. Reshape for 4D Feature-Wise Broadcasting
        # We need to compute A_{ij} + K_{jc} for every feature channel 'c'
        A_expanded = A_bias.unsqueeze(-1)  # (Batch, Nodes_i, Nodes_j, 1)
        K_expanded = K.unsqueeze(1)        # (Batch, 1, Nodes_j, Dim)
        
        # 3. Calculate Routing Logits (A + K)
        # Broadcasting automatically yields shape: (Batch, Nodes_i, Nodes_j, Dim)
        routing_logits = A_expanded + K_expanded
        
        # 4. Apply Numerically Stable Softmax
        # The original paper requires calculating: exp(A)*exp(K) / sum(exp(A)*exp(K)).
        # Doing this naively causes massive FP16 overflow (NaNs). 
        # By adding the logits (A + K) and passing them through PyTorch's native 
        # F.softmax over dim=2, we leverage the built-in LogSumExp trick. 
        # This executes the exact same math safely.
        routing_weights = F.softmax(routing_logits, dim=2) 
        
        # 5. Multiply by Values and Aggregate
        V_expanded = V.unsqueeze(1) # (Batch, 1, Nodes_j, Dim)
        
        # Multiply weights by V, and sum across the 'j' neighbors to get Context_i
        # This mathematically executes the ( ... ) * V operation
        aggregated_context = torch.sum(routing_weights * V_expanded, dim=2) # (Batch, Nodes_i, Dim)
        
        # 6. Apply Query Gate and Final Projection
        out = torch.sigmoid(Q) * aggregated_context
        
        return self.W_out(out)


class AAFMLayer(nn.Module):
    def __init__(self, embed_dim, feed_forward_dim, norm='layer', learn_norm=True, track_norm=False):
        super(AAFMLayer, self).__init__()
        
        self.aafm = SkipConnection(AdaptationAttentionFreeModule(embed_dim))
        self.norm1 = Normalization(embed_dim, norm, learn_norm, track_norm)
        
        self.positionwise_ff = SkipConnection(
            PositionWiseFeedforward(embed_dim=embed_dim, feed_forward_dim=feed_forward_dim)
        )
        self.norm2 = Normalization(embed_dim, norm, learn_norm, track_norm)

    def forward(self, h, nab_bias, mask=None):
        # We pass nab_bias into the AAFM module as the primary routing mechanism
        h = h + self.aafm.module(h, A_bias=nab_bias)
        h = self.norm1(h, mask=mask)
        h = self.positionwise_ff(h, mask=mask)
        h = self.norm2(h, mask=mask)
        return h


class AAFMEncoder(nn.Module):
    def __init__(self, n_layers, n_heads, hidden_dim, norm='batch', learn_norm=True, track_norm=False, *args, **kwargs):
        super(AAFMEncoder, self).__init__()
        
        feed_forward_hidden = hidden_dim * 4
        
        # --- ARCHITECTURAL SHIFT ---
        # Notice we ignore 'n_heads' entirely. AAFM does not use multi-head attention.
        # It relies on the dense, global (Batch, Nodes, Nodes) NAB matrix to route 
        # features. This eliminates the multi-head computational bottleneck.
        self.layers = nn.ModuleList([
            AAFMLayer(hidden_dim, feed_forward_hidden, norm, learn_norm, track_norm)
            for _ in range(n_layers)
        ])

    def forward(self, x, graph, cost_matrix=None, nab_bias=None, **kwargs):
        # 1. Ensure we have the NAB matrix A
        assert nab_bias is not None, "AAFMEncoder absolutely requires the NAB matrix to function!"
            
        # 2. Pass through AAFM layers
        for layer in self.layers:
            # Note: AAFM doesn't natively use standard Transformer masking (graph).
            # The masking is implicitly handled by the NAB matrix A predicting low routing values.
            x = layer(x, nab_bias=nab_bias, mask=graph)
            
        return x