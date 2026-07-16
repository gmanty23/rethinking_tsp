"""
nets/encoders/edge_gat_encoder.py

Update: The whole file

This novel encoder merges the autoregressive routing architecture of Kool et al. 
with explicit edge-feature integration. Standard Transformers only pass node 
features (X, Y) through the attention layers. This encoder projects the explicit, 
asymmetric Cost Matrix (or NAB Matrix) into high-dimensional space and injects it 
directly into the Keys and Values of every attention head.
"""

import torch
import torch.nn.functional as F
from torch import nn
import math

# Reuse your existing modular blocks from the standard GAT!
from nets.encoders.gat_encoder import SkipConnection, Normalization, PositionWiseFeedforward

class EdgeMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim=None):
        super(EdgeMultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.val_dim = embed_dim // n_heads
        self.key_dim = embed_dim // n_heads

        self.norm_factor = 1 / math.sqrt(self.key_dim)

        # USE NN.LINEAR: Automatically handles the Batch dimension safely!
        self.W_query = nn.Linear(input_dim, n_heads * self.key_dim, bias=False)
        self.W_key = nn.Linear(input_dim, n_heads * self.key_dim, bias=False)
        self.W_val = nn.Linear(input_dim, n_heads * self.val_dim, bias=False)
        self.W_out = nn.Linear(n_heads * self.val_dim, embed_dim, bias=False)

    def forward(self, q, edge_embed, mask=None):
        # q: (Batch, Nodes, Dim)
        # edge_embed: (Batch, Nodes, Nodes, Dim)
        batch_size, n_nodes, _ = q.size()

        # 1. Project and reshape to (Heads, Batch, Nodes, Dim) safely
        Q = self.W_query(q).view(batch_size, n_nodes, self.n_heads, self.key_dim).permute(2, 0, 1, 3)
        K = self.W_key(q).view(batch_size, n_nodes, self.n_heads, self.key_dim).permute(2, 0, 1, 3)
        V = self.W_val(q).view(batch_size, n_nodes, self.n_heads, self.val_dim).permute(2, 0, 1, 3)

        # 2. Reshape Edge Embeddings for Multi-Head processing
        # E: (Batch, Nodes, Nodes, Heads, Key_Dim) -> (Heads, Batch, Nodes, Nodes, Key_Dim)
        E = edge_embed.view(batch_size, n_nodes, n_nodes, self.n_heads, self.key_dim).permute(3, 0, 1, 2, 4)

        # 3. Expand K and V to interact with the Edge matrix
        # K_expanded: (Heads, Batch, 1_QueryNode, All_Keys, Key_Dim)
        K_expanded = K.unsqueeze(2).expand_as(E)
        V_expanded = V.unsqueeze(2).expand_as(E)

        # 4. Inject Edges into Keys and Values (The Kool/Bresson Math)
        # STANDARD ATTENTION: Compatibility = Q @ K^T, Context = sum(attn * V)
        # EDGE ATTENTION: Compatibility = Q @ (K + E)^T, Context = sum(attn * (V + E))
        # RATIONALE: This mathematically forces the query node to consider both the 
        # target node's inherent features (K) AND the directional wind cost to get there (E).
        K_total = K_expanded + E
        V_total = V_expanded + E

        # 5. Calculate Compatibility: Q @ (K + E)^T
        Q_expanded = Q.unsqueeze(3) # (Heads, Batch, Nodes, 1, Key_Dim)
        compatibility = self.norm_factor * torch.matmul(Q_expanded, K_total.transpose(-1, -2)).squeeze(-2)

        # 6. Apply standard mask (with self-loop protection)
        if mask is not None:
            bool_mask = mask.bool().clone()
            idx = torch.arange(bool_mask.size(-1), device=bool_mask.device)
            bool_mask[..., idx, idx] = False 
            compatibility[bool_mask[None, :, :, :].expand_as(compatibility)] = -1e10

        attn = F.softmax(compatibility, dim=-1)

        # 7. Aggregate Values: Context = sum(attn * (V + E))
        attn_expanded = attn.unsqueeze(-1)
        heads = (attn_expanded * V_total).sum(dim=3)

        # 8. Project back out to final embedding dimension
        heads_concat = heads.permute(1, 2, 0, 3).contiguous().view(batch_size, n_nodes, self.n_heads * self.val_dim)
        out = self.W_out(heads_concat)
        
        return out


class EdgeMultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, feed_forward_dim, norm='layer', learn_norm=True, track_norm=False):
        super(EdgeMultiHeadAttentionLayer, self).__init__()
        self.self_attention = EdgeMultiHeadAttention(n_heads, embed_dim, embed_dim)
        self.norm1 = Normalization(embed_dim, norm, learn_norm, track_norm)
        self.positionwise_ff = SkipConnection(PositionWiseFeedforward(embed_dim=embed_dim, feed_forward_dim=feed_forward_dim))
        self.norm2 = Normalization(embed_dim, norm, learn_norm, track_norm)

    def forward(self, h, edge_embed, mask):
        # Custom manual skip-connection due to the extra edge_embed argument
        h = h + self.self_attention(h, edge_embed=edge_embed, mask=mask)
        h = self.norm1(h, mask=mask)
        h = self.positionwise_ff(h, mask=mask)
        h = self.norm2(h, mask=mask)
        return h


class EdgeGATEncoder(nn.Module):
    def __init__(self, n_layers, n_heads, hidden_dim, norm='batch', learn_norm=True, track_norm=False, *args, **kwargs):
        super(EdgeGATEncoder, self).__init__()
        
        # MLPs to project the 1D physical cost into a high-dimensional edge feature
        # FIX: We MUST apply LayerNorm here to prevent the variance of E_ij from 
        # exploding when we add it to the Keys and Values inside the attention head!
        # Use Tanh instead of LayerNorm to preserve scalar magnitudes!
        self.edge_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        feed_forward_hidden = hidden_dim * 4
        
        self.layers = nn.ModuleList([
            EdgeMultiHeadAttentionLayer(n_heads, hidden_dim, feed_forward_hidden, norm, learn_norm, track_norm)
            for _ in range(n_layers)
        ])

    def forward(self, x, graph, cost_matrix=None, nab_bias=None, **kwargs):
        
        # 1. Choose edge initialization (MUST use elif to prevent overwriting!)
        if nab_bias is not None:
            # --- The NAB-EdgeGAT Ablation ---
            # Instead of raw costs, we use the fused Distance/Angle/Cost representations 
            # generated by the Neural Adaptive Bias module.
            # nab_bias is (B, N, N). We add the feature dim to make it (B, N, N, 1)
            edge_embed = self.edge_proj(nab_bias.unsqueeze(-1))
        elif cost_matrix is not None:
            # --- Fallback to Raw Wind Costs ---
            # Safely compress the raw exponential physics using log() to prevent 
            # numerical instability before passing it to the Linear layer.
            safe_costs = torch.log(cost_matrix + 1e-8)
            edge_embed = self.edge_proj(safe_costs.unsqueeze(-1))        
        else:
            raise ValueError("EdgeGAT requires either cost_matrix or nab_bias to function!")
            
        # 2. Pass through Edge-Attention layers
        for layer in self.layers:
            x = layer(x, edge_embed=edge_embed, mask=graph)
            
        return x