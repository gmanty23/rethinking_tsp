"""
nets/critic_network.py

Critic Network for Actor-Critic Reinforcement Learning baselines.

BASE IMPLEMENTATION ONLY:
- This file is preserved from the original "Rethinking Generalization" framework.
- It is used to estimate the baseline cost of a graph for REINFORCE.

ON ASYMMETRY / WINDY TSP:
- WARNING: This module has a hardcoded 2D input projection (nn.Linear(2, ...)) 
  and does NOT support the 4D spatial features or Asymmetric Node Embeddings (ANE) 
  used in the Windy TSP extensions. 
- For asymmetric routing, it is highly recommended to use the standard rollout 
  baseline (--baseline rollout) instead of the critic baseline.
"""

from torch import nn


class CriticNetwork(nn.Module):
    def __init__(self, 
                 embedding_dim,
                 encoder_class,
                 n_encode_layers,
                 aggregation="sum",
                 normalization="layer",
                 learn_norm=True,
                 track_norm=False,
                 gated=True,
                 n_heads=8):
        """Critic model for enabling RL training with critic baseline

        References:
            - I. Bello, H. Pham, Q. V. Le, M. Norouzi, and S. Bengio. Neural combinatorial optimization with reinforcement learning. In International Conference on Learning Representations, 2017.
            - M. Deudon, P. Cournut, A. Lacoste, Y. Adulyasak, and L.-M. Rousseau. Learning heuristics for the tsp by policy gradient. In International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pages 170–181. Springer, 2018.
        """

        super(CriticNetwork, self).__init__()

        self.encoder_class = encoder_class
        
        self.init_embed = nn.Linear(2, embedding_dim, bias=True)
        self.encoder = self.encoder_class(n_layers=n_encode_layers, 
                                          n_heads=n_heads,
                                          hidden_dim=embedding_dim, 
                                          aggregation=aggregation, 
                                          norm=normalization, 
                                          learn_norm=learn_norm,
                                          track_norm=track_norm,
                                          gated=gated)

        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, inputs, graph):
        graph_embeddings = self.encoder(self.init_embed(inputs), graph).mean(1)
        return self.value_head(graph_embeddings)
