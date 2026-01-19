#!/usr/bin/env python3
"""
Message Passing Neural Network (MPNN) Backbone

Implements the molecular encoder from the NEST-DRUG architecture:
- T=6 message passing iterations
- 256-dimensional hidden states
- GRU update function
- Edge-conditioned convolutions
- Graph-level readout via mean + max pooling

References:
- Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops
from typing import Optional, Tuple


class MPNNLayer(MessagePassing):
    """
    Single message passing layer with edge-conditioned convolutions.

    Message function: m_ij = MLP([h_i; h_j; e_ij])
    Aggregation: sum over neighbors
    Update: GRU(h_i, aggregated_messages)
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__(aggr='add')  # Sum aggregation

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        # Message MLP: [h_i; h_j; e_ij] -> message
        # Input: 2*hidden_dim (node features) + edge_dim (edge features)
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GRU for node state update
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for one message passing iteration.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        # Message passing
        aggregated = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # GRU update
        x_new = self.gru(aggregated, x)

        # Residual connection + layer norm
        x_new = self.layer_norm(x_new + x)

        return x_new

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute messages from neighbors.

        Args:
            x_i: Features of target nodes [num_edges, hidden_dim]
            x_j: Features of source nodes [num_edges, hidden_dim]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Messages [num_edges, hidden_dim]
        """
        # Concatenate source, target, and edge features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)


class MPNN(nn.Module):
    """
    Full Message Passing Neural Network for molecular encoding.

    Architecture:
    - Input projection: atom features -> hidden_dim
    - T message passing layers with GRU updates
    - Graph-level readout: concat(mean_pool, max_pool) -> 2*hidden_dim

    Output: 512-dimensional molecular embedding (with hidden_dim=256)
    """

    def __init__(
        self,
        node_input_dim: int = 70,
        edge_input_dim: int = 12,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = 2 * hidden_dim  # mean + max pooling

        # Input projection
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Message passing layers
        # Note: edge_dim is hidden_dim after encoding
        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(
                hidden_dim=hidden_dim,
                edge_dim=hidden_dim,  # After edge encoding
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final projection after pooling
        self.output_projection = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode molecular graphs to fixed-size embeddings.

        Args:
            node_features: Atom features [num_atoms, node_input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_features: Bond features [num_edges, edge_input_dim]
            batch: Batch assignment [num_atoms]

        Returns:
            Molecular embeddings [batch_size, 2*hidden_dim]
        """
        # Encode inputs
        x = self.node_encoder(node_features)
        edge_attr = self.edge_encoder(edge_features)

        # Message passing iterations
        for layer in self.mpnn_layers:
            x = layer(x, edge_index, edge_attr)

        # Graph-level readout: concatenate mean and max pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_graph = torch.cat([x_mean, x_max], dim=-1)

        # Final projection
        h_mol = self.output_projection(x_graph)

        return h_mol

    def get_node_embeddings(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get node-level embeddings (for interpretability).

        Returns:
            Node embeddings [num_atoms, hidden_dim]
        """
        x = self.node_encoder(node_features)
        edge_attr = self.edge_encoder(edge_features)

        for layer in self.mpnn_layers:
            x = layer(x, edge_index, edge_attr)

        return x


class AttentivePooling(nn.Module):
    """
    Optional: Attention-weighted graph pooling for interpretability.

    Computes attention weights over nodes and returns weighted sum.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted pooling.

        Args:
            x: Node features [num_atoms, hidden_dim]
            batch: Batch assignment [num_atoms]

        Returns:
            Pooled features [batch_size, hidden_dim]
            Attention weights [num_atoms]
        """
        # Compute attention scores
        attn_scores = self.attention(x).squeeze(-1)

        # Softmax over nodes within each graph
        attn_weights = torch.zeros_like(attn_scores)
        for i in batch.unique():
            mask = batch == i
            attn_weights[mask] = F.softmax(attn_scores[mask], dim=0)

        # Weighted sum
        x_weighted = x * attn_weights.unsqueeze(-1)
        x_pooled = global_mean_pool(x_weighted, batch) * batch.bincount().float().unsqueeze(-1)

        return x_pooled, attn_weights


if __name__ == '__main__':
    # Test MPNN
    print("Testing MPNN backbone...")

    # Create dummy data (69 atom features, 9 bond features from data_utils)
    batch_size = 4
    num_atoms = 20
    num_edges = 40

    node_features = torch.randn(num_atoms, 69)
    edge_index = torch.randint(0, num_atoms, (2, num_edges))
    edge_features = torch.randn(num_edges, 9)
    batch = torch.tensor([0]*5 + [1]*5 + [2]*5 + [3]*5)

    # Create model
    mpnn = MPNN(
        node_input_dim=69,
        edge_input_dim=9,
        hidden_dim=256,
        num_layers=6,
    )

    # Forward pass
    h_mol = mpnn(node_features, edge_index, edge_features, batch)

    print(f"  Input: {num_atoms} atoms, {num_edges} edges")
    print(f"  Output shape: {h_mol.shape}")
    print(f"  Expected: [{batch_size}, 512]")
    print(f"  Parameters: {sum(p.numel() for p in mpnn.parameters()):,}")
