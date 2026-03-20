"""HGT (Heterogeneous Graph Transformer) encoder."""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import HGTConv


class HGTEncoder(nn.Module):
    """Multi-layer HGT encoder producing post node embeddings.

    Args:
        metadata: HeteroData.metadata() — tuple of (node_types, edge_types).
        feat_dims: Dict mapping node type name → raw feature dimension.
        hidden_dim: Unified hidden dimension after input projection.
        heads: Number of attention heads in HGTConv.
        num_layers: Number of HGT layers.
        dropout: Dropout probability applied between layers.
    """

    def __init__(
        self,
        metadata: tuple,
        feat_dims: dict[str, int],
        hidden_dim: int,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        node_types, edge_types = metadata

        # Per-node-type input projections
        self.input_proj = nn.ModuleDict({
            node_type: nn.Linear(feat_dims[node_type], hidden_dim)
            for node_type in node_types
            if node_type in feat_dims
        })

        # HGT layers
        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=heads,
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple, Tensor],
    ) -> Tensor:
        """Run HGT and return post node embeddings.

        Args:
            x_dict: Node feature tensors by type.
            edge_index_dict: Edge indices by edge type.

        Returns:
            Post node embeddings of shape [num_post_nodes, hidden_dim].
        """
        # Project each node type to hidden_dim
        h_dict: dict[str, Tensor] = {}
        for node_type, proj in self.input_proj.items():
            if node_type in x_dict:
                h_dict[node_type] = proj(x_dict[node_type])

        # Run HGT layers with residual + norm + dropout
        for conv, norm in zip(self.convs, self.norms):
            new_h = conv(h_dict, edge_index_dict)
            for node_type in h_dict:
                if node_type in new_h and new_h[node_type] is not None:
                    h_dict[node_type] = self.dropout(
                        norm(new_h[node_type] + h_dict[node_type])
                    )

        return h_dict["post"]
