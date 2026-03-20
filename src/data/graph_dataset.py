"""NeighborLoader factory for mini-batch HGT training."""

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from src.config import Config


def make_neighbor_loader(
    data: HeteroData,
    mask_attr: str,
    config: Config,
    shuffle: bool = True,
) -> NeighborLoader:
    """Create a NeighborLoader for mini-batch HGT training.

    Args:
        data: The full HeteroData graph.
        mask_attr: One of 'train_mask', 'val_mask', 'test_mask'.
        config: Config object.
        shuffle: Whether to shuffle the input nodes each epoch.

    Returns:
        NeighborLoader targeting post nodes specified by mask_attr.

    Notes:
        - Reverse edge types have num_neighbors=[0,0] to prevent subgraph explosion.
        - Target nodes are the first batch['post'].batch_size entries in each batch.
    """
    n_layers = len(config.num_neighbors)
    reverse_zero = [0] * n_layers

    num_neighbors = {
        ("post", "published_by", "influencer"): config.num_neighbors,
        ("post", "mentions", "brand"):          config.num_neighbors,
        ("influencer", "rev_published_by", "post"): reverse_zero,
        ("brand", "rev_mentions", "post"):          reverse_zero,
    }

    return NeighborLoader(
        data,
        input_nodes=("post", data["post"][mask_attr]),
        num_neighbors=num_neighbors,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
    )
