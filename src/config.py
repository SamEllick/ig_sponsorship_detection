from dataclasses import dataclass, field
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    dataset_dir: str = field(default_factory=lambda: os.getenv("DATASET_FP", ""))
    graph_cache_path: str = "data/graph.pt"
    split_cache_path: str = "data/splits.json"
    embed_cache_dir: str = "data/embeddings/"

    # Graph / HGT
    num_neighbors: list = field(default_factory=lambda: [15, 10])
    hgt_heads: int = 4
    hgt_layers: int = 2
    hidden_dim: int = 256

    # Text encoder
    xlmr_model: str = "xlm-roberta-base"
    max_token_len: int = 128

    # Image encoder
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"

    # Training
    batch_size: int = 512
    lr: float = 1e-3
    lr_encoder: float = 1e-5   # XLM-R unfrozen layers
    epochs: int = 30
    dropout: float = 0.5
    weight_decay: float = 1e-4

    # Loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Feature dims
    num_keyword_features: int = 100
    num_categories: int = 22   # influencer + brand categories combined

    # Dataloader
    num_workers: int = 4

    seed: int = 42
