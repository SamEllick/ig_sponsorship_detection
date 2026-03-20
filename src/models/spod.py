"""Full SPoD model assembly."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.config import Config
from src.data.graph_builder import POST_FEAT_DIM, INFL_FEAT_DIM, BRAND_FEAT_DIM
from src.models.graph_encoder import HGTEncoder
from src.models.text_encoder import XLMRobertaEncoder
from src.models.image_encoder import CLIPImageEncoder
from src.models.fusion import CrossModalAttention, AspectAttention


class SPoD(nn.Module):
    """Sponsored Post Detector with HGT + XLM-R + CLIP + cross-modal + aspect attention.

    Args:
        config: Config object with all hyperparameters.
        graph_metadata: HeteroData.metadata() — (node_types, edge_types).
        clip_embed_cache: Optional path to precomputed CLIP embeddings.
    """

    def __init__(
        self,
        config: Config,
        graph_metadata: tuple,
        clip_embed_cache: Optional[str | Path] = None,
    ):
        super().__init__()

        feat_dims = {
            "post":       POST_FEAT_DIM,
            "influencer": INFL_FEAT_DIM,
            "brand":      BRAND_FEAT_DIM,
        }

        # --- Encoders ---
        self.graph_encoder = HGTEncoder(
            metadata=graph_metadata,
            feat_dims=feat_dims,
            hidden_dim=config.hidden_dim,
            heads=config.hgt_heads,
            num_layers=config.hgt_layers,
            dropout=config.dropout,
        )

        self.text_encoder = XLMRobertaEncoder(model_name=config.xlmr_model)

        self.image_encoder = CLIPImageEncoder(
            model_name=config.clip_model,
            pretrained=config.clip_pretrained,
            embed_cache=clip_embed_cache,
        )

        # --- Projection heads (→ hidden_dim) ---
        self.text_proj = nn.Sequential(
            nn.Linear(XLMRobertaEncoder.EMBED_DIM, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(CLIPImageEncoder.EMBED_DIM, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # --- Fusion ---
        self.cross_modal = CrossModalAttention(
            text_dim=XLMRobertaEncoder.EMBED_DIM,
            image_dim=CLIPImageEncoder.EMBED_DIM,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.aspect_attn = AspectAttention(
            hidden_dim=config.hidden_dim,
            num_aspects=2,   # graph + text-image fused
        )

        # --- Sponsorship scorer (F_h → F_p) ---
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        batch,           # NeighborLoader HeteroData batch
        images: Tensor,  # [B, max_imgs, 3, 224, 224]
        image_mask: Tensor,  # [B, max_imgs] bool
        post_indices: Optional[Tensor] = None,  # global post ids for CLIP lookup
    ) -> Tensor:
        """
        Returns:
            logits: [B] unnormalized sponsorship scores.
        """
        batch_size = batch["post"].batch_size

        # 1. Graph encoder → post embeddings [num_sampled_posts, hidden_dim]
        graph_emb = self.graph_encoder(batch.x_dict, batch.edge_index_dict)
        target_graph = graph_emb[:batch_size]   # [B, hidden_dim]

        # 2. Text encoder → [B, 768]
        input_ids = batch["post"].input_ids[:batch_size]
        attn_mask = batch["post"].attention_mask[:batch_size]
        text_emb = self.text_encoder(input_ids, attn_mask)   # [B, 768]

        # 3. Image encoder → [B, 512]
        image_emb = self.image_encoder(images, image_mask, post_indices)   # [B, 512]

        # 4. Cross-modal attention (text ↔ image) → [B, hidden_dim]
        fused = self.cross_modal(text_emb, image_emb)

        # 5. Aspect attention over [graph, fused] → [B, hidden_dim]
        out = self.aspect_attn([target_graph, fused])

        # 6. Sponsorship score → [B]
        logits = self.scorer(out).squeeze(-1)
        return logits

    def encoder_parameters(self):
        """Unfrozen XLM-R parameters — use separate low-LR optimizer group."""
        return list(self.text_encoder.unfrozen_parameters())

    def base_parameters(self):
        """All parameters except unfrozen XLM-R layers."""
        encoder_ids = {id(p) for p in self.encoder_parameters()}
        return [p for p in self.parameters() if id(p) not in encoder_ids]
