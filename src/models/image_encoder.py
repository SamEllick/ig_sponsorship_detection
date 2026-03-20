"""CLIP ViT-B/32 image encoder with max-pooling over per-post images."""

from pathlib import Path
from typing import Optional

import open_clip
import torch
import torch.nn as nn
from torch import Tensor


class CLIPImageEncoder(nn.Module):
    """Frozen CLIP visual encoder with max-pool aggregation over images.

    All CLIP parameters are frozen. Output is a max-pooled embedding
    over the per-post images (up to max_imgs).

    If precomputed embeddings are provided, forward() does a table lookup
    instead of running the visual encoder — saving significant GPU time.

    Args:
        model_name:    open_clip model name (e.g. 'ViT-B-32').
        pretrained:    open_clip pretrained weights tag (e.g. 'openai').
        embed_cache:   Optional path to a pre-computed [N, 512] tensor file.
    """

    EMBED_DIM = 512   # ViT-B/32 output dim

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        embed_cache: Optional[str | Path] = None,
    ):
        super().__init__()

        self._precomputed: Optional[Tensor] = None

        if embed_cache and Path(embed_cache).exists():
            self._precomputed = torch.load(embed_cache, weights_only=True)
            print(f"CLIPImageEncoder: loaded precomputed embeddings {self._precomputed.shape}")
            # No need to load the model itself
            self.model = None
        else:
            model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self.model = model
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, images: Tensor, mask: Tensor, post_indices: Optional[Tensor] = None) -> Tensor:
        """Encode images and return max-pooled CLIP embeddings.

        Args:
            images:       [B, max_imgs, 3, 224, 224] float tensor.
            mask:         [B, max_imgs] bool tensor — True = real image.
            post_indices: [B] int tensor of global post indices.
                          Required when using precomputed embeddings.

        Returns:
            [B, 512] float tensor.
        """
        if self._precomputed is not None:
            assert post_indices is not None, "post_indices required for precomputed lookup"
            return self._precomputed[post_indices.cpu()].to(post_indices.device)

        # Online encoding via CLIP
        B, M, C, H, W = images.shape
        images_flat = images.view(B * M, C, H, W)

        with torch.no_grad():
            embeds_flat = self.model.encode_image(images_flat)   # [B*M, 512]

        embeds = embeds_flat.view(B, M, -1)   # [B, M, 512]

        # Masked max-pool
        mask_exp = mask.unsqueeze(-1).float()   # [B, M, 1]
        neg_inf = torch.full_like(embeds, float("-inf"))
        embeds = torch.where(mask.unsqueeze(-1), embeds, neg_inf)
        pooled = embeds.max(dim=1).values       # [B, 512]
        # Posts with no real images → all -inf → replace with zeros
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        return pooled
