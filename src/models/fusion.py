"""Cross-modal attention and aspect-attention fusion modules."""

import torch
import torch.nn as nn
from torch import Tensor


class CrossModalAttention(nn.Module):
    """Bidirectional cross-modal attention between text and image features.

    Text (768-d) attends to image (512-d) and vice versa; outputs are
    concatenated and projected to hidden_dim.

    Args:
        text_dim:   Input dimension of text features (default 768 for XLM-R).
        image_dim:  Input dimension of image features (default 512 for CLIP).
        hidden_dim: Output dimension.
        dropout:    Dropout on attention weights.
    """

    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        attn_dim = hidden_dim

        # Project both modalities to common attention dimension
        self.text_proj = nn.Linear(text_dim, attn_dim)
        self.image_proj = nn.Linear(image_dim, attn_dim)

        # Cross-attention: text queries over image keys/values
        self.text_attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=1, dropout=dropout, batch_first=True
        )
        # Cross-attention: image queries over text keys/values
        self.image_attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=1, dropout=dropout, batch_first=True
        )

        # Fuse: concat both attended features → hidden_dim
        self.fuse_proj = nn.Linear(2 * attn_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, text_feat: Tensor, image_feat: Tensor) -> Tensor:
        """
        Args:
            text_feat:  [B, text_dim]
            image_feat: [B, image_dim]

        Returns:
            [B, hidden_dim] fused cross-modal representation.
        """
        # Project to common dim; unsqueeze seq dim for MHA (seq_len=1)
        t = self.text_proj(text_feat).unsqueeze(1)    # [B, 1, attn_dim]
        v = self.image_proj(image_feat).unsqueeze(1)  # [B, 1, attn_dim]

        # Text attends to image
        t_out, _ = self.text_attn(query=t, key=v, value=v)   # [B, 1, attn_dim]
        # Image attends to text
        v_out, _ = self.image_attn(query=v, key=t, value=t)  # [B, 1, attn_dim]

        t_out = t_out.squeeze(1)   # [B, attn_dim]
        v_out = v_out.squeeze(1)   # [B, attn_dim]

        fused = self.fuse_proj(torch.cat([t_out, v_out], dim=-1))   # [B, hidden_dim]
        return self.norm(fused)


class AspectAttention(nn.Module):
    """Aspect-level attention over K feature vectors.

    Each aspect V_i is projected via tanh(FC(V_i)) to produce r_i.
    A learned context vector r_c computes attention weights:
        α_i = softmax(r_i · r_c)
    Final representation: X = Σ α_i · V_i

    Args:
        hidden_dim: Dimension of each aspect vector.
        num_aspects: K — number of aspects to fuse (default 2: graph, text-image).
    """

    def __init__(self, hidden_dim: int = 256, num_aspects: int = 2):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Parameter(torch.randn(hidden_dim))
        self.num_aspects = num_aspects

    def forward(self, aspects: list[Tensor]) -> Tensor:
        """
        Args:
            aspects: List of K tensors, each [B, hidden_dim].

        Returns:
            [B, hidden_dim] weighted combination.
        """
        # Stack aspects: [B, K, hidden_dim]
        stacked = torch.stack(aspects, dim=1)

        # r_i = tanh(FC(V_i))  →  [B, K, hidden_dim]
        r = torch.tanh(self.proj(stacked))

        # Attention scores: r_i · r_c  → [B, K]
        scores = (r * self.context).sum(dim=-1)   # [B, K]
        alpha = torch.softmax(scores, dim=-1)     # [B, K]

        # Weighted sum: [B, K, 1] * [B, K, hidden_dim] → sum over K → [B, hidden_dim]
        out = (alpha.unsqueeze(-1) * stacked).sum(dim=1)
        return out
