"""XLM-RoBERTa text encoder with partial fine-tuning."""

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel


class XLMRobertaEncoder(nn.Module):
    """XLM-RoBERTa encoder returning [CLS] token embeddings.

    Freezes the first 10 of 12 encoder layers; layers 10-11 and the
    embedding layer are kept frozen. Only the last 2 layers + pooler
    are updated during training (via a separate low-LR param group).

    Args:
        model_name: HuggingFace model identifier.
        frozen_layers: Number of encoder layers to freeze (default 10).
    """

    EMBED_DIM = 768   # XLM-RoBERTa-base hidden size

    def __init__(self, model_name: str = "xlm-roberta-base", frozen_layers: int = 10):
        super().__init__()
        self.xlmr = AutoModel.from_pretrained(model_name)

        # Freeze embeddings
        for p in self.xlmr.embeddings.parameters():
            p.requires_grad = False

        # Freeze first `frozen_layers` transformer layers
        for i, layer in enumerate(self.xlmr.encoder.layer):
            if i < frozen_layers:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Encode captions and return [CLS] token embeddings.

        Args:
            input_ids:      [B, seq_len] int tensor.
            attention_mask: [B, seq_len] int tensor.

        Returns:
            [B, 768] float tensor.
        """
        out = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]   # [CLS]

    def unfrozen_parameters(self):
        """Yield parameters that require gradients (for low-LR param group)."""
        return (p for p in self.parameters() if p.requires_grad)
