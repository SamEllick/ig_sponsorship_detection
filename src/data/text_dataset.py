"""Caption tokenization dataset for XLM-RoBERTa.

Used by build_graph.py to pre-tokenize captions and store them
directly on post nodes in the HeteroData graph (avoiding re-tokenization
on every batch during training).
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.data.post_parser import parse_post_json


class CaptionTokenDataset(Dataset):
    """Dataset that tokenizes post captions with XLM-RoBERTa.

    __getitem__(idx) -> {"input_ids": Tensor[seq_len], "attention_mask": Tensor[seq_len]}

    Designed to be used with pre_tokenize_all() to build tensors for
    offline storage in HeteroData.
    """

    def __init__(
        self,
        post_info_path: str | Path,
        pst_dir: str | Path,
        model_name: str = "xlm-roberta-base",
        max_length: int = 128,
    ):
        with open(post_info_path, "r") as f:
            self.posts: list[dict] = json.load(f)
        self.pst_dir = Path(pst_dir)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self) -> int:
        return len(self.posts)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        post = self.posts[idx]
        p = parse_post_json(self.pst_dir / post["post"])
        caption = p["caption"] if p else ""

        enc = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),       # [seq_len]
            "attention_mask": enc["attention_mask"].squeeze(0),   # [seq_len]
        }


def pre_tokenize_all(
    post_info_path: str | Path,
    pst_dir: str | Path,
    model_name: str = "xlm-roberta-base",
    max_length: int = 128,
    batch_size: int = 512,
    num_workers: int = 4,
) -> tuple[Tensor, Tensor]:
    """Pre-tokenize all captions and return (input_ids, attention_mask).

    Returns:
        input_ids:      [N, max_length] int32 tensor.
        attention_mask: [N, max_length] int32 tensor.
    """
    from torch.utils.data import DataLoader

    dataset = CaptionTokenDataset(post_info_path, pst_dir, model_name, max_length)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=lambda x: {
            "input_ids": torch.stack([d["input_ids"] for d in x]),
            "attention_mask": torch.stack([d["attention_mask"] for d in x]),
        }
    )

    all_ids, all_masks = [], []
    from tqdm import tqdm
    for batch in tqdm(loader, desc="Tokenizing captions"):
        all_ids.append(batch["input_ids"].to(torch.int32))
        all_masks.append(batch["attention_mask"].to(torch.int32))

    return torch.cat(all_ids, dim=0), torch.cat(all_masks, dim=0)
