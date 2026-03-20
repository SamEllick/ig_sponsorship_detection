#!/usr/bin/env python
"""Pre-compute and cache CLIP image embeddings and (optionally) XLM-R text embeddings.

Run once before training to avoid re-encoding each epoch.

Usage:
    python scripts/precompute_embeddings.py [--clip] [--xlmr] [--batch-size 128]

Outputs:
    data/embeddings/clip.pt  — FloatTensor [N, 512]
    data/embeddings/xlmr.pt  — FloatTensor [N, 768]  (optional, ~4.9 GB)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

load_dotenv()

from src.config import Config
from src.data.image_dataset import PostImageDataset


def precompute_clip(cfg: Config, batch_size: int, device: torch.device):
    import open_clip

    out_path = Path(cfg.embed_cache_dir) / "clip.pt"
    if out_path.exists():
        print(f"CLIP embeddings already exist at {out_path}. Skipping.")
        return

    print("Loading CLIP model …")
    model, _, transform = open_clip.create_model_and_transforms(
        cfg.clip_model, pretrained=cfg.clip_pretrained
    )
    model = model.to(device).eval()

    dataset = PostImageDataset(
        post_info_path="nbs/post_info.json",
        img_dir=Path(cfg.dataset_dir) / "images",
        clip_transform=transform,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers)

    all_embeds = []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="CLIP"):
            # images: [B, max_imgs, 3, 224, 224]  masks: [B, max_imgs]
            B, M, C, H, W = images.shape
            images_flat = images.view(B * M, C, H, W).to(device)
            embeds_flat = model.encode_image(images_flat)           # [B*M, 512]
            embeds_flat = embeds_flat.view(B, M, -1)                # [B, M, 512]

            # Masked max-pool
            masks_dev = masks.to(device).unsqueeze(-1).float()     # [B, M, 1]
            embeds_flat = embeds_flat * masks_dev                   # zero out padding
            # Replace zeros with -inf for max-pool, then restore
            neg_inf = torch.full_like(embeds_flat, float("-inf"))
            embeds_flat = torch.where(masks_dev.bool(), embeds_flat, neg_inf)
            pooled = embeds_flat.max(dim=1).values                  # [B, 512]
            # Posts with no images → all -inf → replace with zeros
            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
            all_embeds.append(pooled.cpu())

    all_embeds = torch.cat(all_embeds, dim=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_embeds, out_path)
    print(f"Saved CLIP embeddings: {all_embeds.shape} → {out_path}")


def precompute_xlmr(cfg: Config, batch_size: int, device: torch.device):
    from transformers import AutoModel, AutoTokenizer

    out_path = Path(cfg.embed_cache_dir) / "xlmr.pt"
    if out_path.exists():
        print(f"XLM-R embeddings already exist at {out_path}. Skipping.")
        return

    print("Loading XLM-R model …")
    tokenizer = AutoTokenizer.from_pretrained(cfg.xlmr_model)
    model = AutoModel.from_pretrained(cfg.xlmr_model).to(device).eval()

    with open("nbs/post_info.json", "r") as f:
        posts = json.load(f)

    dataset_dir = Path(cfg.dataset_dir)
    pst_dir = dataset_dir / "json_files" / "json"

    from src.data.post_parser import parse_post_json

    print("Reading captions …")
    captions: list[str] = []
    for post in tqdm(posts, desc="  captions"):
        p = parse_post_json(pst_dir / post["post"])
        captions.append(p["caption"] if p else "")

    all_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), batch_size), desc="XLM-R"):
            batch_caps = captions[i: i + batch_size]
            enc = tokenizer(
                batch_caps,
                max_length=cfg.max_token_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            out = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            cls = out.last_hidden_state[:, 0, :].cpu()   # [B, 768]
            all_embeds.append(cls)

    all_embeds = torch.cat(all_embeds, dim=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_embeds, out_path)
    print(f"Saved XLM-R embeddings: {all_embeds.shape} → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", action="store_true", default=True)
    parser.add_argument("--no-clip", dest="clip", action="store_false")
    parser.add_argument("--xlmr", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    if args.clip:
        precompute_clip(cfg, args.batch_size, device)
    if args.xlmr:
        precompute_xlmr(cfg, args.batch_size, device)


if __name__ == "__main__":
    main()
