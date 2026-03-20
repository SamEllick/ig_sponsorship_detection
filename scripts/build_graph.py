#!/usr/bin/env python
"""One-time offline script to build and save the HeteroData graph.

Usage:
    python scripts/build_graph.py [--config overrides]

Saves to data/graph.pt (~8-12 GB RAM peak, ~30-60 min runtime).
"""

import argparse
import sys
from pathlib import Path
import os
# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from dotenv import load_dotenv

load_dotenv()
ds_path = os.environ.get('DATASET_FP')

from src.config import Config
from src.data.graph_builder import build_graph


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Build HeteroData graph from raw dataset.")
    parser.add_argument("--dataset-dir", default=None, help="Override DATASET_FP env var")
    parser.add_argument("--graph-cache-path", default=None)
    parser.add_argument("--split-cache-path", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config()
    if args.dataset_dir:
        cfg.dataset_dir = args.dataset_dir
    if args.graph_cache_path:
        cfg.graph_cache_path = args.graph_cache_path
    if args.split_cache_path:
        cfg.split_cache_path = args.split_cache_path

    return cfg, args.max_workers


def main():
    cfg, max_workers = parse_args()

    if not cfg.dataset_dir:
        print("ERROR: DATASET_FP env var not set and --dataset-dir not provided.")
        sys.exit(1)

    post_info_path = Path(f"{ds_path}/post_info.json")
    if not post_info_path.exists():
        print(f"ERROR: post_info.json not found at {post_info_path}")
        sys.exit(1)

    data = build_graph(
        post_info_path=post_info_path,
        dataset_dir=cfg.dataset_dir,
        split_cache_path=cfg.split_cache_path,
        n_keywords=cfg.num_keyword_features,
        n_categories=cfg.num_categories,
        max_workers=max_workers,
        seed=cfg.seed,
    )

    # Pre-tokenize all captions and embed them in the graph so
    # train.py can read input_ids/attention_mask from each batch.
    print("\nPre-tokenizing captions with XLM-R …")
    from src.data.text_dataset import pre_tokenize_all
    pst_dir = Path(cfg.dataset_dir) / "json_files" / "json"
    input_ids, attention_mask = pre_tokenize_all(
        post_info_path=post_info_path,
        pst_dir=pst_dir,
        model_name=cfg.xlmr_model,
        max_length=cfg.max_token_len,
        num_workers=max_workers,
    )
    data["post"].input_ids = input_ids           # [N, max_token_len] int32
    data["post"].attention_mask = attention_mask  # [N, max_token_len] int32

    out_path = Path(cfg.graph_cache_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving graph to {out_path} …")
    torch.save(data, out_path)
    print("Done.")

    # Quick sanity check
    data2 = torch.load(out_path, weights_only=False)
    print("\nVerification:")
    print(f"  post nodes:       {data2['post'].num_nodes:,}")
    print(f"  influencer nodes: {data2['influencer'].num_nodes:,}")
    print(f"  brand nodes:      {data2['brand'].num_nodes:,}")
    print(f"  train posts:      {data2['post'].train_mask.sum().item():,}")
    print(f"  val posts:        {data2['post'].val_mask.sum().item():,}")
    print(f"  test posts:       {data2['post'].test_mask.sum().item():,}")


if __name__ == "__main__":
    main()
