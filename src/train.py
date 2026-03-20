"""Training entry point.

Usage:
    python -m src.train [--epochs 30] [--batch-size 512] ...
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.config import Config
from src.data.graph_dataset import make_neighbor_loader
from src.data.image_dataset import PostImageDataset
from src.losses import CombinedLoss
from src.metrics import compute_all_metrics
from src.models.spod import SPoD


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Train SPoD model")
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--lr-encoder", type=float, default=cfg.lr_encoder)
    parser.add_argument("--hidden-dim", type=int, default=cfg.hidden_dim)
    parser.add_argument("--dropout", type=float, default=cfg.dropout)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--graph-cache", default=cfg.graph_cache_path)
    parser.add_argument("--embed-cache-dir", default=cfg.embed_cache_dir)
    parser.add_argument("--checkpoint-dir", default="data/checkpoints/")
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    args = parser.parse_args()

    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.lr_encoder = args.lr_encoder
    cfg.hidden_dim = args.hidden_dim
    cfg.dropout = args.dropout
    cfg.seed = args.seed
    cfg.graph_cache_path = args.graph_cache
    cfg.embed_cache_dir = args.embed_cache_dir
    cfg.num_workers = args.num_workers

    return cfg, args.checkpoint_dir


def evaluate(
    model: SPoD,
    loader,
    image_dataset: PostImageDataset,
    criterion: CombinedLoss,
    device: torch.device,
    clip_cached: bool,
) -> dict:
    model.eval()
    all_scores, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_size = batch["post"].batch_size
            post_ids = batch["post"].n_id[:batch_size]

            if clip_cached:
                images = torch.zeros(batch_size, 1, 3, 224, 224, device=device)
                image_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
                post_indices = post_ids
            else:
                imgs, masks = image_dataset.collate_indexed(post_ids.tolist())
                images = imgs.to(device)
                image_mask = masks.to(device)
                post_indices = None

            logits = model(batch, images, image_mask, post_indices)
            labels = batch["post"].y[:batch_size].to(device)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    metrics = compute_all_metrics(scores, labels)
    metrics["loss"] = total_loss / max(n_batches, 1)
    metrics["auroc"] = float(roc_auc_score(labels, scores)) if labels.sum() > 0 else 0.0

    return metrics


def train_epoch(
    model: SPoD,
    loader,
    image_dataset: PostImageDataset,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedLoss,
    device: torch.device,
    clip_cached: bool,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="  train", leave=False):
        batch = batch.to(device)
        batch_size = batch["post"].batch_size
        post_ids = batch["post"].n_id[:batch_size]

        if clip_cached:
            images = torch.zeros(batch_size, 1, 3, 224, 224, device=device)
            image_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            post_indices = post_ids
        else:
            imgs, masks = image_dataset.collate_indexed(post_ids.tolist())
            images = imgs.to(device)
            image_mask = masks.to(device)
            post_indices = None

        optimizer.zero_grad()
        logits = model(batch, images, image_mask, post_indices)
        labels = batch["post"].y[:batch_size].to(device)

        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    cfg, checkpoint_dir = parse_args()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load graph
    graph_path = Path(cfg.graph_cache_path)
    if not graph_path.exists():
        print(f"ERROR: graph not found at {graph_path}. Run scripts/build_graph.py first.")
        return

    print(f"Loading graph from {graph_path} …")
    data = torch.load(graph_path, weights_only=False)
    print(f"  post nodes: {data['post'].num_nodes:,}")

    # Check for precomputed CLIP embeddings
    clip_cache = Path(cfg.embed_cache_dir) / "clip.pt"
    clip_cached = clip_cache.exists()
    print(f"CLIP embeddings precomputed: {clip_cached}")

    # Image dataset (still needed for online encoding or as placeholder)
    image_dataset = PostImageDataset(
        post_info_path="nbs/post_info.json",
        img_dir=Path(cfg.dataset_dir) / "images",
    )

    # Data loaders
    print("Creating data loaders …")
    train_loader = make_neighbor_loader(data, "train_mask", cfg, shuffle=True)
    val_loader   = make_neighbor_loader(data, "val_mask",   cfg, shuffle=False)
    test_loader  = make_neighbor_loader(data, "test_mask",  cfg, shuffle=False)

    # Model
    print("Building model …")
    model = SPoD(
        config=cfg,
        graph_metadata=data.metadata(),
        clip_embed_cache=str(clip_cache) if clip_cached else None,
    ).to(device)

    # Optimizer — two param groups
    optimizer = torch.optim.AdamW(
        [
            {"params": model.base_parameters(), "lr": cfg.lr},
            {"params": model.encoder_parameters(), "lr": cfg.lr_encoder},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    criterion = CombinedLoss(
        pos_weight=6.22,
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
    ).to(device)

    # Training loop
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auroc = 0.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        train_loss = train_epoch(
            model, train_loader, image_dataset, optimizer, criterion, device, clip_cached
        )
        scheduler.step()

        val_metrics = evaluate(
            model, val_loader, image_dataset, criterion, device, clip_cached
        )
        val_metrics["train_loss"] = train_loss

        row = {"epoch": epoch, **val_metrics}
        history.append(row)
        print(
            f"  train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"auroc={val_metrics['auroc']:.4f}  "
            f"map={val_metrics['map']:.4f}  "
            f"mrr={val_metrics['mrr']:.4f}"
        )

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val_metrics}, ckpt_path)
            print(f"  [✓] Saved best checkpoint (auroc={best_auroc:.4f})")

    # Save history
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final test evaluation
    print("\n=== Test Evaluation ===")
    best_ckpt = torch.load(ckpt_dir / "best.pt", weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    test_metrics = evaluate(
        model, test_loader, image_dataset, criterion, device, clip_cached
    )
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    with open(ckpt_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
