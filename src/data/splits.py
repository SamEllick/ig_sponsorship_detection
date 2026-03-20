"""Influencer-disjoint train/val/test split logic."""

import json
import random
from pathlib import Path
from typing import Any


def make_splits(
    posts: list[dict],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    save_path: str | Path | None = None,
) -> dict[str, list[int]]:
    """Create influencer-disjoint train/val/test splits.

    Args:
        posts: List of post dicts (from post_info.json). Each must have 'name' key.
        seed: Random seed for reproducibility.
        ratios: (train, val, test) proportions — must sum to 1.
        save_path: If given, save the splits JSON to this path.

    Returns:
        {"train": [post_idx, ...], "val": [...], "test": [...]}
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"

    # 1. Unique influencer names, sorted for determinism before shuffling
    influencers = sorted({p["name"] for p in posts})

    # 2. Shuffle with seed
    rng = random.Random(seed)
    rng.shuffle(influencers)

    # 3. Assign influencers to splits
    n = len(influencers)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_inf = set(influencers[:n_train])
    val_inf = set(influencers[n_train:n_train + n_val])
    # remainder goes to test

    # 4. Map each post index to its split
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for idx, post in enumerate(posts):
        name = post["name"]
        if name in train_inf:
            splits["train"].append(idx)
        elif name in val_inf:
            splits["val"].append(idx)
        else:
            splits["test"].append(idx)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(splits, f)

    return splits


def load_splits(path: str | Path) -> dict[str, list[int]]:
    with open(path, "r") as f:
        return json.load(f)
