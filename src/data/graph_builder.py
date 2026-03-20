"""Build the PyG HeteroData graph from raw dataset files.

Intended to be called once offline via scripts/build_graph.py.
"""

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.stats import chi2_contingency
from torch_geometric.data import HeteroData
from tqdm import tqdm

from src.data.post_parser import parse_post_json, parse_profile, LEAKAGE_HASHTAGS
from src.data.splits import make_splits, load_splits


_MENTION_RE = re.compile(r"@(\w+)")
_HASHTAG_RE = re.compile(r"#(\w+)")

# ------------------------------------------------------------------
# Node feature dimensions (must match config)
# post:        [likes_log1p, comments_log1p, hashtag_count, usertag_count,
#               caption_len_log1p, image_count, day_onehot(7), node_type(3)] = 16
# influencer:  [keyword_freqs(100), followers_log1p, followees_log1p,
#               post_count_log1p, category_onehot(22), node_type(3)]          = 128
# brand:       [followers_log1p, followees_log1p, post_count_log1p,
#               category_onehot(22), node_type(3)]                             = 28
# ------------------------------------------------------------------

POST_FEAT_DIM = 16
INFL_FEAT_DIM = 128   # 100 + 3 + 22 + 3
BRAND_FEAT_DIM = 28   # 3 + 22 + 3


def _log1p(x: float) -> float:
    return math.log1p(max(x, 0))


def _day_onehot(day: int) -> list[float]:
    v = [0.0] * 7
    if 0 <= day < 7:
        v[day] = 1.0
    return v


def _category_onehot(cat: str, vocab: dict[str, int], n: int) -> list[float]:
    v = [0.0] * n
    idx = vocab.get(cat, -1)
    if idx >= 0:
        v[idx] = 1.0
    return v


# ------------------------------------------------------------------
# Chi-square keyword features for influencer nodes
# ------------------------------------------------------------------

def _build_chi2_vocab(
    posts: list[dict],
    parsed_posts: list[dict | None],
    train_indices: list[int],
    n_keywords: int = 100,
) -> list[str]:
    """Build top-n chi-square keywords from training-split captions only."""
    from collections import Counter

    word_re = re.compile(r"\b[a-z]{2,}\b")

    # Count (word, class) occurrences over training posts
    pos_counts: Counter = Counter()
    neg_counts: Counter = Counter()

    for i in train_indices:
        p = parsed_posts[i]
        if p is None:
            continue
        label = posts[i]["class"]
        words = set(word_re.findall(p["caption"].lower()))
        if label == 1:
            pos_counts.update(words)
        else:
            neg_counts.update(words)

    all_words = set(pos_counts) | set(neg_counts)
    n_pos = sum(1 for i in train_indices if posts[i]["class"] == 1)
    n_neg = len(train_indices) - n_pos

    scores: list[tuple[float, str]] = []
    for w in all_words:
        tp = pos_counts.get(w, 0)
        fp = neg_counts.get(w, 0)
        tn = n_neg - fp
        fn = n_pos - tp
        table = [[tp, fn], [fp, tn]]
        try:
            chi2, *_ = chi2_contingency(table, correction=False)
        except Exception:
            chi2 = 0.0
        scores.append((chi2, w))

    scores.sort(reverse=True)
    return [w for _, w in scores[:n_keywords]]


def _influencer_keyword_freqs(
    inf_name: str,
    inf_post_indices: list[int],
    parsed_posts: list[dict | None],
    keyword_vocab: list[str],
) -> np.ndarray:
    """Compute TF-style keyword frequency vector for one influencer."""
    word_re = re.compile(r"\b[a-z]{2,}\b")
    vocab_set = {w: i for i, w in enumerate(keyword_vocab)}
    freq = np.zeros(len(keyword_vocab), dtype=np.float32)

    for i in inf_post_indices:
        p = parsed_posts[i]
        if p is None:
            continue
        words = word_re.findall(p["caption"].lower())
        for w in words:
            if w in vocab_set:
                freq[vocab_set[w]] += 1.0

    # Normalize by number of posts
    if inf_post_indices:
        freq /= max(len(inf_post_indices), 1)
    return freq


# ------------------------------------------------------------------
# Main builder
# ------------------------------------------------------------------

def build_graph(
    post_info_path: str | Path,
    dataset_dir: str | Path,
    split_cache_path: str | Path | None = None,
    n_keywords: int = 100,
    n_categories: int = 22,
    max_workers: int = 8,
    seed: int = 42,
) -> HeteroData:
    """Build and return a PyG HeteroData object from raw files.

    Args:
        post_info_path: Path to post_info.json.
        dataset_dir: Root dataset directory.
        split_cache_path: If given, load splits from this path; otherwise compute them.
        n_keywords: Number of chi-square keywords per influencer.
        n_categories: Number of category one-hot dimensions.
        max_workers: Thread pool size for parallel JSON parsing.
        seed: Random seed for splits.

    Returns:
        HeteroData with node features, edge indices, masks, and labels.
    """
    dataset_dir = Path(dataset_dir)
    pst_dir = dataset_dir / "json_files" / "json"
    inf_dir = dataset_dir / "profiles_influencers" / "users_influencers_SPOD"
    brd_dir = dataset_dir / "profiles_brands" / "users_brands_SPOD"

    print("Loading post_info.json …")
    with open(post_info_path, "r") as f:
        posts: list[dict] = json.load(f)
    n_posts = len(posts)
    print(f"  {n_posts:,} posts loaded")

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------
    if split_cache_path and Path(split_cache_path).exists():
        print(f"Loading splits from {split_cache_path} …")
        from src.data.splits import load_splits
        splits = load_splits(split_cache_path)
    else:
        print("Computing influencer-disjoint splits …")
        splits = make_splits(posts, seed=seed, save_path=split_cache_path)

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    print(f"  train={len(train_set):,} val={len(val_set):,} test={len(test_set):,}")

    # ------------------------------------------------------------------
    # Build node index mappings
    # ------------------------------------------------------------------
    print("Building node index mappings …")
    influencer_names = sorted({p["name"] for p in posts})
    influencer_to_idx = {name: i for i, name in enumerate(influencer_names)}
    n_influencers = len(influencer_names)

    # Brand nodes: unique @mentioned usernames across all posts (populated below)
    brand_name_set: set[str] = set()

    # ------------------------------------------------------------------
    # Parse all post JSONs in parallel
    # ------------------------------------------------------------------
    print(f"Parsing {n_posts:,} post JSONs (threads={max_workers}) …")
    parsed_posts: list[dict | None] = [None] * n_posts

    def _parse_one(i: int):
        post_filename = posts[i]["post"]
        return i, parse_post_json(pst_dir / post_filename)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_parse_one, i): i for i in range(n_posts)}
        for fut in tqdm(as_completed(futs), total=n_posts, desc="  posts"):
            i, result = fut.result()
            parsed_posts[i] = result

    n_failed = sum(1 for p in parsed_posts if p is None)
    print(f"  {n_failed:,} posts failed to parse (will use zero features)")

    # Collect brand mentions
    for i, p in enumerate(parsed_posts):
        if p is None:
            continue
        # Use raw post JSON brand mentions — but they're from cleaned caption
        # Use usertag names from original caption; we stored usertag_count only.
        # Re-extract @mentions from parsed caption for brand matching:
        for post_entry in [posts[i]]:
            if post_entry.get("brnd"):
                # We'll extract brand names from the post JSON caption directly
                pass

    # Second pass: extract brand @mentions from parsed captions for edge building
    # We need to re-open post JSONs to get raw caption for @mention names.
    # Alternatively, re-read from parsed_posts caption (already leakage-stripped,
    # but @mentions are preserved since we only strip #hashtags).
    print("Extracting brand mentions for edges …")
    post_brand_mentions: list[list[str]] = [[] for _ in range(n_posts)]
    for i, p in enumerate(parsed_posts):
        if p is None:
            continue
        # @mentions are preserved after leakage stripping (we only strip #tags)
        mentions = [m.lower() for m in _MENTION_RE.findall(p["caption"])]
        post_brand_mentions[i] = mentions
        if posts[i].get("brnd"):
            brand_name_set.update(mentions)

    # Filter brand_name_set: only keep names that actually appear in brd_dir
    print("Filtering brand names against profile files …")
    existing_brands: set[str] = set()
    for name in tqdm(brand_name_set, desc="  brands"):
        if (brd_dir / name).exists():
            existing_brands.add(name)
    brand_name_set = existing_brands
    print(f"  {len(brand_name_set):,} brand nodes with profile files")

    brand_names = sorted(brand_name_set)
    brand_to_idx = {name: i for i, name in enumerate(brand_names)}
    n_brands = len(brand_names)

    # ------------------------------------------------------------------
    # Category vocabulary (influencer + brand combined)
    # ------------------------------------------------------------------
    print("Building category vocabulary …")
    category_counts: dict[str, int] = {}

    for name in tqdm(influencer_names, desc="  influencer profiles"):
        prof = parse_profile(inf_dir / name)
        if prof and prof["category"]:
            cat = prof["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

    for name in tqdm(brand_names, desc="  brand profiles"):
        prof = parse_profile(brd_dir / name)
        if prof and prof["category"]:
            cat = prof["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

    # Take top-(n_categories - 1) categories; reserve last slot for "other"
    top_cats = sorted(category_counts, key=lambda c: -category_counts[c])[: n_categories - 1]
    category_vocab = {cat: i for i, cat in enumerate(top_cats)}
    print(f"  Category vocab: {list(category_vocab)[:5]} …")

    # ------------------------------------------------------------------
    # Chi-square keyword vocabulary (fit on train only)
    # ------------------------------------------------------------------
    print("Computing chi-square keyword vocabulary (train only) …")
    keyword_vocab = _build_chi2_vocab(
        posts, parsed_posts, splits["train"], n_keywords=n_keywords
    )
    print(f"  Top keywords: {keyword_vocab[:10]} …")

    # Influencer → list of post indices (for keyword freq computation)
    inf_post_indices: dict[str, list[int]] = {name: [] for name in influencer_names}
    for i, post in enumerate(posts):
        inf_post_indices[post["name"]].append(i)

    # ------------------------------------------------------------------
    # Build influencer node features
    # ------------------------------------------------------------------
    print(f"Building influencer node features ({n_influencers:,}) …")
    infl_feats = np.zeros((n_influencers, INFL_FEAT_DIM), dtype=np.float32)
    node_type_infl = [0.0, 1.0, 0.0]   # one-hot for "influencer"

    for i, name in enumerate(tqdm(influencer_names, desc="  influencers")):
        prof = parse_profile(inf_dir / name)
        followers = _log1p(prof["followers"]) if prof else 0.0
        followees = _log1p(prof["followees"]) if prof else 0.0
        post_count = _log1p(prof["post_count"]) if prof else 0.0
        cat = prof["category"] if prof else ""
        cat_vec = _category_onehot(cat, category_vocab, n_categories)

        kw_freq = _influencer_keyword_freqs(
            name, inf_post_indices[name], parsed_posts, keyword_vocab
        )

        feat = list(kw_freq) + [followers, followees, post_count] + cat_vec + node_type_infl
        infl_feats[i] = feat

    # ------------------------------------------------------------------
    # Build brand node features
    # ------------------------------------------------------------------
    print(f"Building brand node features ({n_brands:,}) …")
    brand_feats = np.zeros((n_brands, BRAND_FEAT_DIM), dtype=np.float32)
    node_type_brand = [0.0, 0.0, 1.0]   # one-hot for "brand"

    for i, name in enumerate(tqdm(brand_names, desc="  brands")):
        prof = parse_profile(brd_dir / name)
        followers = _log1p(prof["followers"]) if prof else 0.0
        followees = _log1p(prof["followees"]) if prof else 0.0
        post_count = _log1p(prof["post_count"]) if prof else 0.0
        cat = prof["category"] if prof else ""
        cat_vec = _category_onehot(cat, category_vocab, n_categories)

        feat = [followers, followees, post_count] + cat_vec + node_type_brand
        brand_feats[i] = feat

    # ------------------------------------------------------------------
    # Build post node features
    # ------------------------------------------------------------------
    print(f"Building post node features ({n_posts:,}) …")
    post_feats = np.zeros((n_posts, POST_FEAT_DIM), dtype=np.float32)
    post_labels = np.zeros(n_posts, dtype=np.float32)
    node_type_post = [1.0, 0.0, 0.0]   # one-hot for "post"

    for i, post in enumerate(tqdm(posts, desc="  posts")):
        p = parsed_posts[i]
        likes = _log1p(p["likes"]) if p else 0.0
        comments = _log1p(p["comments"]) if p else 0.0
        hashtag_count = float(p["hashtag_count"]) if p else 0.0
        usertag_count = float(p["usertag_count"]) if p else 0.0
        cap_len = _log1p(p["caption_length"]) if p else 0.0
        image_count = float(len(post.get("imgs", [])))
        day = p["posting_day"] if p else 0
        day_vec = _day_onehot(day)

        feat = [likes, comments, hashtag_count, usertag_count, cap_len, image_count] + day_vec + node_type_post
        post_feats[i] = feat
        post_labels[i] = float(post["class"])

    # ------------------------------------------------------------------
    # Build edge indices
    # ------------------------------------------------------------------
    print("Building edge indices …")

    # post → influencer (authorship)
    post_to_inf_src: list[int] = []
    post_to_inf_dst: list[int] = []
    for i, post in enumerate(posts):
        inf_idx = influencer_to_idx.get(post["name"])
        if inf_idx is not None:
            post_to_inf_src.append(i)
            post_to_inf_dst.append(inf_idx)

    # post → brand (@mention)
    post_to_brd_src: list[int] = []
    post_to_brd_dst: list[int] = []
    for i, mentions in enumerate(tqdm(post_brand_mentions, desc="  brand edges")):
        for m in mentions:
            brd_idx = brand_to_idx.get(m)
            if brd_idx is not None:
                post_to_brd_src.append(i)
                post_to_brd_dst.append(brd_idx)

    print(f"  post→influencer edges: {len(post_to_inf_src):,}")
    print(f"  post→brand edges:      {len(post_to_brd_src):,}")

    # ------------------------------------------------------------------
    # Assemble HeteroData
    # ------------------------------------------------------------------
    print("Assembling HeteroData …")
    data = HeteroData()

    data["post"].x = torch.from_numpy(post_feats)
    data["post"].y = torch.from_numpy(post_labels)
    data["post"].num_nodes = n_posts

    data["influencer"].x = torch.from_numpy(infl_feats)
    data["influencer"].num_nodes = n_influencers

    data["brand"].x = torch.from_numpy(brand_feats)
    data["brand"].num_nodes = n_brands

    # Forward edges
    data["post", "published_by", "influencer"].edge_index = torch.tensor(
        [post_to_inf_src, post_to_inf_dst], dtype=torch.long
    )
    data["post", "mentions", "brand"].edge_index = torch.tensor(
        [post_to_brd_src, post_to_brd_dst], dtype=torch.long
    )

    # Reverse edges (required for HGT message passing)
    data["influencer", "rev_published_by", "post"].edge_index = torch.tensor(
        [post_to_inf_dst, post_to_inf_src], dtype=torch.long
    )
    data["brand", "rev_mentions", "post"].edge_index = torch.tensor(
        [post_to_brd_dst, post_to_brd_src], dtype=torch.long
    )

    # Train/val/test masks on post nodes
    train_mask = torch.zeros(n_posts, dtype=torch.bool)
    val_mask = torch.zeros(n_posts, dtype=torch.bool)
    test_mask = torch.zeros(n_posts, dtype=torch.bool)
    for idx in splits["train"]:
        train_mask[idx] = True
    for idx in splits["val"]:
        val_mask[idx] = True
    for idx in splits["test"]:
        test_mask[idx] = True

    data["post"].train_mask = train_mask
    data["post"].val_mask = val_mask
    data["post"].test_mask = test_mask

    print(f"\nGraph summary:")
    print(f"  post nodes:       {n_posts:,}")
    print(f"  influencer nodes: {n_influencers:,}")
    print(f"  brand nodes:      {n_brands:,}")
    total_edges = (
        len(post_to_inf_src) + len(post_to_brd_src) +
        len(post_to_inf_src) + len(post_to_brd_src)  # + reverses
    )
    print(f"  total edges:      {total_edges:,}")

    return data
