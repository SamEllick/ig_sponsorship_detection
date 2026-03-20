"""Parse raw post JSON files and profile tab-separated files."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# Hashtags that directly signal sponsorship — remove before training to prevent leakage.
LEAKAGE_HASHTAGS: frozenset[str] = frozenset([
    "ad", "sponsored", "paidad", "paidpartnership", "gifted",
    "spon", "collab", "collaboration", "promo", "promotion",
    "advertisement", "brandedcontent", "partner", "partnership",
])

_HASHTAG_RE = re.compile(r"#(\w+)")
_MENTION_RE = re.compile(r"@(\w+)")


def _strip_leakage(text: str) -> str:
    """Remove leakage hashtags from caption text."""
    def replace_tag(m: re.Match) -> str:
        tag = m.group(1).lower()
        return "" if tag in LEAKAGE_HASHTAGS else m.group(0)
    return _HASHTAG_RE.sub(replace_tag, text).strip()


def parse_post_json(json_path: str | Path) -> Optional[dict]:
    """Parse a single post JSON file.

    Returns a dict with keys:
        caption, likes, comments, usertag_count, hashtag_count,
        caption_length, is_video, taken_at_timestamp, posting_day
    Returns None if the file cannot be parsed.
    """
    try:
        with open(json_path, "r", encoding="utf-8", errors="replace") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Caption — may be absent or nested
    caption_raw = ""
    try:
        edges = raw.get("edge_media_to_caption", {}).get("edges", [])
        if edges:
            caption_raw = edges[0]["node"]["text"] or ""
    except (KeyError, IndexError, TypeError):
        pass

    # Count usertags BEFORE stripping anything (raw caption)
    usertag_count = len(_MENTION_RE.findall(caption_raw))

    # Strip leakage hashtags from caption
    caption = _strip_leakage(caption_raw)

    # Count hashtags AFTER stripping leakage
    hashtag_count = len(_HASHTAG_RE.findall(caption))

    likes = 0
    try:
        likes = int(raw.get("edge_media_preview_like", {}).get("count", 0))
    except (TypeError, ValueError):
        pass

    comments = 0
    try:
        comments = int(raw.get("edge_media_to_comment", {}).get("count", 0))
    except (TypeError, ValueError):
        pass

    is_video = bool(raw.get("is_video", False))

    timestamp = 0
    try:
        timestamp = int(raw.get("taken_at_timestamp", 0))
    except (TypeError, ValueError):
        pass

    posting_day = 0
    if timestamp:
        try:
            posting_day = datetime.utcfromtimestamp(timestamp).weekday()  # 0=Monday
        except (OSError, OverflowError, ValueError):
            pass

    return {
        "caption": caption,
        "likes": likes,
        "comments": comments,
        "usertag_count": usertag_count,
        "hashtag_count": hashtag_count,
        "caption_length": len(caption),
        "is_video": is_video,
        "taken_at_timestamp": timestamp,
        "posting_day": posting_day,
    }


def parse_profile(profile_path: str | Path) -> Optional[dict]:
    """Parse an influencer or brand profile file (tab-separated, 11 fields).

    Field order: name, followers, followees, post_count, url,
                 verified, category, bio, email, phone, avatar_url

    Returns a dict with keys:
        name, followers, followees, post_count, category
    Returns None if the file cannot be parsed.
    """
    try:
        with open(profile_path, "r", encoding="utf-8", errors="replace") as f:
            line = f.readline()
    except OSError:
        return None

    parts = line.rstrip("\n").split("\t")
    # Pad to at least 11 fields
    while len(parts) < 11:
        parts.append("")

    def _int(s: str) -> int:
        try:
            return int(s.strip().replace(",", "").replace(".", ""))
        except ValueError:
            return 0

    return {
        "name": parts[0].strip(),
        "followers": _int(parts[1]),
        "followees": _int(parts[2]),
        "post_count": _int(parts[3]),
        "category": parts[6].strip(),
    }
