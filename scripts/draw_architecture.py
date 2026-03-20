"""Generate a LinkedIn-ready architecture diagram for the improved SPoD model."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── colour palette ────────────────────────────────────────────────────────────
BG        = "#0e0f13"
C_INPUT   = "#4a9eff"
C_GRAPH   = "#7c5cbf"
C_TEXT    = "#2ecc71"
C_IMAGE   = "#e67e22"
C_CROSS   = "#e74c3c"
C_ASPECT  = "#f39c12"
C_SCORE   = "#1abc9c"
C_LOSS    = "#95a5a6"
WHITE     = "#ffffff"
LIGHTGRAY = "#c8ccd4"

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")

# ── helpers ───────────────────────────────────────────────────────────────────
def box(ax, cx, cy, w, h, color, label, sublabel=None, fontsize=11, subsize=8.5):
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.12", linewidth=1.5,
        edgecolor=color, facecolor=color + "33",
    )
    ax.add_patch(rect)
    yo = 0.10 if sublabel else 0
    ax.text(cx, cy + yo, label, ha="center", va="center",
            color=WHITE, fontsize=fontsize, fontweight="bold")
    if sublabel:
        ax.text(cx, cy - 0.25, sublabel, ha="center", va="center",
                color=color, fontsize=subsize, style="italic")

def arrow(ax, x1, y1, x2, y2, color=LIGHTGRAY, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14))

def label(ax, x, y, text, color=LIGHTGRAY, size=8):
    ax.text(x, y, text, ha="center", va="center", color=color, fontsize=size)

# ── layout constants ──────────────────────────────────────────────────────────
# rows (y): input=9.0, encoders=6.8, cross=4.8, aspect=3.0, score=1.6, loss=0.5
# cols (x): graph=2.5, text=7, image=11.5

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(7, 9.7, "Improved SPoD — Instagram Sponsorship Detection",
        ha="center", va="center", color=WHITE, fontsize=14, fontweight="bold")

# ── input node ────────────────────────────────────────────────────────────────
box(ax, 7, 9.0, 4.2, 0.55, C_INPUT, "Instagram Post",
    sublabel="caption  ·  images  ·  graph neighbourhood", fontsize=12, subsize=8)

# ── encoder row ───────────────────────────────────────────────────────────────
# graph
box(ax, 2.5, 7.1, 3.6, 1.0, C_GRAPH,
    "Graph Encoder", "Heterogeneous Graph Transformer (HGT)", subsize=8.5)
label(ax, 2.5, 6.35, "posts · influencers · brands", C_GRAPH, 7.5)

# text
box(ax, 7, 7.1, 3.6, 1.0, C_TEXT,
    "Text Encoder", "XLM-RoBERTa  (layers 10–11 fine-tuned)", subsize=8.5)
label(ax, 7, 6.35, "caption → 768-dim [CLS]", C_TEXT, 7.5)

# image
box(ax, 11.5, 7.1, 3.6, 1.0, C_IMAGE,
    "Image Encoder", "CLIP ViT-B/32  (frozen)", subsize=8.5)
label(ax, 11.5, 6.35, "≤5 imgs → max-pool → 512-dim", C_IMAGE, 7.5)

# ── input → encoders ─────────────────────────────────────────────────────────
for tx, tc in [(2.5, C_GRAPH), (7, C_TEXT), (11.5, C_IMAGE)]:
    arrow(ax, 7, 8.72, tx, 7.62, color=tc)

# ── cross-modal attention ─────────────────────────────────────────────────────
box(ax, 9.25, 5.15, 4.8, 0.75, C_CROSS,
    "Cross-Modal Attention", "text ↔ image interaction", subsize=9)

arrow(ax, 7,    6.60, 8.10, 5.50, color=C_TEXT)
arrow(ax, 11.5, 6.60, 10.40, 5.50, color=C_IMAGE)
label(ax, 9.25, 5.95, "fused text+image", LIGHTGRAY, 7.5)

# ── aspect attention ──────────────────────────────────────────────────────────
box(ax, 7, 3.45, 5.6, 0.80, C_ASPECT,
    "Aspect Attention", "learned weights over [graph, fused text+image]", subsize=9)

arrow(ax, 2.5,  6.60, 4.65, 3.82, color=C_GRAPH)
arrow(ax, 9.25, 4.77, 8.35, 3.82, color=C_CROSS)

# ── sponsorship scorer ────────────────────────────────────────────────────────
box(ax, 7, 2.15, 3.8, 0.72, C_SCORE,
    "Sponsorship Scorer", "FC → σ → FC → scalar ŷ", subsize=9)

arrow(ax, 7, 3.05, 7, 2.52, color=C_ASPECT)

# ── loss ─────────────────────────────────────────────────────────────────────
box(ax, 7, 0.90, 5.4, 0.72, C_LOSS,
    "Loss", "0.5 × BCE (pos_weight=6.22)  +  0.5 × Focal Loss", subsize=9)

arrow(ax, 7, 1.79, 7, 1.27, color=C_SCORE)

# ── legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (C_GRAPH,  "Graph stream"),
    (C_TEXT,   "Text stream"),
    (C_IMAGE,  "Image stream"),
    (C_CROSS,  "Cross-modal fusion  ← new"),
    (C_ASPECT, "Aspect attention"),
    (C_SCORE,  "Scoring head"),
]
for i, (c, lbl) in enumerate(legend_items):
    bx = 0.35 + (i % 3) * 4.7
    by = 0.28 if i >= 3 else 0.62
    patch = mpatches.Patch(facecolor=c + "44", edgecolor=c, linewidth=1.2)
    ax.legend(handles=[patch], labels=[lbl],
              loc="lower left", bbox_to_anchor=(bx / 14, by / 10),
              fontsize=7.5, framealpha=0, labelcolor=c,
              handlelength=1.2, handleheight=0.9)

plt.tight_layout(pad=0)
out = "scripts/architecture.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
