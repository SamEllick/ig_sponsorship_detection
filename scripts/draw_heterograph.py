"""
Draw a small illustrative subgraph of the heterogeneous Instagram network.
Saves: scripts/heterograph.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ── palette ───────────────────────────────────────────────────────────────────
BG        = "#0f1117"
C_POST    = "#4a9eff"   # blue
C_INFL    = "#2ecc71"   # green
C_BRAND   = "#e67e22"   # orange
EDGE_AUTH = "#7c5cbf"   # purple  (authored_by)
EDGE_MENT = "#e74c3c"   # red     (mentions)
WHITE     = "#ffffff"
GRAY      = "#8899aa"

# ── build graph ───────────────────────────────────────────────────────────────
G = nx.DiGraph()

posts  = ["Post 1\n(sponsored)", "Post 2\n(sponsored)", "Post 3\n(unknown)", "Post 4\n(unknown)"]
infls  = ["@alice", "@bob"]
brands = ["NikeRun", "GlowBeauty"]

for n in posts:  G.add_node(n, ntype="post")
for n in infls:  G.add_node(n, ntype="infl")
for n in brands: G.add_node(n, ntype="brand")

# authored_by edges  (post → influencer)
G.add_edge("Post 1\n(sponsored)", "@alice",   etype="authored_by")
G.add_edge("Post 2\n(sponsored)", "@alice",   etype="authored_by")
G.add_edge("Post 3\n(unknown)",   "@bob",     etype="authored_by")
G.add_edge("Post 4\n(unknown)",   "@bob",     etype="authored_by")

# mentions edges  (post → brand)
G.add_edge("Post 1\n(sponsored)", "NikeRun",    etype="mentions")
G.add_edge("Post 2\n(sponsored)", "NikeRun",    etype="mentions")
G.add_edge("Post 3\n(unknown)",   "GlowBeauty", etype="mentions")

# ── layout: manual positions for clarity ─────────────────────────────────────
pos = {
    # posts down the middle
    "Post 1\n(sponsored)": (0,  1.5),
    "Post 2\n(sponsored)": (0,  0.5),
    "Post 3\n(unknown)":   (0, -0.5),
    "Post 4\n(unknown)":   (0, -1.5),
    # influencers on the left
    "@alice": (-2.8,  1.0),
    "@bob":   (-2.8, -1.0),
    # brands on the right
    "NikeRun":    (2.8,  1.0),
    "GlowBeauty": (2.8, -0.5),
}

node_color = [
    C_POST if G.nodes[n]["ntype"] == "post"
    else C_INFL if G.nodes[n]["ntype"] == "infl"
    else C_BRAND
    for n in G.nodes()
]

node_shape_map = {n: G.nodes[n]["ntype"] for n in G.nodes()}

auth_edges = [(u, v) for u, v, d in G.edges(data=True) if d["etype"] == "authored_by"]
ment_edges = [(u, v) for u, v, d in G.edges(data=True) if d["etype"] == "mentions"]

# ── draw ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

node_sizes = {
    n: 2800 if G.nodes[n]["ntype"] == "post" else 2200
    for n in G.nodes()
}

# draw nodes by type to get different shapes
for ntype, shape, color in [("post", "s", C_POST), ("infl", "o", C_INFL), ("brand", "D", C_BRAND)]:
    nodelist = [n for n in G.nodes() if G.nodes[n]["ntype"] == ntype]
    nx.draw_networkx_nodes(
        G, pos, nodelist=nodelist,
        node_color=color + "55", edgecolors=color,
        node_size=[node_sizes[n] for n in nodelist],
        node_shape=shape, linewidths=2, ax=ax,
    )

nx.draw_networkx_edges(
    G, pos, edgelist=auth_edges,
    edge_color=EDGE_AUTH, arrows=True,
    arrowstyle="-|>", arrowsize=18,
    width=2, connectionstyle="arc3,rad=0.08", ax=ax,
    min_source_margin=28, min_target_margin=28,
)
nx.draw_networkx_edges(
    G, pos, edgelist=ment_edges,
    edge_color=EDGE_MENT, arrows=True,
    arrowstyle="-|>", arrowsize=18,
    width=2, connectionstyle="arc3,rad=0.08", ax=ax,
    min_source_margin=28, min_target_margin=28,
)

nx.draw_networkx_labels(
    G, pos, font_color=WHITE, font_size=8.5, font_weight="bold", ax=ax,
)

# edge labels (mid-arc, offset manually)
edge_label_pos = {
    ("Post 1\n(sponsored)", "@alice"):      (-1.55,  1.60),
    ("Post 2\n(sponsored)", "@alice"):      (-1.45,  0.55),
    ("Post 3\n(unknown)",   "@bob"):        (-1.45, -0.55),
    ("Post 4\n(unknown)",   "@bob"):        (-1.55, -1.60),
    ("Post 1\n(sponsored)", "NikeRun"):     ( 1.45,  1.60),
    ("Post 2\n(sponsored)", "NikeRun"):     ( 1.45,  0.55),
    ("Post 3\n(unknown)",   "GlowBeauty"):  ( 1.45, -0.35),
}
edge_labels = {
    ("Post 1\n(sponsored)", "@alice"):      "authored_by",
    ("Post 2\n(sponsored)", "@alice"):      "authored_by",
    ("Post 3\n(unknown)",   "@bob"):        "authored_by",
    ("Post 4\n(unknown)",   "@bob"):        "authored_by",
    ("Post 1\n(sponsored)", "NikeRun"):     "mentions",
    ("Post 2\n(sponsored)", "NikeRun"):     "mentions",
    ("Post 3\n(unknown)",   "GlowBeauty"):  "mentions",
}
for edge, lbl_pos in edge_label_pos.items():
    color = EDGE_AUTH if "authored_by" in edge_labels[edge] else EDGE_MENT
    ax.text(*lbl_pos, edge_labels[edge],
            ha="center", va="center", fontsize=7.5,
            color=color, style="italic",
            bbox=dict(facecolor=BG, edgecolor="none", pad=1))

# callout: same brand → temporal regularisation
ax.annotate(
    "Posts 1 & 2: same brand +\nsimilar timestamp → temporal\nregularisation ties their scores",
    xy=(1.4, 0.95), xytext=(1.7, -0.15),
    fontsize=7.5, color=GRAY, ha="left",
    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8, linestyle="dashed"),
    bbox=dict(facecolor="#1a1f2e", edgecolor=GRAY, boxstyle="round,pad=0.4", linewidth=0.8),
)

# callout: post with no brand mention
ax.annotate(
    "Post 4: no brand mention\n→ no 'mentions' edge",
    xy=(-0.55, -1.5), xytext=(-2.0, -2.1),
    fontsize=7.5, color=GRAY, ha="center",
    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8, linestyle="dashed"),
    bbox=dict(facecolor="#1a1f2e", edgecolor=GRAY, boxstyle="round,pad=0.4", linewidth=0.8),
)

# ── legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=C_POST  + "55", edgecolor=C_POST,   label="Post node  (■)"),
    mpatches.Patch(facecolor=C_INFL  + "55", edgecolor=C_INFL,   label="Influencer node  (●)"),
    mpatches.Patch(facecolor=C_BRAND + "55", edgecolor=C_BRAND,  label="Brand node  (◆)"),
    mpatches.Patch(facecolor=EDGE_AUTH,       edgecolor=EDGE_AUTH, label="authored_by edge"),
    mpatches.Patch(facecolor=EDGE_MENT,       edgecolor=EDGE_MENT, label="mentions edge"),
]
leg = ax.legend(
    handles=legend_handles, loc="lower right",
    framealpha=0.25, facecolor="#1a1f2e", edgecolor=GRAY,
    labelcolor=WHITE, fontsize=8.5, title="Node / Edge types",
    title_fontsize=9,
)
leg.get_title().set_color(GRAY)

ax.set_title(
    "Heterogeneous graph: posts · influencers · brands\n"
    "1.6 M posts · 38 K influencers · 27 K brands · 3.9 M edges  (toy example shown)",
    color=WHITE, fontsize=11, pad=12,
)
ax.axis("off")
plt.tight_layout()

out = "scripts/heterograph.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
