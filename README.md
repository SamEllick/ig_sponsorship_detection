# ig_sponsorship_detection

Binary classification of Instagram posts as sponsored (1) or not (0), using a heterogeneous graph neural network with multimodal encoders. Based on the SPoD paper (Kim et al., WSDM 2021) with modernised components.

The core differences are:
* using a more modern multilingual language model for creating text embeddings
* using a clip model for image embeddings - the clip model is trained on paired image and text data therefore the embeddings produced may be more correlated with any potential text data from other parts of the graph.
* using a specific heterogenous graph transformer implementation over a regular graph convolutional neural network

## Architecture

Three encoders produce per-post representations which are fused via attention:

- **Graph encoder** — HGT (Heterogeneous Graph Transformer) over a heterogeneous network of post, influencer, and brand nodes (~1.6M posts, ~38K influencers, ~27K brands, ~3.9M edges)
- **Text encoder** — XLM-RoBERTa (layers 0–9 frozen, 10–11 fine-tuned at low LR) on post captions
- **Image encoder** — CLIP ViT-B/32 (fully frozen, precomputed) with max-pool over up to 5 images per post

Text and image embeddings are first combined via bidirectional cross-modal attention, then fused with graph embeddings via learned aspect-attention. A two-layer scoring head produces a sponsorship logit trained with 0.5×BCE + 0.5×FocalLoss.

### Differences from the paper

| | Paper (SPoD) | This repo |
|---|---|---|
| Graph | Vanilla GCN, concatenated layer outputs | HGTConv, residual + LayerNorm |
| Text | BERT | XLM-RoBERTa (multilingual) |
| Image | Inception-V3 (1000-d object probs) | CLIP ViT-B/32 (512-d semantic embeddings) |
| Fusion | 3-way aspect-attention (graph, text, image) | Cross-modal attention (text↔image) → 2-way aspect-attention |
| Loss | ListMLE (learning-to-rank) | BCE + Focal (binary classification) |
| Scale | Full-batch GCN | Mini-batch NeighborLoader |
| Temporal reg. | Manifold regularisation on posting time + brand | Not implemented |

## Dataset

Uses the dataset from the paper: 1,601,074 Instagram posts by 38,113 influencers mentioning 26,910 brands. 221,710 (13.8%) are labelled sponsored; 1,379,364 (86.2%) are unknown. Train/val/test split is 7:1:2 with no influencer overlap across splits.

Expected location (set via `DATASET_FP` in `.env`):
```
/path/to/ig_brands/
├── post_info.txt
├── json_files/json/*.json
├── images/
├── profiles_influencers/users_influencers_SPOD/
└── profiles_brands/users_brands_SPOD/
```


## References

Kim, S., Jiang, J.-Y., & Wang, W. (2021). Discovering Undisclosed Paid Partnership on Social Media via Aspect-Attentive Sponsored Post Learning. *WSDM '21*.
