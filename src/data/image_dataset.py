"""Image dataset for loading and CLIP-preprocessing post images."""

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

MAX_IMGS = 5


class PostImageDataset(Dataset):
    """Dataset that loads up to MAX_IMGS images per post, CLIP-preprocessed.

    __getitem__(post_idx) -> (images: [MAX_IMGS, 3, 224, 224], mask: [MAX_IMGS] bool)

    Images are zero-padded; mask is True for real images, False for padding.
    Posts with no images return all-zero tensors and all-False masks.
    """

    def __init__(
        self,
        post_info_path: str | Path,
        img_dir: str | Path,
        clip_transform=None,
        max_imgs: int = MAX_IMGS,
    ):
        with open(post_info_path, "r") as f:
            self.posts: list[dict] = json.load(f)

        self.img_dir = Path(img_dir)
        self.max_imgs = max_imgs

        if clip_transform is None:
            import open_clip
            _, _, clip_transform = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
        self.transform = clip_transform

    def __len__(self) -> int:
        return len(self.posts)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        post = self.posts[idx]
        img_names: list[str] = post.get("imgs", []) or []
        img_names = img_names[: self.max_imgs]

        images = []
        for name in img_names:
            path = self.img_dir / name
            try:
                img = Image.open(path).convert("RGB")
                images.append(self.transform(img))
            except (OSError, Exception):
                # Corrupt or missing image — skip
                pass

        n_real = len(images)
        # Zero-pad to max_imgs
        while len(images) < self.max_imgs:
            images.append(torch.zeros(3, 224, 224))

        images_tensor = torch.stack(images, dim=0)          # [max_imgs, 3, 224, 224]
        mask = torch.zeros(self.max_imgs, dtype=torch.bool)
        mask[:n_real] = True

        return images_tensor, mask

    def collate_indexed(self, indices: list[int]) -> tuple[Tensor, Tensor]:
        """Fetch a batch by index list (used in training loop)."""
        imgs_list, masks_list = zip(*[self[i] for i in indices])
        return torch.stack(imgs_list), torch.stack(masks_list)
