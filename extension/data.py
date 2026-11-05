"""Datasets and DataLoaders for the extended ViC-MAE continued training.

Two batch types are produced:

* **Video pairs** (``VideoPairDataset``) — yields ``frame_i``, ``frame_j``,
  ``delta_t`` (normalized continuous time gap ``|t_j - t_i|``) and
  ``is_video=True``.
* **Static images** (``ImageBatchDataset``) — yields ``image`` and
  ``is_video=False``.

Video pairs can be supplied either as a manifest (JSONL or CSV with
``frame_i``, ``frame_j`` and ``delta_t`` columns) or as a directory of
per-video frame folders (``root/<video_id>/<frame>.jpg``), from which pairs
are sampled at random with ``delta_t = |i - j| / fps`` normalized by a
configurable time constant.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------
def _joint_transform(img_i: Image.Image, img_j: Image.Image,
                     img_size: int = 224, scale: Tuple[float, float] = (0.2, 1.0),
                     flip: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared crop + flip, then tensor conversion and ImageNet normalization."""
    params = _rcrop_params(img_i, img_size, scale)
    img_i = TF.resized_crop(img_i, *params, [img_size, img_size], InterpolationMode.BICUBIC)
    img_j = TF.resized_crop(img_j, *params, [img_size, img_size], InterpolationMode.BICUBIC)
    if flip and random.random() < 0.5:
        img_i = TF.hflip(img_i)
        img_j = TF.hflip(img_j)
    return _to_normalized_tensor(img_i), _to_normalized_tensor(img_j)


def _rcrop_params(img: Image.Image, img_size: int,
                  scale: Tuple[float, float]) -> Tuple[int, int, int, int]:
    """Sample RandomResizedCrop parameters (top, left, height, width)."""
    area = img.size[0] * img.size[1]
    log_ratio = (0.75, 1.3333333333333333)
    for _ in range(10):
        target_area = area * random.uniform(*scale)
        aspect_ratio = torch.exp(torch.empty(1).uniform_(
            torch.log(torch.tensor(log_ratio[0])), torch.log(torch.tensor(log_ratio[1]))
        )).item()
        w = int(round((target_area * aspect_ratio) ** 0.5))
        h = int(round((target_area / aspect_ratio) ** 0.5))
        if 0 < w <= img.size[0] and 0 < h <= img.size[1]:
            top = random.randint(0, img.size[1] - h)
            left = random.randint(0, img.size[0] - w)
            return top, left, h, w
    # Fallback: center crop of the smaller side.
    in_ratio = img.size[0] / img.size[1]
    if in_ratio < log_ratio[0]:
        w, h = img.size[0], int(round(img.size[0] / log_ratio[0]))
    elif in_ratio > log_ratio[1]:
        h, w = img.size[1], int(round(img.size[1] * log_ratio[1]))
    else:
        w, h = img.size[0], img.size[1]
    top = (img.size[1] - h) // 2
    left = (img.size[0] - w) // 2
    return top, left, h, w


def _to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    t = TF.to_tensor(img)
    return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)


def _single_transform(img: Image.Image, img_size: int = 224,
                      scale: Tuple[float, float] = (0.2, 1.0)) -> torch.Tensor:
    """Single-view augmentation for image batches."""
    img = TF.resized_crop(img, *_rcrop_params(img, img_size, scale),
                          [img_size, img_size], InterpolationMode.BICUBIC)
    if random.random() < 0.5:
        img = TF.hflip(img)
    return _to_normalized_tensor(img)


def _eval_transform(img: Image.Image, img_size: int = 224) -> torch.Tensor:
    """Deterministic resize + center crop for evaluation."""
    img = TF.resize(img, int(img_size / 0.875), InterpolationMode.BICUBIC)
    img = TF.center_crop(img, [img_size, img_size])
    return _to_normalized_tensor(img)


# ----------------------------------------------------------------------
# Video pair dataset
# ----------------------------------------------------------------------
class VideoPairDataset(Dataset):
    """Pairs of frames from the same video with a continuous time gap.

    Two input modes:

    * **Manifest** — JSONL (or CSV) where each row has ``frame_i``,
      ``frame_j`` and ``delta_t`` (seconds, already normalized if desired).
      ``delta_t`` may alternatively be derived from ``t_i`` / ``t_j`` columns.
    * **Directory** — ``root/<video_id>/<frame_idx>.jpg``; two frame indices
      are sampled uniformly per video and ``delta_t = |i - j| / fps`` is
      normalized by ``time_norm`` seconds and clamped to ``[0, 1]``.
    """

    def __init__(
        self,
        manifest: Optional[str] = None,
        root: Optional[str] = None,
        fps: float = 30.0,
        time_norm: float = 10.0,
        img_size: int = 224,
        training: bool = True,
    ) -> None:
        if manifest is None and root is None:
            raise ValueError("VideoPairDataset needs a manifest path or a frame root")
        self.img_size = img_size
        self.training = training
        self.pairs: List[Tuple[str, str, float]] = []

        if manifest is not None:
            self.pairs = self._load_manifest(manifest)
        else:
            self._scan_root(root, fps, time_norm)

        if len(self.pairs) == 0:
            raise RuntimeError("VideoPairDataset found no frame pairs")

    # -- manifest -----------------------------------------------------
    @staticmethod
    def _load_manifest(manifest: str) -> List[Tuple[str, str, float]]:
        pairs: List[Tuple[str, str, float]] = []
        path = Path(manifest)
        if path.suffix.lower() == ".csv":
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    pairs.append((
                        row["frame_i"],
                        row["frame_j"],
                        float(row.get("delta_t", 0.0)),
                    ))
        else:  # JSONL
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if "delta_t" in rec:
                        delta = float(rec["delta_t"])
                    else:
                        delta = abs(float(rec["t_j"]) - float(rec["t_i"]))
                    pairs.append((rec["frame_i"], rec["frame_j"], delta))
        return pairs

    # -- directory scan -----------------------------------------------
    def _scan_root(self, root: str, fps: float, time_norm: float) -> None:
        root_path = Path(root)
        for video_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
            frames = sorted(
                f for f in video_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS
            )
            if len(frames) < 2:
                continue
            for fi in frames:
                for fj in frames:
                    if fi == fj:
                        continue
                    delta = abs(int(fi.stem) - int(fj.stem)) / fps
                    delta = min(delta / time_norm, 1.0)
                    self.pairs.append((str(fi), str(fj), delta))

    # -- dataset protocol ----------------------------------------------
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path_i, path_j, delta_t = self.pairs[idx]
        with Image.open(path_i) as im:
            img_i = im.convert("RGB")
        with Image.open(path_j) as im:
            img_j = im.convert("RGB")

        if self.training:
            t_i, t_j = _joint_transform(img_i, img_j, self.img_size)
        else:
            t_i = _eval_transform(img_i, self.img_size)
            t_j = _eval_transform(img_j, self.img_size)
        return {"frame_i": t_i, "frame_j": t_j, "delta_t": torch.tensor(delta_t)}


# ----------------------------------------------------------------------
# Static image dataset
# ----------------------------------------------------------------------
class ImageBatchDataset(Dataset):
    """Image-folder dataset yielding ``image`` tensors (no labels needed).

    Wraps a directory of images (recursively searched) so the same transform
    pipeline as the video stream can be reused for the preservation stage.
    """

    def __init__(self, root: str, img_size: int = 224, training: bool = True) -> None:
        self.root = Path(root)
        self.img_size = img_size
        self.training = training
        self.files = sorted(
            f for f in self.root.rglob("*") if f.suffix.lower() in IMG_EXTENSIONS
        )
        if len(self.files) == 0:
            raise RuntimeError(f"no images found under {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with Image.open(self.files[idx]) as im:
            img = im.convert("RGB")
        if self.training:
            t = _single_transform(img, self.img_size)
        else:
            t = _eval_transform(img, self.img_size)
        return {"image": t}


# ----------------------------------------------------------------------
# Collate functions (add the is_video flag and stack)
# ----------------------------------------------------------------------
def video_collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    """Stack video-pair samples into a batch dict with ``is_video=True``."""
    return {
        "frame_i": torch.stack([s["frame_i"] for s in samples]),
        "frame_j": torch.stack([s["frame_j"] for s in samples]),
        "delta_t": torch.stack([s["delta_t"] for s in samples]).reshape(-1, 1),
        "is_video": True,
    }


def image_collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    """Stack image samples into a batch dict with ``is_video=False``."""
    return {
        "image": torch.stack([s["image"] for s in samples]),
        "is_video": False,
    }


# ----------------------------------------------------------------------
# DataLoader factory
# ----------------------------------------------------------------------
def build_dataloaders(
    video_manifest: Optional[str] = None,
    video_root: Optional[str] = None,
    image_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    seed: int = 0,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """Build the video-pair and image DataLoaders.

    Returns ``(video_loader, image_loader)``; either may be ``None`` when its
    data source is not provided.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    video_loader = None
    if video_manifest or video_root:
        ds = VideoPairDataset(
            manifest=video_manifest, root=video_root, img_size=img_size, training=True
        )
        video_loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=video_collate_fn,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            generator=g,
        )

    image_loader = None
    if image_path:
        ds = ImageBatchDataset(root=image_path, img_size=img_size, training=True)
        image_loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=image_collate_fn,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            generator=g,
        )

    if video_loader is None and image_loader is None:
        raise ValueError("provide at least one of video_manifest/video_root/image_path")
    return video_loader, image_loader
