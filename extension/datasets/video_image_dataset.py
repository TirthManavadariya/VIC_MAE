"""
datasets/video_image_dataset.py
--------------------------------
Mixed video-frame / static-image dataset for ViC-MAE temporal extension.

Each sample yields a dictionary with a stable schema so that the training
loop can transparently handle both modalities:

  Video sample (is_video=True):
    {
        "frame_i"  : Tensor[C, H, W],   # context frame
        "frame_j"  : Tensor[C, H, W],   # target  frame
        "delta_t"  : Tensor[1],          # normalised |t_j - t_i| in [0, 1]
        "is_video" : True,
    }

  Image sample (is_video=False):
    {
        "image"    : Tensor[C, H, W],
        "is_video" : False,
    }

The two datasets can be merged via MixedDataset, which interleaves
samples from both sources according to a configurable video_ratio.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image


try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    max_fps: Optional[int] = 15,
) -> int:
    """
    Extract frames from a raw video file into output_dir as JPG files.

    Args:
        video_path : Path to video file (.mp4, .avi, etc.).
        output_dir : Output directory for extracted frames.
        max_fps    : Target sampling FPS (None for all frames).

    Returns:
        Number of frames saved.
    """
    if not _HAS_CV2:
        raise ImportError("opencv-python (cv2) is required to extract video frames.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / max_fps))) if max_fps else 1

    frame_idx = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame_idx % step == 0:
            out_file = output_path / f"{saved_count:06d}.jpg"
            cv2.imwrite(str(out_file), frame)
            saved_count += 1
        frame_idx += 1
    cap.release()
    return saved_count


def default_frame_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def default_image_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class VideoFramePairDataset(Dataset):
    """
    Loads pairs of frames from pre-extracted clip directories OR directly
    from raw video files (.mp4, .avi, .mov, etc.).

    Directory layout option 1 (Image clips):
        root/
          clip_001/
            000000.jpg
            000001.jpg
            ...

    Directory layout option 2 (Raw video files):
        root/
          video_001.mp4
          video_002.mp4
          ...
        OR a single video file passed as `root`.

    Args:
        root          : Root directory containing clip sub-folders or video files.
        max_frame_gap : Maximum frame-index separation; delta_t normalised by this.
        transform     : Optional transform applied to each frame.
        img_size      : Spatial size for default transform.
        extensions    : Valid image extensions for clip sub-folders.
        video_extensions: Valid raw video extensions.
    """

    def __init__(
        self,
        root: str,
        max_frame_gap: int = 48,
        transform: Optional[Callable] = None,
        img_size: int = 224,
        extensions: Tuple[str, ...] = IMAGE_EXTENSIONS,
        video_extensions: Tuple[str, ...] = VIDEO_EXTENSIONS,
    ) -> None:
        super().__init__()
        self.root          = Path(root)
        self.max_frame_gap = max_frame_gap
        self.img_size      = img_size
        self.transform     = transform or default_frame_transform(img_size)

        self.clips: List[List[Path]] = []
        self.video_files: List[Path] = []

        if self.root.is_file() and self.root.suffix.lower() in video_extensions:
            self.video_files = [self.root]
        elif self.root.is_dir():
            # Check for direct video files first
            self.video_files = sorted([
                p for p in self.root.rglob("*")
                if p.is_file() and p.suffix.lower() in video_extensions
            ])
            # Check for directories containing image frames
            for clip_dir in sorted(self.root.iterdir()):
                if not clip_dir.is_dir():
                    continue
                frames = sorted([
                    p for p in clip_dir.iterdir()
                    if p.suffix.lower() in extensions
                ])
                if len(frames) >= 2:
                    self.clips.append(frames)

        if not self.clips and not self.video_files:
            raise RuntimeError(
                f"No valid video clips or video files found under {root}. "
                f"Supported: folders of frames ({extensions}) or video files ({video_extensions})."
            )

    def __len__(self) -> int:
        return len(self.video_files) if self.video_files else len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        if self.video_files:
            if not _HAS_CV2:
                raise ImportError("opencv-python (cv2) is required to read raw video files.")
            vpath = self.video_files[idx % len(self.video_files)]
            cap = cv2.VideoCapture(str(vpath))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n >= 2:
                gap = random.randint(1, min(self.max_frame_gap, n - 1))
                i_idx = random.randint(0, n - 1 - gap)
                j_idx = i_idx + gap
                cap.set(cv2.CAP_PROP_POS_FRAMES, i_idx)
                ret_i, f_i = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, j_idx)
                ret_j, f_j = cap.read()
            else:
                ret_i, ret_j, f_i, f_j, gap = False, False, None, None, 1
            cap.release()

            if ret_i and ret_j and f_i is not None and f_j is not None:
                img_i = Image.fromarray(cv2.cvtColor(f_i, cv2.COLOR_BGR2RGB))
                img_j = Image.fromarray(cv2.cvtColor(f_j, cv2.COLOR_BGR2RGB))
            else:
                img_i = Image.new("RGB", (self.img_size, self.img_size), color=(128, 128, 128))
                img_j = Image.new("RGB", (self.img_size, self.img_size), color=(128, 128, 128))

            frame_i = self.transform(img_i)
            frame_j = self.transform(img_j)
            delta_t = torch.tensor([gap / self.max_frame_gap], dtype=torch.float32)
            return {"frame_i": frame_i, "frame_j": frame_j, "delta_t": delta_t, "is_video": True}

        frames = self.clips[idx]
        n      = len(frames)
        gap    = random.randint(1, min(self.max_frame_gap, n - 1))
        i_idx  = random.randint(0, n - 1 - gap)
        j_idx  = i_idx + gap

        frame_i = self.transform(Image.open(frames[i_idx]).convert("RGB"))
        frame_j = self.transform(Image.open(frames[j_idx]).convert("RGB"))
        delta_t = torch.tensor([gap / self.max_frame_gap], dtype=torch.float32)

        return {"frame_i": frame_i, "frame_j": frame_j,
                "delta_t": delta_t, "is_video": True}


class StaticImageDataset(Dataset):
    """
    Flat / class-organised image dataset.

    Args:
        root      : Root directory; images discovered recursively.
        transform : Optional transform applied to each image.
        img_size  : Spatial size for default transform.
        extensions: Valid file extensions.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        img_size: int = 224,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPEG"),
    ) -> None:
        super().__init__()
        self.transform = transform or default_image_transform(img_size)
        self.paths: List[Path] = sorted([
            p for p in Path(root).rglob("*")
            if p.is_file() and p.suffix.lower() in {e.lower() for e in extensions}
        ])
        if not self.paths:
            raise RuntimeError(f"No images found under {root}.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        image = self.transform(Image.open(self.paths[idx]).convert("RGB"))
        return {"image": image, "is_video": False}


class MixedDataset(Dataset):
    """
    Interleaves video-frame pairs and static images.

    Each __getitem__ samples from video_dataset with probability video_ratio
    and from image_dataset otherwise.
    """

    def __init__(
        self,
        video_dataset: VideoFramePairDataset,
        image_dataset: StaticImageDataset,
        video_ratio: float = 0.5,
        total_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.video_ds    = video_dataset
        self.image_ds    = image_dataset
        self.video_ratio = video_ratio
        self._total      = total_size or max(len(video_dataset), len(image_dataset))

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> dict:
        if random.random() < self.video_ratio:
            return self.video_ds[idx % len(self.video_ds)]
        return self.image_ds[idx % len(self.image_ds)]


def mixed_collate_fn(batch: List[dict]) -> dict:
    """Collate for homogeneous batches (all video or all image)."""
    is_video = batch[0]["is_video"]
    if is_video:
        return {
            "frame_i"  : torch.stack([b["frame_i"] for b in batch]),
            "frame_j"  : torch.stack([b["frame_j"] for b in batch]),
            "delta_t"  : torch.stack([b["delta_t"] for b in batch]),
            "is_video" : True,
        }
    return {
        "image"    : torch.stack([b["image"] for b in batch]),
        "is_video" : False,
    }


def build_dataloaders(
    video_root: Optional[str] = None,
    image_root: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    max_frame_gap: int = 48,
    pin_memory: bool = True,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """Build separate DataLoaders for video and image datasets."""
    video_loader = None
    image_loader = None

    if video_root is not None:
        vds = VideoFramePairDataset(root=video_root, max_frame_gap=max_frame_gap, img_size=img_size)
        video_loader = DataLoader(
            vds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=True, collate_fn=mixed_collate_fn,
        )

    if image_root is not None:
        ids = StaticImageDataset(root=image_root, img_size=img_size)
        image_loader = DataLoader(
            ids, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=True, collate_fn=mixed_collate_fn,
        )

    return video_loader, image_loader
