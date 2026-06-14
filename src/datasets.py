"""UCF101 / HMDB51 video datasets with a single shared transform.

Every clip -> (T, C, H, W) float tensor, normalized identically. Batched by the
DataLoader into (B, T, C, H, W), which `src.models` adapts per architecture.
"""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop

import decord
from decord import VideoReader, cpu

from .config import NUM_FRAMES, IMG_SIZE, MEAN, STD

decord.bridge.set_bridge("torch")

VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".mov"}


# ---------------------------------------------------------------- frame sampling
def _uniform_indices(total, n):
    """n frame indices spread uniformly across a clip of `total` frames."""
    if total <= 0:
        return [0] * n
    if total < n:                       # short clip: pad with the last frame
        pad = [total - 1] * (n - total)
        return list(range(total)) + pad
    return [int(round(x)) for x in np.linspace(0, total - 1, n)]


# ---------------------------------------------------------------- transforms
def _transform(clip, train):
    """clip: (T, C, H, W) in [0, 1] -> resized/cropped/normalized."""
    clip = TF.resize(clip, 256, antialias=True)
    if train:
        i, j, h, w = RandomCrop.get_params(clip, (IMG_SIZE, IMG_SIZE))
        clip = TF.crop(clip, i, j, h, w)
        if random.random() < 0.5:
            clip = TF.hflip(clip)
    else:
        clip = TF.center_crop(clip, (IMG_SIZE, IMG_SIZE))
    return TF.normalize(clip, MEAN, STD)


class VideoClipDataset(Dataset):
    def __init__(self, samples, videos_root, train):
        self.samples = samples              # list of (relative_path, label_idx)
        self.videos_root = Path(videos_root)
        self.train = train
        self._warned = False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rel, label = self.samples[i]
        path = str(self.videos_root / rel)
        try:
            vr = VideoReader(path, ctx=cpu(0))
            idx = _uniform_indices(len(vr), NUM_FRAMES)
            frames = vr.get_batch(idx)                  # (T, H, W, C) uint8
            clip = frames.permute(0, 3, 1, 2).float() / 255.0
            return _transform(clip, self.train), label
        except Exception as e:                          # tolerate a few corrupt files
            if not self._warned:
                print(f"[datasets] decode failed for {path}: {e} (using black clip)")
                self._warned = True
            black = torch.zeros(NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE)
            return TF.normalize(black, MEAN, STD), label


# ---------------------------------------------------------------- split parsing
def _find(root, name):
    hits = list(Path(root).rglob(name))
    return hits[0] if hits else None


def _videos_root(root):
    for cand in ("UCF-101", "UCF101", "videos"):
        p = Path(root) / cand
        if p.is_dir():
            return p
    return Path(root)


def _parse_ucf101(root, split, train):
    root = Path(root)
    classind = _find(root, "classInd.txt")
    if classind is None:
        raise FileNotFoundError(
            "classInd.txt not found. Run `python scripts/setup_data.py "
            "--dataset ucf101 --splits-only` or place the official split lists "
            "under data/ucf101/splits/."
        )
    class_to_idx = {}
    for line in classind.read_text().strip().splitlines():
        num, name = line.split()
        class_to_idx[name] = int(num) - 1               # official is 1-based

    listname = f"{'trainlist' if train else 'testlist'}{split:02d}.txt"
    listfile = _find(root, listname)
    if listfile is None:
        raise FileNotFoundError(f"{listname} not found under {root}")

    samples = []
    for line in listfile.read_text().strip().splitlines():
        rel = line.split()[0]                           # 'Class/video.avi'
        cls = rel.split("/")[0]
        samples.append((rel, class_to_idx[cls]))
    return _videos_root(root), samples, len(class_to_idx)


def _parse_folder_split(root, train, seed=42, ratio=0.7):
    """Fallback (used for HMDB51 for now): deterministic per-class 70/30 split.

    NOTE: not the official HMDB51 protocol -- wire the official split files here
    before reporting final HMDB51 numbers.
    """
    vroot = _videos_root(root)
    classes = sorted(d.name for d in vroot.iterdir() if d.is_dir())
    if not classes:
        raise FileNotFoundError(f"no class folders under {vroot}")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rng = random.Random(seed)
    samples = []
    for c in classes:
        vids = sorted(p.name for p in (vroot / c).glob("*") if p.suffix.lower() in VIDEO_EXTS)
        rng.shuffle(vids)
        k = int(len(vids) * ratio)
        for v in (vids[:k] if train else vids[k:]):
            samples.append((f"{c}/{v}", class_to_idx[c]))
    return vroot, samples, len(classes)


def build_dataset(name, data_root, split=1, train=True):
    """Returns (VideoClipDataset, num_classes)."""
    base = Path(data_root) / name
    if not base.exists():
        base = Path(data_root)                          # e.g. Kaggle: data_root IS the dataset

    if name == "ucf101":
        vroot, samples, n = _parse_ucf101(base, split, train)
    else:
        vroot, samples, n = _parse_folder_split(base, train)
    return VideoClipDataset(samples, vroot, train), n
