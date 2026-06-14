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


def _is_class_dir_root(p):
    """True if p holds >=2 subdirs and at least one directly contains a video."""
    p = Path(p)
    if not p.is_dir():
        return False
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    if len(subdirs) < 2:
        return False
    for d in subdirs:
        if any(f.is_file() and f.suffix.lower() in VIDEO_EXTS for f in d.iterdir()):
            return True
    return False


def _samples_from_class_root(vroot, class_to_idx):
    samples = []
    for c, idx in class_to_idx.items():
        cdir = Path(vroot) / c
        if not cdir.is_dir():
            continue
        for f in sorted(cdir.iterdir()):
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
                samples.append((f"{c}/{f.name}", idx))
    return samples


def _parse_ucf101_official(root, split, train):
    """Official UCF101 protocol: classInd.txt + trainlist/testlist. None if absent."""
    classind = _find(root, "classInd.txt")
    if classind is None:
        return None
    class_to_idx = {}
    for line in classind.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            class_to_idx[parts[1]] = int(parts[0]) - 1      # official is 1-based
    listname = f"{'trainlist' if train else 'testlist'}{split:02d}.txt"
    listfile = _find(root, listname)
    if listfile is None:
        return None
    samples = []
    for line in listfile.read_text().strip().splitlines():
        rel = line.split()[0]                               # 'Class/video.avi'
        cls = rel.split("/")[0]
        if cls in class_to_idx:
            samples.append((rel, class_to_idx[cls]))
    return _videos_root(root), samples, len(class_to_idx)


def _find_split_dir(root, names):
    """Locate a train/test/val dir (at root or one level down) of class subfolders."""
    root = Path(root)
    bases = [root]
    if root.is_dir():
        bases += [c for c in root.iterdir() if c.is_dir()]
    for base in bases:
        for n in names:
            d = base / n
            if _is_class_dir_root(d):
                return d
    return None


def _parse_presplit_folders(root, train):
    """Dataset pre-split into train/ and test|val/ dirs of class subfolders."""
    train_dir = _find_split_dir(root, ["train", "Train", "training"])
    test_dir = _find_split_dir(root, ["test", "Test", "val", "Val", "validation", "testing"])
    if train_dir is None or test_dir is None:
        return None
    classes = sorted(set(d.name for d in train_dir.iterdir() if d.is_dir()) |
                     set(d.name for d in test_dir.iterdir() if d.is_dir()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    vroot = train_dir if train else test_dir
    return vroot, _samples_from_class_root(vroot, class_to_idx), len(classes)


def _parse_folder_split(root, train, seed=42, ratio=0.7):
    """Single folder of class dirs -> deterministic per-class 70/30 split.

    Used for HMDB51 and as a last-resort fallback. NOTE: not an official
    protocol -- wire official split files before reporting final numbers.
    """
    vroot = _videos_root(root)
    if not _is_class_dir_root(vroot) and Path(vroot).is_dir():
        for child in Path(vroot).iterdir():                 # class dirs one level down?
            if _is_class_dir_root(child):
                vroot = child
                break
    classes = sorted(d.name for d in Path(vroot).iterdir() if d.is_dir())
    if not classes:
        raise FileNotFoundError(f"no class folders under {vroot}")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rng = random.Random(seed)
    samples = []
    for c in classes:
        vids = sorted(p.name for p in (Path(vroot) / c).glob("*")
                      if p.suffix.lower() in VIDEO_EXTS)
        rng.shuffle(vids)
        k = int(len(vids) * ratio)
        for v in (vids[:k] if train else vids[k:]):
            samples.append((f"{c}/{v}", class_to_idx[c]))
    return vroot, samples, len(classes)


def build_dataset(name, data_root, split=1, train=True):
    """Returns (VideoClipDataset, num_classes). Robust to several on-disk layouts:
    official UCF101 split files, a pre-split train/test folder tree, or a single
    folder of class subdirectories.
    """
    base = Path(data_root) / name
    if not base.exists():
        base = Path(data_root)                              # Kaggle: data_root IS the dataset

    result = None
    if name == "ucf101":
        result = _parse_ucf101_official(base, split, train)
    if result is None:
        result = _parse_presplit_folders(base, train)
    if result is None:
        result = _parse_folder_split(base, train)

    vroot, samples, n = result
    if not samples:
        raise FileNotFoundError(
            f"No video clips found for '{name}' under {base}. Ensure the dataset "
            f"is added and contains class subfolders or official split files."
        )
    return VideoClipDataset(samples, vroot, train), n
