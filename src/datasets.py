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


# ------------------------------------------------- universal layout-agnostic scan
_TRAIN_TOKENS = {"train", "training"}
_TEST_TOKENS = {"test", "testing", "val", "valid", "validation", "eval"}


def _scan_videos(root):
    """Every video file anywhere under root."""
    return [f for f in Path(root).rglob("*")
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS]


def _split_tag(path, root):
    """'train' / 'test' if any path component says so, else None."""
    parts = {p.lower() for p in Path(path).relative_to(root).parts[:-1]}
    if parts & _TEST_TOKENS:
        return "test"
    if parts & _TRAIN_TOKENS:
        return "train"
    return None


def _parse_by_scan(root, train, seed=42, ratio=0.7):
    """Layout-agnostic: find every video under root, label each by its PARENT
    folder name (the class), and split by any train/test path component. If the
    data isn't pre-split, fall back to a deterministic per-class 70/30 split.

    Handles arbitrary nesting and wrapper folders, e.g.:
        root/train/ApplyEyeMakeup/v_x.avi
        root/UCF-101/ApplyEyeMakeup/v_x.avi
        root/anything/ApplyEyeMakeup/v_x.avi
    """
    root = Path(root)
    vids = _scan_videos(root)
    if not vids:
        return None
    classes = sorted({v.parent.name for v in vids})
    class_to_idx = {c: i for i, c in enumerate(classes)}

    tags = {v: _split_tag(v, root) for v in vids}
    pre_split = any(t == "train" for t in tags.values()) and \
                any(t == "test" for t in tags.values())

    samples = []
    if pre_split:
        want = "train" if train else "test"
        for v in vids:
            if tags[v] == want:
                samples.append((v.relative_to(root).as_posix(), class_to_idx[v.parent.name]))
    else:
        rng = random.Random(seed)
        by_class = {}
        for v in vids:
            by_class.setdefault(v.parent.name, []).append(v)
        for c in classes:
            files = sorted(by_class[c], key=lambda p: p.name)
            rng.shuffle(files)
            k = int(len(files) * ratio)
            for v in (files[:k] if train else files[k:]):
                samples.append((v.relative_to(root).as_posix(), class_to_idx[c]))
    return root, samples, len(classes)


def _describe_tree(root, max_lines=25):
    """Compact tree (<=2 levels deep) for self-diagnosing error messages."""
    root = Path(root)
    if not root.exists():
        return f"  {root} does not exist"
    lines, n = [], 0
    for p in sorted(root.rglob("*")):
        depth = len(p.relative_to(root).parts) - 1
        if depth > 2:
            continue
        lines.append("  " + "  " * depth + p.name + ("/" if p.is_dir() else ""))
        n += 1
        if n >= max_lines:
            lines.append("  ...")
            break
    return "\n".join(lines) if lines else "  (empty)"


def find_data_root(input_dir="/kaggle/input"):
    """Auto-locate a video dataset under input_dir (handy on Kaggle).

    Scores each immediate sub-directory and returns the best match, preferring
    (1) official split files, (2) a 'ucf'/'hmdb' name, (3) more videos -- so a
    stray folder that merely happens to contain a clip never wins over the real
    dataset. Prints a per-entry diagnostic.
    """
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"{input_dir} does not exist")
    best = None
    print(f"Scanning {input_dir} ...")
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        has_split = next(iter(p.rglob("classInd.txt")), None) is not None
        vcount = 0
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
                vcount += 1
                if vcount >= 100:                           # cap for speed
                    break
        name_hit = any(k in p.name.lower() for k in ("ucf", "hmdb"))
        print(f"  {p.name:35s}  split_files={has_split}  videos>={vcount}  name_match={name_hit}")
        if has_split or vcount > 0:
            score = (has_split, name_hit, vcount)
            if best is None or score > best[0]:
                best = (score, str(p))
    if best is None:
        raise FileNotFoundError(
            f"No dataset with videos or split files under {input_dir}. "
            "Add a UCF101 *dataset* (not a notebook) via '+ Add Input'."
        )
    print(f"Using DATA_ROOT = {best[1]}")
    return best[1]


def build_dataset(name, data_root, split=1, train=True):
    """Returns (VideoClipDataset, num_classes). Layout-agnostic: uses official
    UCF101 split files when present, otherwise scans for videos and labels each
    by its parent folder, respecting any train/test folders in the path.
    """
    base = Path(data_root) / name
    if not base.exists():
        base = Path(data_root)                              # Kaggle: data_root IS the dataset

    result = None
    if name == "ucf101":
        result = _parse_ucf101_official(base, split, train)
    if result is None:
        result = _parse_by_scan(base, train)

    if not result or not result[1]:
        raise FileNotFoundError(
            f"No video clips found for '{name}' under {base}.\n"
            f"Directory structure seen (<=2 levels):\n{_describe_tree(base)}"
        )
    vroot, samples, n = result
    return VideoClipDataset(samples, vroot, train), n
