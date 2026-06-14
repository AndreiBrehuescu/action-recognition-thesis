# Dizertație — Action Recognition: CNN vs 3D-CNN vs Transformer

A comparative study of three deep-learning architecture families for **human action
recognition**, focused on the **accuracy ↔ efficiency trade-off**.

## Models (one per architectural era)

| Family        | Model                | Source                                              | Params | Pretrained   |
|---------------|----------------------|-----------------------------------------------------|--------|--------------|
| 2D-CNN        | ResNet-50 (TSN-style)| `torchvision` (per-frame + temporal averaging)      | ~25 M  | ImageNet     |
| 3D-CNN        | R(2+1)D-18           | `torchvision.models.video`                          | ~31 M  | Kinetics-400 |
| Transformer   | VideoMAE-base        | HF `MCG-NJU/videomae-base-finetuned-kinetics`       | ~87 M  | Kinetics-400 |

Optional later additions: **TimeSformer** (transformer) or **SlowFast** (3D-CNN) as a 4th point.

## Datasets

- **UCF101** (primary) — 101 classes, 13,320 clips (~7 GB). Use the official **split 1**.
- **HMDB51** (secondary, generalization check) — 51 classes, ~6,800 clips (~2 GB).

## Why the comparison is *fair*

Every model is driven through **one shared pipeline** (`src/config.py` + `src/datasets.py`):
16 frames/clip, 224×224, identical ImageNet normalization, identical train/test splits.
Only the model swaps — so any accuracy/efficiency difference is attributable to the
architecture, not the data plumbing. This is the methodological core of the thesis.

## Evaluation axes

- **Accuracy:** top-1 / top-5 on UCF101 (then HMDB51), plus a confusion matrix.
- **Efficiency:** parameter count, **GFLOPs/clip** (hardware-independent), inference
  **latency** (ms/clip) + throughput, **peak GPU memory**, and training time.
- **Headline figure:** an accuracy-vs-GFLOPs **Pareto scatter**.

## Compute

PyTorch sees AMD ROCm as `device="cuda"`, so the **same code runs on both** of:

1. **Kaggle / Colab** (NVIDIA, zero setup) — start here.
2. **Local RX 7900 XTX via WSL2 + ROCm** — the workhorse (24 GB, unlimited hours).

> DirectML, ZLUDA, and native-Windows ROCm do **not** reliably train video models in 2026
> (3D-conv gaps / training unsupported), so we don't use them.

### Kaggle quickstart
1. New Notebook → **Add Data** → UCF101 dataset.
2. Set accelerator to **GPU (T4/P100)**.
3. `!pip -q install transformers decord fvcore` (torch/torchvision are preinstalled).
4. Get this code into the notebook (clone from GitHub once pushed, or upload as a dataset).
5. Run `python -m src.train --model videomae --data-root /kaggle/input/<ucf101-path>`.

### WSL2 + ROCm quickstart (local 7900 XTX)
1. Install **AMD Adrenalin 26.1.1+** on Windows, then Ubuntu 22.04 via WSL2.
2. Follow AMD's guide: <https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html>
3. `export PYTORCH_ROCM_ARCH="gfx1100"` and `export HSA_OVERRIDE_GFX_VERSION=11.0.0`
4. Install PyTorch/torchvision from **repo.radeon.com** (NOT from pypi.org).
5. `pip install -r requirements.txt` for the rest.
6. Sanity check: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

## Project layout

```
.
├── PROJECT.md            # this file
├── requirements.txt      # everything EXCEPT torch/torchvision (env-specific)
├── src/
│   ├── config.py         # single source of truth for the shared pipeline
│   ├── datasets.py       # UCF101/HMDB51 loaders + shared transforms
│   ├── models.py         # build_model(): the 3 families, common (B,T,C,H,W) I/O
│   ├── train.py          # one model-agnostic training loop
│   ├── evaluate.py       # top-1/top-5 + confusion matrix
│   └── benchmark.py      # params / FLOPs / latency / peak memory
├── scripts/
│   └── setup_data.py     # lay out folders + fetch official split lists
└── results/              # metrics CSVs, checkpoints, plots (gitignored)
```

## Usage

```bash
# 0) one-time: fetch UCF101 split lists + create folders (then add the videos)
python scripts/setup_data.py --dataset ucf101 --splits-only

# 1) train any model through the identical pipeline
python -m src.train    --model videomae    --dataset ucf101 --epochs 5 --batch-size 8
python -m src.train    --model r2plus1d_18  --dataset ucf101 --epochs 5 --batch-size 8
python -m src.train    --model resnet50_tsn --dataset ucf101 --epochs 5 --batch-size 8

# 2) evaluate a saved checkpoint (+ optional confusion matrix)
python -m src.evaluate --model videomae --dataset ucf101 --confusion

# 3) efficiency numbers (params / FLOPs / latency / memory) -> results/efficiency.csv
python -m src.benchmark --model videomae
python -m src.benchmark --model r2plus1d_18
python -m src.benchmark --model resnet50_tsn
```

## Roadmap

- **M1** Data pipeline runs (UCF101 decodes, splits load).
- **M2** VideoMAE fine-tunes end-to-end on UCF101 (first accuracy number).
- **M3** All three families train through the unified harness.
- **M4** Efficiency benchmarking (FLOPs / latency / memory) collected.
- **M5** HMDB51 generalization + accuracy-vs-FLOPs Pareto analysis.
- **M6** Write-up.
