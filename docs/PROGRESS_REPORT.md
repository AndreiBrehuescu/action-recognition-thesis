# Master's Thesis — Progress Report

**Title:** *Comparative study of CNN, RNN, 3D-CNN, and Transformer architectures for human action recognition: accuracy vs. efficiency trade-offs.*

**Status:** **Experiments complete.** All four architecture families were trained and benchmarked end-to-end on both datasets (UCF101 and HMDB51) under one strictly shared pipeline. Final accuracy and efficiency results are in §8. Remaining work is analysis and write-up.

---

## 1. Research question and motivation

**Question.** Human action recognition — classifying *what a person is doing* in a short video clip — has been addressed by four successive families of deep-learning architectures. This thesis asks a single, focused question:

> As we move from older to newer architectures, **how much accuracy do we gain, and at what computational cost?** Is the newest, most powerful model always the right choice, or do older/simpler models remain competitive once *efficiency* (compute, memory, speed) is taken into account?

**Why it matters.** Most published papers report only accuracy, and they each use *different* training recipes (different input resolution, number of frames, augmentation, training budget). That makes it impossible to tell whether one model beats another **because of its architecture** or **because it was trained with more resources**. In real deployments (mobile devices, embedded cameras, real-time systems), efficiency is just as important as accuracy. A rigorous, *controlled* accuracy-vs-efficiency comparison is therefore both scientifically and practically useful — and it is the contribution of this work.

---

## 2. The four architectures under study

Each represents one era / paradigm in the evolution of video understanding:

| Family | Model used | Paradigm | Temporal modeling | Pretraining |
|---|---|---|---|---|
| **2D-CNN** | ResNet-50 (TSN-style) | Per-frame image CNN + averaging | None (frames pooled) | ImageNet |
| **CNN-RNN** | ResNet-50 + bidirectional LSTM | Per-frame CNN features + recurrence | Sequential (LSTM over time) | ImageNet |
| **3D-CNN** | R(2+1)D-18 | Spatiotemporal convolution | Learned 3D filters (factorized) | Kinetics-400 |
| **Transformer** | VideoMAE-base | Self-attention over patches | Global attention | Kinetics-400 (self-supervised + supervised) |

**Why these four:**

- **ResNet-50 (2D-CNN, TSN-style)** — the baseline. It applies a standard image classifier to each of the 16 frames independently and averages the results. It has **no understanding of motion or temporal order** — it sees video as a "bag of frames." This tells us how far you can get with pure appearance/spatial information. (TSN = Temporal Segment Networks, Wang et al. 2016.)

- **ResNet-50 + LSTM (CNN-RNN)** — represents the recurrent era (the dominant approach ~2015–2018). It extracts a feature vector from each frame with the **same ResNet-50 backbone as the baseline**, then feeds the sequence of features to a bidirectional LSTM that models temporal order explicitly. This is the LRCN ("Long-term Recurrent Convolutional Network", Donahue et al. 2015) design. **Deliberately sharing the baseline's backbone turns this into a controlled experiment:** the only difference from the 2D-CNN baseline is *temporal averaging vs. temporal recurrence*, so any accuracy difference is attributable to recurrence alone.

- **R(2+1)D-18 (3D-CNN)** — represents the convolutional spatiotemporal era. It learns filters that span both space *and* time, so it can model motion directly. R(2+1)D factorizes each expensive 3D convolution into a 2D (spatial) convolution followed by a 1D (temporal) convolution, which trains better and is cheaper than full 3D. (Tran et al. 2018.)

- **VideoMAE-base (Transformer)** — represents the current state of the art. It splits the clip into spatiotemporal patches and uses self-attention to relate all patches to each other (long-range, global context). It is pretrained with *masked autoencoding* (≈90% of patches are hidden and the model learns to reconstruct them), which is highly data-efficient. (Tong et al. 2022.)

This gives a clean progression — **bag-of-frames → recurrence → spatiotemporal convolution → global attention** — along which we measure the accuracy/efficiency trade-off.

---

## 3. The methodological core: a strictly fair comparison

This is the scientific heart of the thesis. **The only thing allowed to differ between the four experiments is the architecture itself.** Everything else is held identical through a single shared pipeline:

- **Identical input:** every model receives exactly the same tensor — **16 frames**, sampled uniformly across the clip, resized and cropped to **224×224**, normalized with the same ImageNet mean/standard deviation. Shape `[batch, 16, 3, 224, 224]`.
- **Identical data handling:** same frame-sampling rule, same data augmentation (resize to 256 → random 224 crop + horizontal flip during training; center crop for evaluation).
- **Identical training recipe:** same optimizer (AdamW), same learning-rate schedule (cosine annealing), same mixed-precision training (AMP), same number of epochs (10), same batch size (8), same dataset splits.
- **Common interface:** all four models are wrapped so they accept the identical input and output class scores `[batch, num_classes]`, even though internally they are completely different.

Because the training recipe and data are frozen, any difference in accuracy or efficiency can be **attributed to the architecture alone** — which is exactly what most prior comparisons cannot claim.

---

## 4. Datasets

- **UCF101 (primary)** — 101 action classes (e.g., *ApplyEyeMakeup, Archery, PlayingGuitar*), ~13,300 video clips.
- **HMDB51 (secondary)** — 51 classes, a harder and smaller benchmark (~6,800 clips), used to check that conclusions generalize beyond a single dataset.

Both are standard, widely-cited academic benchmarks for action recognition.

**Split protocol — a noted limitation.** The publicly mirrored versions used here do not ship the official UCF101/HMDB51 split files, so a **deterministic, per-class train/test split** is used instead (UCF101 from the mirror's own train/test folders; HMDB51 a fixed 70/30 per-class split). This means absolute numbers are not directly comparable to papers that use the official split 1 — but the comparison **between our four models remains valid**, because every model is trained and evaluated on the *exact same* split. Wiring the official split files is a possible refinement (§10).

---

## 5. Why transfer learning (pretraining + fine-tuning), not training from scratch

UCF101 and HMDB51 are **too small** to train large modern models from scratch to competitive accuracy — doing so would overfit and would not reflect how these models are actually used. The standard and correct practice is:

1. Start from weights **pretrained on Kinetics-400** (a large action-recognition dataset of ~240k clips) or **ImageNet**.
2. **Fine-tune** on UCF101 / HMDB51.

This is also what makes the comparison fair and realistic: each architecture starts from its best publicly available pretrained checkpoint, and we measure how well each *adapts*. (When the Transformer is loaded, its final classification layer is automatically resized from 400 Kinetics classes to the dataset's class count and re-initialized — expected, standard fine-tuning behavior.)

A consequence worth noting: the two 2D-CNN-based models (ResNet-TSN and the CNN-LSTM) start from **ImageNet** weights, whereas the 3D-CNN and Transformer start from **Kinetics-400** (video) weights. This reflects each architecture's *best available* checkpoint and is part of what the comparison legitimately captures — but it should be stated, since some of the appearance-vs-motion gap follows from the pretraining source as well as the architecture.

---

## 6. What was implemented (engineering work done)

A clean, modular, reproducible codebase was built and version-controlled on GitHub:

- `config.py` — single source of truth for the shared pipeline (16 frames, 224 px, normalization). **This file is what guarantees fairness.**
- `datasets.py` — loads and decodes videos (using the fast `decord` library), samples frames, applies the shared transforms. Made **layout-agnostic**: it handles both video files and the "extracted-frame" layout (where each clip is a folder of JPGs), so it works regardless of how a dataset is physically organized on disk.
- `models.py` — a factory that builds any of the four models behind one common interface.
- `train.py` — the shared training loop (AdamW + cosine schedule + mixed precision), logging accuracy per epoch and saving the best checkpoint. Includes a **skip-guard** so finished models are not retrained when a run resumes.
- `evaluate.py` — computes top-1 / top-5 accuracy.
- `benchmark.py` — measures efficiency (parameters, GFLOPs, latency, throughput, peak memory).
- `report.py` / `run_all.py` — a single driver that trains → benchmarks → produces the final comparison plot for all four models.
- A self-contained **Google Colab notebook** that clones the repository, downloads the data, trains all four models, benchmarks them, and produces the Pareto plot — persisting results to Google Drive so a dropped session resumes where it left off.

The whole pipeline was tested and run end-to-end on both datasets: all four architectures correctly ingest the shared input and produce valid predictions.

---

## 7. Compute infrastructure (and why)

- **Google Colab Pro (NVIDIA L4, 22 GB)** — the **primary** training platform. All final results were produced here. Chosen over Kaggle because Kaggle's hard 12-hour session limit cannot fit all models × 10 epochs in one run; Colab Pro's longer runtimes and Drive-persisted results (with the skip-guard) make the full multi-model sweep practical and resumable.
- **Kaggle free GPUs (NVIDIA T4)** — used earlier for prototyping and the first validation runs.
- **Local AMD Radeon RX 7900 XTX (24 GB)** via **WSL2 + ROCm** — available as a fallback training engine (the same PyTorch `device="cuda"` code runs on it). Other Windows paths were evaluated and rejected: DirectML has broken 3D convolution, ZLUDA is not training-ready, and native-Windows ROCm is inference-only — WSL2 + ROCm is the only fully working local training path as of 2026.

---

## 8. Final results

All four models, both datasets, 10 epochs, batch size 8, identical shared pipeline. Accuracy is best top-1 over training; efficiency is measured on a single clip.

### UCF101 (101 classes)

| Model | Top-1 (%) | Params (M) | GFLOPs/clip | Latency (ms) | Peak mem (MB) |
|---|---|---|---|---|---|
| VideoMAE (Transformer) | **98.14** | 86 | 135 | 52 | 432 |
| R(2+1)D-18 (3D-CNN) | 98.00 | 31 | 163 | 45 | 582 |
| ResNet-50 + LSTM (CNN-RNN) | 97.91 | 34 | 66* | 19 | 323 |
| ResNet-50 TSN (2D-CNN) | 97.59 | 24 | 66 | 18 | 281 |

### HMDB51 (51 classes)

| Model | Top-1 (%) | Params (M) | GFLOPs/clip | Latency (ms) | Peak mem (MB) |
|---|---|---|---|---|---|
| VideoMAE (Transformer) | **85.48** | 86 | 135 | 52 | 432 |
| R(2+1)D-18 (3D-CNN) | 79.63 | 31 | 163 | 45 | 581 |
| ResNet-50 + LSTM (CNN-RNN) | 75.05 | 34 | 66* | 19 | 323 |
| ResNet-50 TSN (2D-CNN) | 74.27 | 24 | 66 | 18 | 281 |

\* The CNN-LSTM reports the same GFLOPs as the 2D-CNN because the FLOP counter (`fvcore`) has no handler for LSTM layers, so the recurrent part is not counted. The omission is negligible — the bidirectional LSTM adds on the order of ~0.3 GFLOPs versus the backbone's 66 — but it is noted here for transparency.

### Key findings

1. **Clear paradigm ranking on the hard dataset.** On HMDB51 the four families separate cleanly: **Transformer (85.5) ≫ 3D-CNN (79.6) > CNN-RNN (75.1) ≈ 2D-CNN (74.3).** This is the dataset that makes the thesis argument, because it actually discriminates between architectures.

2. **UCF101 is near-saturated by transfer learning.** All four models land within **0.55%** of each other (97.6–98.1%). This is itself a finding: with strong pretraining, UCF101 no longer distinguishes architectures — which justifies using HMDB51 as the discriminative second benchmark rather than relying on UCF101 alone.

3. **The controlled recurrence experiment.** Because the CNN-LSTM and the ResNet-TSN baseline share an identical backbone, their gap isolates *recurrence vs. averaging*. Recurrence wins by **+0.32% on UCF101** and **+0.78% on HMDB51** — a small but **consistent** gain, at the cost of +10M parameters and slightly higher latency. Conclusion: explicit temporal recurrence over 16 sparsely-sampled frames adds only marginally over naive temporal pooling — which helps explain why the field moved on to 3D convolution and attention.

4. **The trade-off is multi-dimensional.** "Efficiency" depends on the axis:
   - **By compute (GFLOPs):** VideoMAE (135) is *both cheaper and more accurate* than R(2+1)D (163) — i.e., on this axis the 3D-CNN is **Pareto-dominated** by the Transformer.
   - **By latency / parameters:** R(2+1)D (45 ms, 31M) is **lighter** than VideoMAE (52 ms, 86M).
   - **The cheapest models** (ResNet-TSN and CNN-LSTM, ~66 GFLOPs, ~18–19 ms) give up ~6–11 accuracy points on HMDB51 for roughly **half the compute and memory** of the Transformer.

   So the "best" model genuinely depends on the deployment budget — exactly the trade-off the thesis sets out to characterize.

---

## 9. Evaluation: how the models are compared

**Accuracy (effectiveness):**

| Metric | What it tells us |
|---|---|
| Top-1 accuracy | Standard correctness — exact-match rate |
| Top-5 accuracy | Whether the correct class is among the 5 best guesses |
| Confusion matrix | *Which* actions get confused (qualitative error analysis) |

**Efficiency (cost):**

| Metric | What it tells us |
|---|---|
| Parameters (millions) | Model size / storage footprint |
| GFLOPs per clip | Compute per inference — **hardware-independent**, the fairest cost measure |
| Latency (ms/clip) | Real inference time on a given GPU |
| Throughput (clips/s) | Clips processed per second — deployment relevance |
| Peak memory (MB) | GPU memory required — determines what hardware can run it |

**Headline figure:** an **accuracy-vs-GFLOPs Pareto scatter plot** (one per dataset), where each model is a point (bubble size = parameter count). This single chart captures the central thesis question: which architectures sit on the "best trade-off" frontier, and which are dominated. Both plots (`pareto_ucf101.png`, `pareto_hmdb51.png`) are generated automatically by the pipeline.

---

## 10. Next steps (roadmap)

The experimental phase is finished. Remaining work is analysis and writing:

1. **Write the results chapter** around the two tables and the four findings in §8.
2. **Confusion-matrix / per-class error analysis** — which actions each architecture confuses, especially on HMDB51.
3. **Discussion** of the accuracy/efficiency trade-off: when the Transformer's accuracy gain justifies its compute cost, and where the cheap 2D models remain the pragmatic choice.
4. *(Optional refinements)* wire the **official UCF101/HMDB51 split files** to make absolute numbers literature-comparable; report mean±std over the three official splits; equalize pretraining sources for a stricter ablation.

---

## 11. Deliverables

- A reproducible, open-source codebase (on GitHub).
- A results table: accuracy + all efficiency metrics, for **4 models × 2 datasets** (complete — §8).
- The accuracy-vs-efficiency Pareto figures (complete — one per dataset).
- Confusion matrices and per-class analysis (in progress).
- A written discussion answering the research question: **how the accuracy/efficiency trade-off evolves from 2D-CNN → CNN-RNN → 3D-CNN → Transformer, and what that means for choosing a model in practice.**
