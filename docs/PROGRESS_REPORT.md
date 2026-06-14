# Master's Thesis — Progress Report

**Title:** *Comparative study of CNN, 3D-CNN, and Transformer architectures for human action recognition: accuracy vs. efficiency trade-offs.*

**Status:** Pipeline implemented and validated end-to-end; training in progress. First model (Transformer / VideoMAE) reaching ~94.7% top-1 on UCF101 after one epoch, confirming correctness.

---

## 1. Research question and motivation

**Question.** Human action recognition — classifying *what a person is doing* in a short video clip — has been solved with three successive families of deep-learning architectures. This thesis asks a single, focused question:

> As we move from older to newer architectures, **how much accuracy do we gain, and at what computational cost?** Is the newest, most powerful model always the right choice, or do older/simpler models remain competitive once *efficiency* (compute, memory, speed) is taken into account?

**Why it matters.** Most published papers report only accuracy, and they each use *different* training recipes (different input resolution, number of frames, augmentation, training budget). That makes it impossible to tell whether one model beats another **because of its architecture** or **because it was trained with more resources**. In real deployments (mobile devices, embedded cameras, real-time systems), efficiency is just as important as accuracy. A rigorous, *controlled* accuracy-vs-efficiency comparison is therefore both scientifically and practically useful — and it is the contribution of this work.

---

## 2. The three architectures under study

Each represents one era / paradigm in the evolution of video understanding:

| Family | Model used | Paradigm | Temporal modeling | Pretraining |
|---|---|---|---|---|
| **2D-CNN** | ResNet-50 (TSN-style) | Per-frame image CNN + averaging | None (frames pooled) | ImageNet |
| **3D-CNN** | R(2+1)D-18 | Spatiotemporal convolution | Learned 3D filters (factorized) | Kinetics-400 |
| **Transformer** | VideoMAE-base | Self-attention over patches | Global attention | Kinetics-400 (self-supervised + supervised) |

**Why these three:**

- **ResNet-50 (2D-CNN, TSN-style)** — the baseline. It applies a standard image classifier to each of the 16 frames independently and averages the results. It has **no real understanding of motion or temporal order** — it sees video as a "bag of frames." This tells us how far you can get with pure appearance/spatial information. (TSN = Temporal Segment Networks, Wang et al. 2016.)

- **R(2+1)D-18 (3D-CNN)** — represents the convolutional spatiotemporal era. It learns filters that span both space *and* time, so it can model motion directly. R(2+1)D factorizes each expensive 3D convolution into a 2D (spatial) convolution followed by a 1D (temporal) convolution, which trains better and is cheaper than full 3D. (Tran et al. 2018.)

- **VideoMAE-base (Transformer)** — represents the current state of the art. It splits the clip into spatiotemporal patches and uses self-attention to relate all patches to each other (long-range, global context). It is pretrained with *masked autoencoding* (≈90% of patches are hidden and the model learns to reconstruct them), which is highly data-efficient. (Tong et al. 2022.)

This gives a clean progression — **bag-of-frames → spatiotemporal convolution → global attention** — along which we measure the accuracy/efficiency trade-off.

---

## 3. The methodological core: a strictly fair comparison

This is the scientific heart of the thesis. **The only thing allowed to differ between the three experiments is the architecture itself.** Everything else is held identical through a single shared pipeline:

- **Identical input:** every model receives exactly the same tensor — **16 frames**, sampled uniformly across the clip, resized and cropped to **224×224**, normalized with the same ImageNet mean/standard deviation. Shape `[batch, 16, 3, 224, 224]`.
- **Identical data handling:** same frame-sampling rule, same data augmentation (resize to 256 → random 224 crop + horizontal flip during training; center crop for evaluation).
- **Identical training recipe:** same optimizer (AdamW), same learning-rate schedule (cosine annealing), same mixed-precision training (AMP), same number of epochs, same dataset splits.
- **Common interface:** all three models are wrapped so they accept the identical input and output class scores `[batch, num_classes]`, even though internally they are completely different.

Because the training recipe and data are frozen, any difference in accuracy or efficiency can be **attributed to the architecture alone** — which is exactly what most prior comparisons cannot claim.

---

## 4. Datasets

- **UCF101 (primary)** — 101 action classes (e.g., *ApplyEyeMakeup, Archery, PlayingGuitar*), ~13,320 video clips. We use the **official split 1** train/test protocol so results are comparable with the literature.
- **HMDB51 (secondary)** — 51 classes, a harder and smaller benchmark, used to check that conclusions generalize beyond a single dataset.

Both are standard, widely-cited academic benchmarks for action recognition.

---

## 5. Why transfer learning (pretraining + fine-tuning), not training from scratch

UCF101 and HMDB51 are **too small** to train large modern models from scratch to competitive accuracy — doing so would overfit and would not reflect how these models are actually used. The standard and correct practice is:

1. Start from weights **pretrained on Kinetics-400** (a large action-recognition dataset of ~240k clips) or ImageNet.
2. **Fine-tune** on UCF101 / HMDB51.

This is also what makes the comparison fair and realistic: each architecture starts from its best publicly available pretrained checkpoint, and we measure how well each *adapts*. (When the Transformer is loaded, its final classification layer is automatically resized from 400 Kinetics classes to 101 UCF101 classes and re-initialized — expected, standard fine-tuning behavior.)

---

## 6. What was implemented (engineering work done so far)

A clean, modular, reproducible codebase was built and version-controlled on GitHub:

- `config.py` — single source of truth for the shared pipeline (16 frames, 224 px, normalization). **This file is what guarantees fairness.**
- `datasets.py` — loads and decodes videos (using the fast `decord` library), samples frames, applies the shared transforms. Made **layout-agnostic** so it works regardless of how a dataset is physically organized on disk.
- `models.py` — a factory that builds any of the three models behind one common interface.
- `train.py` — the shared training loop (AdamW + cosine schedule + mixed precision), logging accuracy per epoch and saving the best checkpoint.
- `evaluate.py` — computes top-1 / top-5 accuracy and a confusion matrix.
- `benchmark.py` — measures efficiency (parameters, GFLOPs, latency, throughput, peak memory).
- A self-contained **Kaggle notebook** that clones the repository, auto-detects the dataset, trains all three models, benchmarks them, and produces the final comparison plot.

The whole pipeline was tested end-to-end: all three architectures correctly ingest the shared input and produce valid predictions.

---

## 7. Compute infrastructure (and why)

Two interchangeable compute paths are used, because **the same code runs on both** (PyTorch exposes both NVIDIA and AMD GPUs through the same `device="cuda"` interface):

- **Kaggle free GPUs (NVIDIA T4)** — zero setup, used for prototyping and the first full runs. Currently in use.
- **Local AMD Radeon RX 7900 XTX (24 GB)** via **WSL2 + ROCm** — for bulk and repeated training. (Other Windows options were evaluated and rejected: DirectML has broken 3D convolution, ZLUDA is not training-ready, and native-Windows ROCm is inference-only — WSL2 + ROCm is the only fully working local training path as of 2026.)

This dual setup de-risks the project: prototyping happens immediately on Kaggle while the local machine is the long-term training engine.

---

## 8. Current status and preliminary result

The pipeline is **fully working**. The first model under training — **VideoMAE (Transformer)** on UCF101 — reached, after a **single epoch**:

- **Top-1 accuracy: 94.67%**
- **Top-5 accuracy: 99.82%**

Random-chance accuracy on 101 classes is ~1%, so this result confirms the data, labels, and evaluation are all correct, and it is already in the published state-of-the-art range for VideoMAE on UCF101. The remaining two models (R(2+1)D-18 and ResNet-50) are queued to train through the same pipeline.

---

## 9. Evaluation: how the models will be compared

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

**Headline figure:** an **accuracy-vs-GFLOPs Pareto scatter plot**, where each model is a point (bubble size = parameter count). This single chart captures the central thesis question: which architectures sit on the "best trade-off" frontier, and which are dominated.

---

## 10. Next steps (roadmap)

1. **Finish training** all three models on UCF101 (identical recipe) and record final accuracies.
2. **Run the efficiency benchmark** (GFLOPs, latency, memory) for each model.
3. **Generate the Pareto plot** (accuracy vs. compute) — the thesis headline result.
4. **Repeat on HMDB51** (after wiring its official split protocol) to confirm the findings generalize.
5. **Bring the local AMD 7900 XTX online** via WSL2 + ROCm for repeated/longer runs.
6. **Analysis & write-up:** confusion-matrix and per-class error analysis, and a discussion of the accuracy/efficiency trade-off (e.g., "does the Transformer's accuracy gain justify its compute cost versus the 3D-CNN?").
7. *(Optional)* Add a fourth architecture (e.g., TimeSformer or SlowFast) for a richer trade-off frontier.

---

## 11. Expected deliverables

- A reproducible, open-source codebase (already on GitHub).
- A results table: accuracy + all efficiency metrics, for 3 models × 2 datasets.
- The accuracy-vs-efficiency Pareto figure (headline).
- Confusion matrices and per-class analysis.
- A written discussion answering the research question: **how the accuracy/efficiency trade-off evolves from 2D-CNN → 3D-CNN → Transformer, and what that means for choosing a model in practice.**
