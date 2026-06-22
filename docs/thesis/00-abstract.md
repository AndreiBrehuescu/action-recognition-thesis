# Comparative Analysis of Deep Learning Architectures for Human Action Recognition: CNNs, RNNs, 3D-CNNs, and Transformers

**Author:** Andrei Brehuescu
**Institution:** Technical University of Cluj-Napoca, Faculty of Automation and Computer Science
**Academic Year:** 2025–2026

---

## Abstract

Human action recognition — the automatic identification of physical activities performed by one or more persons from video sequences — represents one of the most practically significant and technically demanding problems in contemporary computer vision. The proliferation of video data across surveillance systems, social media platforms, medical monitoring infrastructure, and autonomous vehicles has created an urgent need for accurate, efficient, and broadly deployable recognition architectures. Over the past decade, four principal deep learning paradigms have emerged as the dominant approaches to this problem: two-dimensional convolutional neural networks applied over temporal segments (2D-CNN / Temporal Segment Networks), recurrent augmentations of convolutional feature extractors (CNN-RNN), volumetric three-dimensional convolutional networks (3D-CNN), and self-attention-based Transformer architectures. Each paradigm has individually demonstrated impressive results on standard benchmarks, yet a principled, controlled comparison of all four families under identical experimental conditions has not been systematically reported in the literature. This dissertation addresses that gap directly.

This work presents a rigorous four-way comparative study evaluating representative models from each paradigm — ResNet-50 Temporal Segment Network (TSN), ResNet-50 with bidirectional Long Short-Term Memory (CNN-LSTM), R(2+1)D-18, and VideoMAE (ViT-B) — on two widely adopted action recognition benchmarks: UCF101 (101 classes, 13,320 clips) and HMDB51 (51 classes, 6,766 clips). All models are trained and evaluated under a strictly unified protocol: 10 epochs, AdamW optimiser with learning rate 1×10⁻⁴, cosine annealing with warmup, batch size 8, FP16 automatic mixed precision, 16 uniformly sampled frames per clip at 224×224 spatial resolution, and single-clip center-crop inference — all executed on a single NVIDIA L4 22 GB GPU accessed via Google Colab Pro.

The experimental results reveal a clear performance hierarchy on the more discriminative HMDB51 dataset, spanning 11.21 percentage points: VideoMAE achieves 85.48%, R(2+1)D-18 achieves 79.63%, CNN-LSTM achieves 75.05%, and ResNet-50 TSN achieves 74.27%. On the saturated UCF101 benchmark, all four models converge within a narrow 0.55 percentage point band (97.59%–98.14%), underscoring the inadequacy of UCF101 as a differentiating benchmark under modern pretraining regimes. Efficiency profiling reveals that VideoMAE Pareto-dominates R(2+1)D-18 on the accuracy-versus-GFLOPs axis, achieving higher accuracy at lower computational cost (135.17 versus 162.58 GFLOPs per clip). A controlled ablation comparing TSN and CNN-LSTM — which share an identical ImageNet-pretrained ResNet-50 backbone — demonstrates that adding recurrence yields only 0.32 percentage points on UCF101 and 0.78 percentage points on HMDB51 at the cost of 10.4 million additional parameters.

The practical conclusion is that for deployment scenarios demanding the highest accuracy, VideoMAE with Kinetics-400 pretraining is the preferred choice, while for resource-constrained environments, ResNet-50 TSN offers competitive accuracy at a fraction of the parameter count and memory footprint. This work contributes an open, reproducible evaluation framework and a multi-dimensional Pareto characterisation of the accuracy-efficiency trade-off space across all four architectural paradigms.

---
