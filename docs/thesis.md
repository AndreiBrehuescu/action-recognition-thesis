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

# Chapter 1: Introduction

## 1.1 Background and Motivation

The capacity to automatically understand human behaviour from video is among the most consequential capabilities that computer vision can offer to society. From the early days of motion picture analysis, researchers have been fascinated by the problem of teaching machines to parse the rich, temporally extended signals that video provides — to infer not merely what objects are present in a scene, but what those objects are *doing*, and to do so reliably across the enormous diversity of viewpoints, lighting conditions, actor appearances, and background contexts that characterise video recorded in the real world.

The scale of the contemporary video landscape makes the problem both urgent and technically demanding. A Cisco Visual Networking Index report projected that video would account for approximately 82 percent of all consumer internet traffic by 2022 [1], a forecast that subsequent empirical measurements have confirmed and in many regions exceeded. This explosion of video data is driven by a convergence of factors: ubiquitous smartphone cameras have democratised video capture; social media platforms such as YouTube, TikTok, and Instagram incentivise massive video upload and consumption; IP security camera networks have expanded to encompass hundreds of millions of devices worldwide; and emerging application domains such as augmented reality, autonomous driving, and telemedicine generate yet further streams of video requiring automated interpretation.

The practical application domains for human action recognition are correspondingly broad and consequential. In **video surveillance and public safety**, automated systems capable of detecting anomalous actions — falls, fights, vehicle theft, loitering — can alert human operators faster and more reliably than exhausted security personnel monitoring dozens of simultaneous feeds. In **sports analytics**, frame-by-frame recognition of athletic actions enables automated statistics generation, tactical analysis, injury risk assessment, and broadcast highlight extraction [2]. In **human-computer interaction**, gesture and activity recognition enables natural, touchless interfaces for gaming, smart home control, and industrial automation [3]. In **healthcare monitoring**, the ability to recognise and quantify patient activities — gait patterns, fall events, rehabilitation exercises, activities of daily living — from unobtrusive cameras or wearable-mounted sensors has significant implications for elderly care, post-surgical monitoring, and neurological disorder diagnosis [4]. In **autonomous vehicles**, the recognition of pedestrian and cyclist actions — crossing the road, signalling a turn, stumbling unexpectedly — is safety-critical for collision avoidance systems [5]. In **augmented and virtual reality**, real-time body and hand action recognition enables immersive interaction and avatar animation [6].

At its core, the action recognition problem presents a fundamental challenge that sets it apart from image classification: video is an inherently spatiotemporal signal. A single video clip encodes not only a spatial scene — its objects, textures, colours, and layout — but also a temporal trajectory of that scene through time. Action classes are defined precisely by this temporal evolution: the difference between "catching a ball" and "throwing a ball" cannot be resolved from any individual frame but only from the sequence of frames and the direction of motion they encode. Any recognition system must therefore simultaneously capture **spatial appearance** — what the scene looks like — and **temporal dynamics** — how it changes. This dual requirement drives the architectural diversity that characterises the action recognition literature.

The historical trajectory of the field reflects successive attempts to solve this dual challenge with progressively more powerful representational tools. The earliest approaches relied entirely on **hand-crafted spatiotemporal features**. Histogram of Oriented Gradients (HOG) [7], originally introduced for pedestrian detection in still images, was extended to the temporal domain as Histogram of Optical Flow (HOF). Laptev's Space-Time Interest Points (STIPs) [8] detected regions of high spatio-temporal energy as sparse feature locations. The most influential pre-deep-learning approach was the Dense Trajectories framework of Wang et al. [9], which sampled feature points on a dense grid and tracked them using optical flow, computing HOG, HOF, and Motion Boundary Histogram (MBH) descriptors along each trajectory. The improved Dense Trajectories (iDT) variant [10] further suppressed camera motion by estimating and subtracting background homography, achieving state-of-the-art performance on UCF101 and HMDB51 benchmarks of its era. Despite their ingenuity, these hand-crafted methods required expensive optical flow computation, relied on fixed-vocabulary codebooks with limited expressive capacity, and provided no mechanism to adapt the representation to training data.

The deep learning revolution, ignited by Krizhevsky et al.'s AlexNet [11] on the ImageNet Large Scale Visual Recognition Challenge [12] in 2012, prompted the community to investigate whether learned convolutional features could supersede hand-crafted ones for video understanding as dramatically as they had for image recognition. Karpathy et al. [13] conducted an early large-scale study of deep CNNs for video classification using the Sports-1M dataset, exploring various late-fusion and early-fusion strategies for combining features from multiple frames. While their models learned genuinely useful representations, the improvement over single-frame baselines was modest, suggesting that naive temporal fusion of 2D CNN features was insufficient to capture motion effectively.

A decisive step forward came with the Two-Stream architecture of Simonyan and Zisserman [14], which processed spatial (RGB) and temporal (optical flow) streams with separate CNNs and fused their predictions at the score level. By explicitly providing the network with pre-computed optical flow as a motion signal, Two-Stream networks achieved a substantial accuracy improvement over single-stream approaches, validating the importance of explicit motion representation. However, the requirement for dense optical flow computation at inference time imposed a significant computational overhead that limited real-world deployability.

Wang et al.'s Temporal Segment Networks (TSN) [15] addressed the temporal modelling problem from a different angle: rather than processing dense optical flow, TSN uniformly sampled sparse snippets from across the full video, applied a shared CNN to each snippet, and fused snippet-level predictions via average pooling. This simple yet effective strategy provided a computationally efficient means of covering the full temporal extent of a video and became a widely adopted baseline. Its simplicity also makes it an ideal representative of the 2D-CNN paradigm for comparative studies: the temporal modelling mechanism is entirely in the pooling layer, and all representational capacity comes from the spatial CNN.

Recognising that frame-level features are not inherently sequential, another family of approaches explored coupling CNNs with Recurrent Neural Networks to model temporal dependencies explicitly. Hochreiter and Schmidhuber's Long Short-Term Memory (LSTM) [16] architecture, designed to capture long-range sequential dependencies through gated memory cells, provided a natural complement to CNN feature extractors. Donahue et al.'s Long-term Recurrent Convolutional Networks (LRCN) [17] demonstrated that stacking a CNN encoder with an LSTM sequence model could jointly be trained for action recognition, video captioning, and other video understanding tasks. Despite their intuitive appeal, CNN-RNN models in practice have shown only modest gains over temporal averaging approaches, suggesting that the expressiveness of the recurrent module is limited by the relatively short temporal spans of typical benchmark clips.

The volumetric convolution approach — treating video clips as three-dimensional inputs and applying 3D convolutional filters that span both space and time — was investigated by Ji et al. [18] and Tran et al.'s C3D [19], the latter demonstrating that a simple 3D CNN trained on a large dataset (Sports-1M) could learn powerful spatiotemporal features transferable to smaller benchmarks. The introduction of large-scale video pretraining datasets, particularly the Kinetics series [20], proved transformative: Carreira and Zisserman's I3D (Inflated 3D ConvNet) [21] demonstrated that inflating ImageNet-pretrained 2D CNN weights into 3D convolutions and then fine-tuning on Kinetics-400 yielded state-of-the-art performance on UCF101 and HMDB51, establishing Kinetics pretraining as the standard initialisation strategy for 3D CNN models. Tran et al.'s R(2+1)D [22] decomposed 3D convolutions into separate 2D spatial and 1D temporal convolutions, improving optimisation and achieving better accuracy than full 3D convolution at equivalent parameter counts.

The most recent architectural paradigm to transform the field is the Transformer [23], originally conceived for natural language processing but adapted to visual domains through the Vision Transformer (ViT) [24]. Extending the self-attention mechanism to video, Bertasius et al.'s TimeSformer [25] and Liu et al.'s Video Swin Transformer [26] demonstrated that purely attention-based architectures could achieve competitive or superior accuracy to 3D CNNs for action recognition. The VideoMAE model of Tong et al. [27], based on masked autoencoder pretraining of a ViT-B backbone on large video corpora, represents the current frontier of the Transformer paradigm and is the most powerful model evaluated in this study.

Across this decades-long trajectory, a consistent pattern is apparent: each new architectural paradigm supersedes its predecessors in terms of representational capacity and, under sufficiently large pretraining regimes, in raw accuracy on standard benchmarks. However, accuracy alone is insufficient to characterise the practical utility of an architecture. Parameters, computational cost, inference latency, and memory footprint are equally important for deployment in resource-constrained environments, embedded systems, or real-time streaming applications. Understanding where each paradigm sits on the multi-dimensional efficiency-accuracy Pareto frontier is a primary motivation for this study.

## 1.2 Problem Statement and Research Gap

Despite the extensive body of research on deep learning for action recognition, a fundamental scientific problem persists in the field: the four dominant architectural paradigms — 2D-CNN, CNN-RNN, 3D-CNN, and Transformer — have each been evaluated extensively in isolation, but rarely if ever under a single unified experimental protocol that controls all confounding variables simultaneously.

The comparison problem is acutely sensitive to experimental configuration. The accuracy of a video recognition model depends not merely on its architecture but on an ensemble of interacting factors: the number of frames sampled per clip; the spatial resolution of those frames; the pretraining dataset and initialisation strategy; the choice of optimiser, learning rate schedule, and number of training epochs; the use or absence of test-time augmentation (multi-clip or multi-crop ensembling); and the specific train-test split of the evaluation benchmark. Small changes in any of these factors can produce accuracy differences that dwarf the architectural differences being studied. A model evaluated with 32 frames, Kinetics-400 pretraining, 10-crop test-time augmentation, and 30 epochs of fine-tuning is not meaningfully comparable to a model evaluated with 8 frames, ImageNet pretraining, single-clip inference, and 5 epochs of fine-tuning — yet precisely such comparisons are routinely implied when citing results from different papers in a survey or related-work section.

A survey of representative results illustrates this problem concretely. State-of-the-art 3D-CNN papers typically report UCF101 accuracy using multi-clip inference with ten spatial crops, yielding figures above 98% — but these results are not directly comparable to single-clip results from CNN-RNN papers that use different frame sampling strategies. Transformer-based papers such as VideoMAE report Kinetics fine-tuning results rather than direct UCF101/HMDB51 numbers from scratch, and when they do evaluate on these benchmarks they typically use multi-clip inference that inflates accuracy relative to single-clip protocols. The consequence is that cross-architecture comparisons in the published literature are unreliable: apparent differences between architectures may reflect differences in evaluation protocols rather than genuine representational differences.

Beyond the evaluation protocol problem, there is a specific gap in the literature regarding the **simultaneous** comparison of all four paradigms in a single study. Most comparative papers evaluate two or three paradigms. Comprehensive surveys [28] review many architectures but do not provide unified experimental results — they aggregate numbers from disparate papers with disparate protocols. To the author's knowledge, no published study evaluates ResNet-50 TSN (2D-CNN), ResNet-50+LSTM (CNN-RNN), R(2+1)D-18 (3D-CNN), and VideoMAE (Transformer) under strictly identical conditions — same optimiser, same learning rate, same number of epochs, same frame count, same resolution, same batch size, same inference protocol — on both UCF101 and HMDB51 simultaneously.

A further gap concerns the design-controlled ablation of recurrence versus temporal averaging. TSN and CNN-LSTM, when both built on an identical ResNet-50 ImageNet-pretrained backbone, provide a uniquely clean comparison: the only difference is whether temporal information is integrated by average pooling of frame features (TSN) or by a bidirectional LSTM operating on the frame feature sequence (CNN-LSTM). This controlled comparison isolates the contribution of recurrence — something that cannot be done when comparing architectures that differ in backbone, pretraining, and temporal integration mechanism simultaneously. This controlled ablation is a specific contribution of the present study that has not, to the author's knowledge, been reported previously for these specific models at these scales.

## 1.3 Contributions

This dissertation makes the following specific contributions to the field of video action recognition:

**(a) Unified Four-Paradigm Evaluation Framework.** An open-source, end-to-end evaluation framework is developed and released that supports training and evaluation of all four architectural paradigms — 2D-CNN (TSN), CNN-RNN (CNN-LSTM), 3D-CNN (R(2+1)D-18), and Transformer (VideoMAE) — under a single unified codebase. The framework handles heterogeneous video input requirements (different model APIs, different frame-count expectations) while enforcing a common training loop, optimisation configuration, and evaluation protocol. This framework enables researchers to add new architectures to the same controlled comparison with minimal modification.

**(b) Strictly Controlled Four-Paradigm Accuracy Comparison.** A rigorous accuracy comparison is performed across all four architectural paradigms on both UCF101 and HMDB51 benchmarks, with every confounding experimental variable held constant. The results provide the most directly comparable assessment of the representational power of these four paradigms currently available in the literature. The finding that all four architectures converge within 0.55 percentage points on UCF101 — a benchmark widely considered a standard evaluation target — is itself a significant scientific result, demonstrating that UCF101 is too saturated to discriminate among modern pretrained architectures and should be supplemented or replaced by HMDB51 or harder benchmarks.

**(c) Multi-Dimensional Pareto Frontier Characterisation.** Beyond accuracy, this study profiles all four models across five efficiency dimensions: parameter count, GFLOPs per clip, single-clip inference latency (milliseconds), throughput (clips per second), and peak GPU memory consumption (megabytes). This multi-dimensional characterisation produces a Pareto frontier analysis that identifies which architectures dominate others across the efficiency-accuracy trade-off space. A notable finding is that VideoMAE Pareto-dominates R(2+1)D-18 on the accuracy-versus-GFLOPs axis: despite having 55 million more parameters, VideoMAE is both more accurate and less computationally expensive per clip (135.17 versus 162.58 GFLOPs), a counter-intuitive result that challenges assumptions about the efficiency of volumetric convolution versus self-attention for video understanding.

**(d) Controlled Ablation of Recurrence versus Temporal Averaging.** By evaluating ResNet-50 TSN and ResNet-50+bidir-LSTM with identical backbones, pretraining, and all other hyperparameters, this study isolates the specific contribution of recurrent sequence modelling to action recognition accuracy. The quantitative finding — that adding a bidirectional LSTM to a TSN-style ResNet-50 yields only 0.32 percentage points improvement on UCF101 and 0.78 percentage points on HMDB51 at the cost of 10.4 million additional parameters and additional inference latency — provides a rigorous, controlled measurement of the marginal value of recurrence for this task, supplementing previous informal comparisons that could not disentangle recurrence from other architectural and training differences.

## 1.4 Thesis Organisation

The remainder of this dissertation is organised as follows.

**Chapter 2** defines the primary and secondary research objectives of the study, specifies the scope and explicit delimitations of the comparative evaluation, and formally defines the evaluation criteria used to assess model performance across both accuracy and efficiency dimensions.

**Chapter 3** presents a comprehensive review of the relevant literature, tracing the development of action recognition from hand-crafted spatiotemporal feature methods through the successive waves of CNN-based, RNN-based, 3D-CNN-based, and Transformer-based approaches. Special attention is paid to the theoretical motivations and empirical results of the four model families evaluated in the experimental study, as well as the benchmark datasets and evaluation conventions that have shaped the field.

**Chapter 4** provides a detailed technical description of the four architectures evaluated in this study: ResNet-50 TSN, ResNet-50+bidirectional LSTM, R(2+1)D-18, and VideoMAE (ViT-B). For each architecture, the theoretical motivation, mathematical formulation, parameter structure, computational complexity, and pretraining strategy are discussed. The chapter also specifies the unified data pipeline, preprocessing configuration, and training protocol applied to all models.

**Chapter 5** presents the full experimental results, organised first by dataset and then by evaluation dimension. Accuracy results on UCF101 and HMDB51 are reported and analysed in the context of pretraining differences, dataset difficulty, and architectural properties. Efficiency profiling results — parameters, GFLOPs, latency, throughput, and memory — are reported and visualised as Pareto frontiers. The controlled TSN-versus-CNN-LSTM ablation result is discussed in detail.

**Chapter 6** discusses the broader implications of the experimental findings, acknowledges the limitations of the study, and proposes directions for future work. Limitations discussed include the use of a non-standard 70/30 random split for HMDB51 (rather than the official three-fold splits), the use of only 10 training epochs (which may underfit larger models), and the restriction to two benchmarks and four specific architectures.

Appendix A provides the complete hyperparameter configurations and training curves for all eight model-dataset combinations. Appendix B documents the software environment, package versions, and reproduction instructions required to reproduce all experimental results.

---

# Chapter 2: Research Objectives and Scope

## 2.1 Primary and Secondary Objectives

The overarching aim of this research is to produce a rigorous, replicable, and practically informative comparison of four dominant deep learning paradigms for human action recognition from video, conducted under conditions that maximise the internal validity of the comparison. This overarching aim decomposes into the following primary and secondary research objectives.

**Primary Objective 1: Accuracy Benchmarking Under Controlled Conditions.**
To measure and compare the top-1 accuracy of ResNet-50 TSN, ResNet-50+bidir-LSTM, R(2+1)D-18, and VideoMAE on UCF101 and HMDB51, with all hyperparameters, training duration, data preprocessing, and inference protocols held strictly constant across all models. This objective directly addresses the research gap identified in Section 1.2: the absence of a controlled four-paradigm comparison in the literature. The hypothesis motivating this objective is that architecture-intrinsic differences in how temporal information is modelled will produce measurable accuracy differences on discriminative benchmarks even when all other factors are equalised, and that these differences will be larger on HMDB51 (which is more discriminative) than on UCF101 (which may be saturated by modern pretraining).

**Primary Objective 2: Multi-Dimensional Efficiency Profiling.**
To measure and compare the computational efficiency of all four models across five dimensions: total learnable parameter count, floating-point operations (GFLOPs) per input clip, single-clip inference latency (milliseconds) on the NVIDIA L4 GPU, inference throughput (clips per second), and peak GPU memory consumption (megabytes) during inference. This objective reflects the practical reality that accuracy alone is insufficient to guide architectural selection for deployment: a model that is marginally more accurate but requires an order of magnitude more computation or memory may be unsuitable for many real-world applications.

**Primary Objective 3: Pareto Frontier Characterisation.**
To synthesise the accuracy and efficiency results into a multi-dimensional Pareto analysis that identifies which architectures are not dominated by any other on the joint accuracy-efficiency space, and which trade-off profiles are available to practitioners making architectural selection decisions for specific deployment constraints. This objective moves beyond pairwise comparison to a holistic characterisation of the design space spanned by the four evaluated paradigms.

**Secondary Objective 1: Controlled Ablation of Recurrence.**
To quantify the specific contribution of bidirectional LSTM recurrence to action recognition accuracy by comparing ResNet-50 TSN and ResNet-50+bidir-LSTM, which share an identical ImageNet-pretrained backbone and all other hyperparameters. This secondary objective provides a controlled measurement of the marginal value of explicit sequential modelling relative to simple temporal averaging, disentangled from confounding variables such as backbone capacity and pretraining data.

**Secondary Objective 2: Pretraining Domain Analysis.**
To characterise the influence of pretraining domain — ImageNet-pretrained (TSN, CNN-LSTM) versus Kinetics-400-pretrained (R(2+1)D-18, VideoMAE) — on the accuracy gap between architectural families, particularly on the more discriminative HMDB51 dataset. Understanding the extent to which observed accuracy differences reflect architectural design versus pretraining data distribution is important for practitioners who may not have access to large-scale video pretraining data.

**Secondary Objective 3: Benchmark Saturation Analysis.**
To assess the extent to which UCF101 and HMDB51 remain discriminative benchmarks for distinguishing among state-of-the-art architectures under modern pretraining regimes. The narrow performance spread observed on UCF101 in the experimental results is analysed to determine whether the benchmark can meaningfully differentiate the four evaluated architectures, with implications for benchmark selection in future comparative studies.

**Secondary Objective 4: Reproducibility and Open Science.**
To produce a fully documented, open-source evaluation framework that enables the research community to reproduce all reported results and to extend the comparison to additional architectures, datasets, or training configurations. This objective reflects the author's commitment to reproducible science and to the community infrastructure that enables cumulative progress.

## 2.2 Scope and Delimitations

A scientific study is defined as much by what it does not attempt to answer as by what it does. The following explicit scope delimitations apply to this dissertation.

**Clip-level recognition, not temporal action localisation.** This study addresses the problem of **video clip classification**: given a short video clip (16 frames in this study), assign it a single action class label. This is distinct from the related but harder problems of **temporal action detection** (locating the start and end times of actions within an untrimmed video) and **spatial action localisation** (detecting the bounding boxes of actors while simultaneously recognising their actions). The clip-level recognition formulation is the dominant evaluation protocol on UCF101 and HMDB51 and is the most directly comparable across the four architectural families. Extending this study to temporal localisation tasks such as ActivityNet [29] or AVA [30] would require different model heads, different evaluation metrics, and different training data, and is left for future work.

**Offline inference, not online or streaming recognition.** All models are evaluated in an **offline** mode where the full 16-frame clip is available before inference begins. This corresponds to the standard academic evaluation protocol and is appropriate for applications such as surveillance clip analysis, post-hoc sports highlight tagging, and medical video indexing. It does not address the **online recognition** scenario, where actions must be classified from a causal stream of frames with low latency and without access to future context. Online recognition imposes additional architectural constraints (no future-frame attention, causal convolutions) that fall outside the scope of this study.

**Two benchmark datasets: UCF101 and HMDB51.** The study is restricted to UCF101 [31] and HMDB51 [32], two of the most widely used action recognition benchmarks with sufficient precedent in the literature to contextualise the reported results. Larger-scale benchmarks such as Kinetics-400 [20] or Something-Something [33] would provide additional discriminative power but would require substantially greater computational resources (hundreds of GPU-hours per model) that exceed the budget of this single-GPU Colab-based study. Activity recognition benchmarks focused on fine-grained sports actions, surgical procedures, or sign language are also outside scope.

**Four specific architectures, not an exhaustive survey.** The four evaluated models — ResNet-50 TSN, ResNet-50+bidir-LSTM, R(2+1)D-18, and VideoMAE — are selected as canonical representatives of their respective architectural paradigms rather than as an exhaustive enumeration of all architectures within those paradigms. Many important models are not evaluated: SlowFast [34], X3D [35], MViT [36], TimeSformer [25], Video Swin Transformer [26], among others. The choice of these four specific models is motivated by the desire to cover the four paradigms with representatives of comparable vintage and availability, and to enable the controlled backbone ablation (TSN versus CNN-LSTM) that requires an identical ResNet-50 backbone.

**Single-GPU, single-clip evaluation.** All training and inference is conducted on a single NVIDIA L4 GPU. Multi-GPU data-parallel or model-parallel training, which would be necessary to reproduce the original training protocols of VideoMAE at full scale, is not employed. Single-clip inference (no multi-clip or multi-crop ensembling) is used for all latency and throughput measurements, ensuring directly comparable latency figures across architectures. Multi-clip ensembling, which would boost accuracy for all models but would violate the controlled comparison requirement and the latency measurement protocol, is not used.

**No custom data augmentation or training tricks.** Beyond the standard ImageNet mean-standard-deviation normalisation, no architecture-specific data augmentation strategies (RandAugment, MixUp, CutMix, random erasing) are applied. This choice ensures that no model benefits from augmentation strategies to which it was specifically adapted during original development, and that any accuracy differences are attributable to the fundamental architectural and pretraining differences.

## 2.3 Evaluation Criteria

The evaluation of the four architectural paradigms proceeds along two primary axes: **predictive accuracy** and **computational efficiency**. The following metrics are formally defined and their practical significance explained.

### 2.3.1 Predictive Accuracy

**Top-1 Accuracy** is the primary accuracy metric used throughout this study. For a test set of N clips with ground-truth class labels y₁, ..., yₙ ∈ {1, ..., C}, and predicted class labels ŷ₁, ..., ŷₙ where ŷᵢ = argmax f(xᵢ) and f is the model, Top-1 Accuracy is defined as:

$$\text{Top-1 Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\hat{y}_i = y_i]$$

Top-1 Accuracy is the standard metric for UCF101 and HMDB51 evaluation and directly corresponds to the probability that a model correctly identifies the action in a randomly sampled test clip on the first attempt. It is the most commonly reported metric in the action recognition literature, making it the natural choice for contextualising results against prior work.

Top-5 Accuracy, which counts a prediction as correct if the true class appears among the top-5 predicted classes, is not reported in this study because the relatively small class counts of UCF101 (101 classes) and HMDB51 (51 classes) make Top-5 Accuracy less discriminative — all models achieve near-perfect Top-5 Accuracy on these benchmarks. Top-5 Accuracy is more meaningful on larger-scale benchmarks such as Kinetics-400 (400 classes).

### 2.3.2 Computational Efficiency Metrics

**Parameter Count (millions, M)** quantifies the total number of learnable scalar parameters in the model, including all convolutional weights, batch normalisation affine parameters, fully-connected weights, and biases. Parameter count is the most commonly cited proxy for model size and is directly relevant to storage requirements (a parameter stored in FP32 requires 4 bytes; in FP16, 2 bytes). It is measured using the `fvcore` library's `flop_count_operators` facility and verified against PyTorch's `sum(p.numel() for p in model.parameters())`. A smaller parameter count is desirable for deployment on memory-constrained edge devices and for faster model loading at inference time.

**GFLOPs per Clip (Giga Floating-Point Operations)** quantifies the computational work required to produce a single clip-level prediction, measured in units of 10⁹ multiply-accumulate (MAC) operations. GFLOPs is the standard measure of algorithmic computational complexity and is architecture-independent in the sense that it reflects the mathematical work required regardless of the specific hardware executing it. It is measured using `fvcore`'s `FlopCountAnalysis` facility with a single clip of the appropriate shape as input. One important caveat applies: `fvcore` may undercount operations for recurrent layers (specifically LSTM operations), as it does not natively profile all LSTM computations; for the CNN-LSTM model, this means the reported GFLOPs figure of 65.75 GFLOPs reflects the ResNet-50 backbone computation and underestimates the total by approximately 0.3 GFLOPs. A lower GFLOPs figure indicates lower computational cost per inference, which is relevant for throughput on fixed hardware and for energy consumption in large-scale deployment.

**Inference Latency (milliseconds, ms)** measures the wall-clock time elapsed from the moment a single pre-processed clip tensor is passed to the model until the class-probability vector is returned, averaged over 100 repeated measurements with CUDA synchronisation before each measurement. Latency is the operationally critical metric for real-time applications: a surveillance system that must respond to a detected event within one second, or an HCI system that must respond to a user gesture without perceptible delay, requires inference latency well below the tolerance threshold. Latency is reported as the median of 100 measurements to mitigate outlier effects from GPU scheduling. All measurements are taken with the model in evaluation mode (BatchNorm running statistics frozen, dropout disabled) and with FP16 input tensors on the NVIDIA L4 GPU.

**Throughput (clips per second, clips/s)** is the reciprocal of latency and measures the maximum rate at which the model can process independent clips on the evaluation GPU. Throughput determines the scalability of a model to high-volume inference workloads — for example, a video platform processing millions of uploaded clips per day requires sufficient throughput to remain within manageable infrastructure costs. While latency and throughput are mathematically related for single-clip inference, they can diverge under batched inference with different batch sizes; in this study both are reported for single-clip inference.

**Peak GPU Memory (megabytes, MB)** measures the maximum GPU VRAM consumed during a single forward pass over one clip, measured using `torch.cuda.max_memory_allocated()` after a warm-up pass and cleared between models. Peak memory is critical for two reasons. First, it determines the minimum GPU specification required to run the model at all: a model requiring 8 GB peak memory cannot run on a 6 GB consumer GPU. Second, in a batched inference setting, peak memory per clip scales approximately linearly with batch size, so the per-clip memory footprint determines the maximum batch size achievable on a fixed GPU, which in turn determines throughput. Models with lower peak memory are preferred for deployment on resource-constrained hardware including mobile GPUs, embedded systems, and consumer gaming cards.

### 2.3.3 Why Multi-Dimensional Evaluation Matters

The practical significance of multi-dimensional evaluation becomes apparent when considering the specific trade-offs revealed by the experimental results of this study. VideoMAE achieves the highest accuracy on both benchmarks but also requires the largest parameter count (86.28 million parameters) and the highest inference latency (51.85 ms per clip). ResNet-50 TSN achieves the lowest accuracy but also the smallest parameter count (23.71 million), the lowest GFLOPs (65.75), the lowest latency (18.39 ms), and the lowest peak memory (281 MB). These observations define a genuine trade-off: no single model simultaneously dominates all others on all dimensions.

A practitioner deploying an action recognition system must navigate this trade-off explicitly, informed by the specific constraints and requirements of their application. An autonomous vehicle system requiring real-time action recognition at 30 frames per second demands latency below approximately 33 ms per clip — a constraint that VideoMAE (51.85 ms) violates while ResNet-50 TSN (18.39 ms) satisfies comfortably. A cloud-based video analytics platform with no strict latency requirement but very high throughput demands (millions of clips per day) might prioritise GFLOPs efficiency over latency. A mobile health monitoring application running on a device with a 4 GB unified memory architecture must prioritise peak memory above all else. Without multi-dimensional profiling, practitioners cannot make these trade-offs explicitly — they must rely on incomplete information or make assumptions that may not hold in their specific deployment context.

The Pareto frontier framework provides a principled tool for navigating these trade-offs. A model is **Pareto-optimal** (or non-dominated) on a given pair of dimensions if no other evaluated model simultaneously achieves both higher accuracy and lower cost on those dimensions. Identifying the Pareto frontier across multiple accuracy-efficiency pairs allows practitioners to focus on only those architectures that offer genuinely distinct trade-off profiles, eliminating dominated options from consideration. This characterisation is a primary analytical contribution of the experimental work in Chapter 5.

---

## References

[1] Cisco Systems, "Cisco Visual Networking Index: Forecast and trends, 2017–2022," White Paper, Feb. 2019.

[2] R. Gade and T. B. Moeslund, "Thermal cameras and applications: A survey," *Machine Vision and Applications*, vol. 25, no. 1, pp. 245–262, 2014.

[3] R. Poppe, "A survey on vision-based human action recognition," *Image and Vision Computing*, vol. 28, no. 6, pp. 976–990, Jun. 2010.

[4] L. Chen, H. Wei, and J. Ferryman, "A survey of human motion analysis using depth imagery," *Pattern Recognition Letters*, vol. 34, no. 15, pp. 1995–2006, Nov. 2013.

[5] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, "Vision meets robotics: The KITTI dataset," *International Journal of Robotics Research*, vol. 32, no. 11, pp. 1231–1237, Sep. 2013.

[6] A. Gupta and L. S. Davis, "Objects in action: An approach for combining action understanding and object perception," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Minneapolis, MN, USA, pp. 1–8, 2007.

[7] N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, San Diego, CA, USA, vol. 1, pp. 886–893, 2005.

[8] I. Laptev, "On space-time interest points," *International Journal of Computer Vision*, vol. 64, no. 2–3, pp. 107–123, Sep. 2005.

[9] H. Wang, A. Kläser, C. Schmid, and C.-L. Liu, "Dense trajectories and motion boundary descriptors for action recognition," *International Journal of Computer Vision*, vol. 103, no. 1, pp. 60–79, May 2013.

[10] H. Wang and C. Schmid, "Action recognition with improved trajectories," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, Sydney, NSW, Australia, pp. 3551–3558, 2013.

[11] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 25, pp. 1097–1105, 2012.

[12] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei, "ImageNet large scale visual recognition challenge," *International Journal of Computer Vision*, vol. 115, no. 3, pp. 211–252, Dec. 2015.

[13] A. Karpathy, G. Toderici, S. Shetty, T. Leung, R. Sukthankar, and L. Fei-Fei, "Large-scale video classification with convolutional neural networks," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Columbus, OH, USA, pp. 1725–1732, 2014.

[14] K. Simonyan and A. Zisserman, "Two-stream convolutional networks for action recognition in videos," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 27, pp. 568–576, 2014.

[15] L. Wang, Y. Xiong, Z. Wang, Y. Qiao, D. Lin, X. Tang, and L. Van Gool, "Temporal segment networks: Towards good practices for deep action recognition," in *Proc. European Conference on Computer Vision (ECCV)*, Amsterdam, Netherlands, pp. 20–36, 2016.

[16] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, Nov. 1997.

[17] J. Donahue, L. Anne Hendricks, S. Guadarrama, M. Rohrbach, S. Venugopalan, K. Saenko, and T. Darrell, "Long-term recurrent convolutional networks for visual recognition and description," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Boston, MA, USA, pp. 2625–2634, 2015.

[18] S. Ji, W. Xu, M. Yang, and K. Yu, "3D convolutional neural networks for human action recognition," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 35, no. 1, pp. 221–231, Jan. 2013.

[19] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, "Learning spatiotemporal features with 3D convolutional networks," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, Santiago, Chile, pp. 4489–4497, 2015.

[20] W. Kay, J. Carreira, K. Simonyan, B. Zhang, C. Hillier, S. Vijayanarasimhan, F. Viola, T. Green, T. Back, P. Natsev, M. Suleyman, and A. Zisserman, "The Kinetics human action video dataset," *arXiv preprint arXiv:1705.06950*, 2017.

[21] J. Carreira and A. Zisserman, "Quo vadis, action recognition? A new model and the Kinetics dataset," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Honolulu, HI, USA, pp. 6299–6308, 2017.

[22] D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri, "A closer look at spatiotemporal convolutions for video understanding," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Salt Lake City, UT, USA, pp. 6450–6459, 2018.

[23] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 5998–6008, 2017.

[24] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16×16 words: Transformers for image recognition at scale," in *Proc. International Conference on Learning Representations (ICLR)*, 2021.

[25] G. Bertasius, H. Wang, and L. Torresani, "Is space-time attention all you need for video understanding?" in *Proc. International Conference on Machine Learning (ICML)*, pp. 813–824, 2021.

[26] Z. Liu, J. Ning, Y. Cao, Y. Wei, Z. Zhang, S. Lin, and H. Hu, "Video Swin Transformer," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, New Orleans, LA, USA, pp. 3202–3211, 2022.

[27] Z. Tong, Y. Song, J. Wang, and L. Wang, "VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, pp. 10078–10093, 2022.

[28] Z. Kong and W. Jing, "A comprehensive study of deep video action recognition," *arXiv preprint arXiv:2012.06567*, 2022.

[29] F. Caba Heilbron, V. Escorcia, B. Ghanem, and J. C. Niebles, "ActivityNet: A large-scale video benchmark for human activity understanding," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Boston, MA, USA, pp. 961–970, 2015.

[30] C. Gu, C. Sun, D. A. Ross, C. Vondrick, C. Pantofaru, Y. Li, S. Vijayanarasimhan, G. Toderici, S. Ricco, R. Sukthankar, C. Schmid, and J. Malik, "AVA: A video dataset of spatio-temporally localized atomic visual actions," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Salt Lake City, UT, USA, pp. 6047–6056, 2018.

[31] K. Soomro, A. Roshan Zamir, and M. Shah, "UCF101: A dataset of 101 human actions classes from videos in the wild," *arXiv preprint arXiv:1212.0402*, 2012.

[32] H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre, "HMDB: A large video database for human motion recognition," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, Barcelona, Spain, pp. 2556–2563, 2011.

[33] R. Goyal, S. Ebrahimi Kahou, V. Michalski, J. Materzynska, S. Westphal, H. Kim, V. Haenel, I. Fruend, P. Yianilos, M. Mueller-Freitag, F. Hoppe, C. Thurau, I. Bax, and R. Memisevic, "The 'something something' video database for learning and evaluating visual common sense," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, Venice, Italy, pp. 5842–5850, 2017.

[34] C. Feichtenhofer, H. Fan, J. Malik, and K. He, "SlowFast networks for video recognition," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, Seoul, South Korea, pp. 6202–6211, 2019.

[35] C. Feichtenhofer, "X3D: Expanding architectures for efficient video recognition," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Seattle, WA, USA, pp. 203–213, 2020.

[36] H. Fan, B. Xiong, K. Mangalam, Y. Li, Z. Yan, J. Malik, and C. Feichtenhofer, "Multiscale vision transformers," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, Montreal, QC, Canada, pp. 6824–6835, 2021.

[37] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. International Conference on Learning Representations (ICLR)*, New Orleans, LA, USA, 2019.

[38] P. Micikevicius, S. Narang, J. Alben, G. Diamos, E. Elsen, D. Garcia, B. Ginsburg, M. Houston, O. Kuchaiev, G. Venkatesh, and H. Wu, "Mixed precision training," in *Proc. International Conference on Learning Representations (ICLR)*, Vancouver, BC, Canada, 2018.


---

# Chapter 3: Literature Review and State of the Art

## 3.1 Early Approaches to Action Recognition

The problem of human action recognition from video has occupied computer vision researchers for several decades. Prior to the deep learning era, the dominant paradigm relied on hand-crafted feature descriptors designed to capture motion patterns, appearance statistics, and spatiotemporal structure from raw pixel data. These methods, while elegant in their motivation, ultimately struggled to generalise across the full diversity of real-world action classes and recording conditions.

### 3.1.1 Local Spatio-Temporal Descriptors

One of the foundational contributions to action recognition came from the extension of image-domain histogram descriptors to the temporal domain. Dalal and Triggs introduced the Histogram of Oriented Gradients (HOG) [1] for pedestrian detection in still images, demonstrating that dense, locally-normalised gradient orientation histograms could robustly encode shape and appearance information. The extension of this idea to video was straightforward: by computing gradient orientations across both spatial and temporal dimensions, Histogram of Optical Flow (HOF) descriptors could encode local motion patterns as a complement to appearance. Motion Boundary Histograms (MBH), introduced by Dalal et al. [2], went further by computing gradient histograms of the optical flow field itself rather than its raw values, yielding a representation that suppressed background motion while emphasising relative body motion â€” a critical advantage for recognition in cluttered, non-static scenes.

The most influential hand-crafted approach of the pre-deep-learning era was the Dense Trajectories framework introduced by Wang et al. [3]. Rather than extracting features at interest points, Dense Trajectories sampled feature points on a dense grid and tracked them through the video using optical flow. For each trajectory, multiple descriptors were computed along the path: HOG, HOF, and MBH computed within a spatio-temporal volume aligned to the trajectory. This combination produced a rich, complementary description of both the static appearance and the dynamic motion along each track. The Improved Dense Trajectories (iDT) of Wang and Schmid [4] addressed a key weakness of the original approach: the conflation of camera motion with body motion. By estimating the homography of background motion and subtracting it from the optical flow field before tracking, iDT effectively factored out camera shake and pan, leading to substantial improvements on every benchmark of the time. At the time of their introduction, iDT features combined with Fisher Vector encoding represented the state of the art on UCF101 and HMDB51 [4], and they remained competitive well into the deep learning era.

### 3.1.2 Interest Point Detectors and Global Encodings

Laptev [5] introduced Space-Time Interest Points (STIPs), extending the Harris corner detector to the spatiotemporal domain by seeking points where the image signal varied significantly along both spatial and temporal axes. STIPs provided a sparse, computationally efficient alternative to dense sampling, though they were sensitive to scale selection and struggled with periodic, low-contrast motions such as walking at a distance. Dollar et al. [6] proposed the Cuboid detector as an alternative, applying separable linear filters in space and time to identify regions of high spatio-temporal energy.

Following the success of Bag-of-Visual-Words (BoVW) models in image retrieval and classification [7], action recognition researchers adopted a similar pipeline: local descriptors were quantised against a codebook of visual words learned by k-means clustering, and each video was represented by a histogram of codebook assignments. While straightforward to implement, hard assignment to a single nearest centroid discarded much of the information in the descriptor. Fisher Vectors [8], derived from the Fisher Information Matrix of a Gaussian Mixture Model fitted to the descriptor space, provided a principled improvement by encoding both first- and second-order statistics of the descriptor population with respect to the mixture components. Fisher Vector encoding consistently outperformed BoVW at equivalent codebook sizes and became the de-facto encoding strategy for iDT [4].

### 3.1.3 Limitations Motivating Deep Learning

Despite their ingenuity, hand-crafted approaches suffered from several fundamental limitations that ultimately motivated the transition to learned representations. First, the multi-stage pipeline â€” detection, description, encoding, classification â€” was brittle, with errors propagating through each stage. Second, optical flow computation was expensive, requiring per-video processing with iterative solvers such as Farneback [9] or TVL1 [10]. Third, the representational capacity of fixed-vocabulary codebooks was inherently limited; rare or compositional actions that did not decompose into standard motion primitives were poorly captured. Fourth, and most critically, the features were entirely handcrafted based on human intuition about what motion patterns matter; there was no mechanism to adapt the representation to the specific statistical regularities of the training data. These limitations set the stage for the deep learning revolution in visual recognition, which began in the image domain before being extended to video.

---

## 3.2 Convolutional Neural Networks for Image Classification

The development of deep CNNs for image classification is among the most consequential intellectual histories in modern computer science, not only because of its intrinsic importance but because every major video recognition architecture discussed in this thesis is directly descended from the image CNN lineage.

### 3.2.1 Foundations: LeNet to AlexNet

LeCun et al. [11] demonstrated in 1998 that convolutional networks trained end-to-end with backpropagation could outperform hand-engineered systems on handwritten digit recognition. LeNet-5, with its alternating convolution and pooling layers and fully-connected classifier, established the architectural blueprint that would define the field for two decades. However, scaling LeNet to natural image datasets was impeded by the computational cost of convolution, the absence of large annotated datasets, and the difficulty of training deep networks with sigmoid activations due to vanishing gradients.

The breakthrough came in 2012 when Krizhevsky, Sutskever, and Hinton introduced AlexNet [12], winning the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) [13] with a top-5 error of 15.3% â€” more than ten percentage points below the second-place entry. AlexNet's success rested on several concurrent advances: GPU-accelerated training on two GTX 580s enabled a network of unprecedented size; Rectified Linear Units (ReLU) [14] replaced saturating sigmoid activations, dramatically improving training speed; local response normalisation and data augmentation addressed overfitting; and dropout regularisation [15] prevented co-adaptation of neurons. The scale of AlexNet's improvement was sufficiently dramatic to redirect the entire field of computer vision toward learned deep features.

### 3.2.2 Depth and Architecture Refinements

Simonyan and Zisserman's VGGNet [16] established that network depth, rather than large receptive fields, was the key driver of performance. By replacing the large (7Ã—7, 11Ã—11) filters of AlexNet with stacks of 3Ã—3 convolutions â€” which have the same effective receptive field at lower parameter count and more non-linearities â€” VGGNet achieved top-5 error of 7.3% on ILSVRC 2014 with its 16- and 19-layer variants. More importantly, VGGNet demonstrated systematic scalability: deeper networks consistently outperformed shallower ones, provided training could proceed. The simplicity and regularity of VGGNet's design made it the preferred backbone for feature extraction in many subsequent video recognition systems, including the Two-Stream Networks discussed in Section 3.3.

Szegedy et al.'s GoogLeNet/Inception architecture [17] pursued a complementary approach: rather than depth, it optimised computational efficiency through the Inception module, which applied convolutions of multiple kernel sizes (1Ã—1, 3Ã—3, 5Ã—5) in parallel and concatenated their feature maps. 1Ã—1 convolutions served as bottlenecks to reduce channel dimensionality before expensive 3Ã—3 and 5Ã—5 convolutions, dramatically reducing the parameter count relative to AlexNet and VGGNet while achieving lower error. Ioffe and Szegedy [18] subsequently introduced Batch Normalisation (BN), which normalised the inputs to each layer to zero mean and unit variance across the mini-batch. BN addressed the internal covariate shift problem, allowing much higher learning rates, reducing sensitivity to weight initialisation, and providing a mild regularisation effect. Batch normalisation became a near-universal component of all subsequent architectures and was critical to the training of the deep 3D CNNs discussed in Section 3.5.

### 3.2.3 Residual Networks and the Backbone of This Thesis

The most significant architectural contribution of the mid-2010s was the Residual Network (ResNet) of He et al. [16], which won ILSVRC 2015 with a top-5 error of 3.57% using a 152-layer network. The central observation of ResNet was that increasing network depth beyond a certain point caused training accuracy â€” not just test accuracy â€” to degrade, indicating an optimisation failure rather than overfitting. He et al. proposed the residual (shortcut) connection: instead of learning a mapping H(x) directly, a block learns a residual F(x) = H(x) âˆ’ x, such that the identity mapping is trivially representable as F(x) = 0. This reformulation dramatically improved gradient flow through very deep networks by providing a highway for gradients to bypass multiple non-linear layers during backpropagation. ResNet-50, the 50-layer variant using bottleneck blocks, achieves an excellent balance between representational capacity and computational cost, and is employed in this thesis as the 2D-CNN backbone for the Temporal Segment Network baseline (Section 3.3.2). Its pre-trained ImageNet weights serve as initialisation, leveraging the transfer learning paradigm discussed below.

### 3.2.4 Transfer Learning and Fine-Tuning

Transfer learning [19] â€” the practice of initialising a model with weights pre-trained on a large source dataset and fine-tuning on a smaller target dataset â€” became a cornerstone of applied deep learning for visual recognition. Yosinski et al. [20] provided empirical evidence that early convolutional layers learn generic, transferable features (edges, textures, blobs) while later layers capture task-specific semantics. Fine-tuning pre-trained ImageNet models consistently outperformed training from scratch on small video datasets such as UCF101 and HMDB51, where insufficient data would otherwise cause severe overfitting. This paradigm is exploited extensively in this thesis: ResNet-50, R(2+1)D, and VideoMAE all leverage large-dataset pretraining before fine-tuning on the target action recognition benchmarks.

---

## 3.3 2D-CNN Approaches for Video Understanding

The ImageNet revolution raised an immediate question for the video recognition community: how should spatial CNNs, which lack an explicit temporal dimension, be adapted to understand motion and temporal dynamics? The first wave of deep video methods largely treated video as a collection of frames or frame-level features and applied various temporal aggregation mechanisms.

### 3.3.1 Two-Stream Networks

Simonyan and Zisserman [21] proposed the Two-Stream Network architecture in 2014, which became the dominant video recognition paradigm for several years. The key insight was to decouple spatial appearance modelling from motion modelling by operating two independent CNN streams: a spatial stream processing individual RGB frames, and a temporal stream processing a stack of pre-computed optical flow frames. Each stream was a deep CNN (based on VGGNet) trained independently; at test time, their softmax scores were fused by averaging or an SVM. This architectural decision was motivated by the observation that optical flow explicitly encodes inter-frame displacement without requiring the network to infer motion implicitly from appearance changes â€” a task that is particularly difficult for CNNs with limited receptive fields in the temporal axis. Two-Stream Networks achieved 88.0% on UCF101 and 59.4% on HMDB51 [21], competitive with iDT at the time, demonstrating that deep CNNs could rival hand-crafted approaches even without explicit temporal modelling.

However, the Two-Stream framework had significant practical limitations. Optical flow pre-computation was expensive both in time and storage, and the two streams were trained in isolation rather than jointly, preventing the learning of cross-modal representations. Furthermore, the temporal stream processed only a small, fixed window of flow frames (typically 10), limiting the temporal context available to the model.

### 3.3.2 Temporal Segment Networks

Wang et al. [22] addressed the limited temporal coverage of Two-Stream Networks with the Temporal Segment Network (TSN), which introduced sparse temporal sampling as a principled strategy for capturing long-range temporal dynamics without the prohibitive memory cost of processing all frames. In TSN, each video is divided into K equally-spaced temporal segments, and a single frame (or short optical flow stack) is randomly sampled from each segment during training. The predictions from all K snippets are aggregated via a consensus function â€” typically averaging the segment-level softmax scores. This design allows the network to cover the entire video duration regardless of length, while remaining computationally tractable through sparse sampling. TSN achieved 94.2% on UCF101 and 69.4% on HMDB51 [22] with a BN-Inception backbone and combined RGB+optical flow streams, substantially advancing the state of the art. The TSN paradigm forms the basis of the 2D-CNN baseline employed in this thesis, using ResNet-50 as the backbone and RGB-only input to isolate the contribution of the spatial stream without the confound of optical flow.

### 3.3.3 ActionVLAD and Temporal Relation Networks

Girdhar et al. [23] proposed ActionVLAD, extending the image-level VLAD (Vector of Locally Aggregated Descriptors) encoding to video by aggregating CNN features across time using a learned attention mechanism. Rather than simple temporal averaging, ActionVLAD assigned local features to a set of learned cluster centres, accumulating residuals within each cluster, producing a fixed-length video-level representation that captured the distribution of local action primitives throughout the clip.

Zhou et al. [24] introduced the Temporal Relation Network (TRN), which explicitly modelled temporal ordering and relations between frames sampled at multiple temporal scales. By applying a small MLP to pairs and tuples of frame-level features, TRN could learn what temporal orderings were discriminative for a given class, capturing ordering-dependent semantics that averaging-based methods missed. TRN demonstrated strong performance on the temporally demanding Something-Something dataset, where the order in which motions occur is crucial to distinguishing classes.

### 3.3.4 Limitations of 2D-CNN Approaches

Despite their success, 2D-CNN approaches for video share a fundamental limitation: they process individual frames independently and aggregate temporal information only through late fusion or temporal attention, without learning spatiotemporal features jointly. Fine-grained motion patterns â€” the subtle interplay of pose and velocity within a short clip â€” are not directly representable by a 2D convolution operating on a single frame. Optical flow can partially compensate for this by providing explicit motion as additional input, but introduces computational overhead and represents a fixed, hand-engineered feature rather than an end-to-end learned one. These limitations motivated the development of three-dimensional convolutional architectures that directly extend convolutions into the temporal domain.

---

## 3.4 Recurrent Neural Networks for Sequence Modelling

Parallel to the CNN-based approaches, a distinct line of work treated action recognition as a sequence modelling problem: given a temporally ordered sequence of frame-level visual features, learn to classify the sequence. This framing naturally called for Recurrent Neural Networks (RNNs), which had been developed for language modelling and time-series analysis.

### 3.4.1 Vanilla RNNs and the Vanishing Gradient Problem

Standard RNNs maintain a hidden state that is updated at each timestep by combining the current input with the previous hidden state through a learned weight matrix. In principle, this hidden state provides a memory of all past inputs, enabling long-range temporal dependencies to be modelled. In practice, however, Bengio et al. [25] demonstrated that vanilla RNNs suffer from the vanishing (and exploding) gradient problem during backpropagation through time (BPTT): gradients of the loss with respect to early timesteps decay exponentially with the number of unrolled timesteps, making it effectively impossible to learn dependencies spanning more than a few dozen steps. This fundamental limitation severely restricted the practical temporal horizon of RNN-based sequence models.

### 3.4.2 Long Short-Term Memory Networks

Hochreiter and Schmidhuber [26] addressed the vanishing gradient problem with the Long Short-Term Memory (LSTM) architecture, which introduced a cell state â€” a separate memory channel that could carry information across many timesteps without multiplicative degradation. Three learned gating mechanisms regulate information flow: the input gate controls how much new information is written to the cell state; the forget gate controls how much existing cell state is retained; and the output gate controls how much of the cell state is exposed as the hidden state output. Because the cell state is modified by addition rather than multiplication through the gates, gradients can propagate over hundreds of timesteps without vanishing, enabling the learning of long-range dependencies that are inaccessible to vanilla RNNs. The forget gate, introduced by Gers et al. [27] as a refinement of the original LSTM, was particularly important for performance on variable-length sequences. LSTMs became the standard sequence model in natural language processing and were quickly adopted for video recognition.

Schuster and Paliwal [28] introduced Bidirectional RNNs (and by extension, Bidirectional LSTMs), which process sequences in both forward and backward temporal directions, concatenating the hidden states from both passes at each timestep. For tasks where the full sequence is available at inference time â€” as in offline action recognition â€” bidirectional processing allows each timestep's representation to incorporate both past and future context, improving classification accuracy at the cost of requiring complete sequences rather than online, streaming prediction.

Cho et al. [29] proposed the Gated Recurrent Unit (GRU) as a simplified variant of the LSTM that merges the cell state and hidden state into a single vector and reduces the three LSTM gates to two (update and reset gates). GRUs have fewer parameters than LSTMs and are faster to train while achieving comparable performance on many sequence modelling tasks, making them a popular choice when computational efficiency is a concern.

### 3.4.3 LRCN: Long-term Recurrent Convolutional Networks

Donahue et al. [30] proposed the Long-term Recurrent Convolutional Network (LRCN), which combined deep CNNs for per-frame visual feature extraction with LSTMs for temporal sequence modelling. In the LRCN framework, each video frame is passed through a CNN (typically AlexNet or VGGNet) to produce a fixed-length feature vector; the sequence of these feature vectors is then processed by one or more LSTM layers whose final hidden state (or a temporal aggregation thereof) is passed to a softmax classifier. This architecture was applied to multiple video understanding tasks including activity recognition, image captioning, and video description, demonstrating the generality of the CNN+LSTM pipeline. LRCN achieved competitive results on HMDB51 and UCF101 while providing an interpretable decomposition of spatial appearance modelling (CNN) and temporal dynamics modelling (LSTM). The LRCN paradigm serves as the RNN baseline in this thesis, using a pre-trained ResNet-50 as the frame encoder and a stacked LSTM for temporal sequence classification.

### 3.4.4 Limitations of RNN-Based Video Models

Despite their theoretical appeal for sequence modelling, RNN-based approaches for video recognition carry significant practical limitations. First, the sequential nature of RNN computation prevents parallelism across timesteps during training â€” each step must wait for the previous hidden state â€” resulting in substantially longer training times than feed-forward architectures of comparable capacity. Second, while LSTMs mitigate the vanishing gradient problem relative to vanilla RNNs, they still struggle with very long-range dependencies spanning hundreds of frames. Third, the combination of a CNN feature extractor and an LSTM creates a two-stage pipeline whose components are often not jointly optimised with equal effectiveness, and the CNN features may not be optimised for the temporal modelling task downstream. Fourth, the sequential bottleneck of hidden states imposes a fixed-capacity information bottleneck that may be insufficient to represent the full complexity of long videos. These limitations motivated both the development of 3D convolutional architectures (Section 3.5) and, ultimately, the attention-based Transformer models (Section 3.6) that overcome sequential computation constraints.

---

## 3.5 Three-Dimensional Convolutional Neural Networks for Video

The most conceptually direct extension of image CNNs to video is to extend the two-dimensional convolution kernel to three dimensions, adding a temporal axis. This approach, known as 3D CNN, jointly captures spatial appearance and temporal motion within the convolutional operation itself, without requiring separate optical flow estimation or sequential processing.

### 3.5.1 Early 3D CNNs

Ji et al. [31] and Baccouche et al. [32] independently explored 3D convolutions for action recognition in the early 2010s. Ji et al. applied 3D convolutions to sequences of video frames for human action recognition in surveillance video, demonstrating the feasibility of spatiotemporal feature learning. Baccouche et al. combined 3D convolutions with an LSTM for temporal aggregation. However, these early systems operated on small, low-resolution inputs and relied on relatively shallow architectures, limiting their representational capacity. The lack of large-scale video datasets for pre-training â€” analogous to ImageNet for image classification â€” also constrained the achievable performance.

### 3.5.2 C3D

Tran et al. [33] provided the first convincing demonstration of the power of deep 3D CNNs with C3D (Convolutional 3D), a VGG-like architecture using 3Ã—3Ã—3 convolution kernels throughout all layers, pre-trained on the Sports-1M dataset. The choice of uniform 3Ã—3Ã—3 kernels was motivated by empirical comparisons showing that this kernel size outperformed alternatives with asymmetric temporal extents. C3D achieved competitive accuracy on UCF101 and became the standard 3D CNN baseline due to its simplicity and the public availability of pre-trained weights. However, C3D suffered from two notable limitations: its large parameter count (~78M parameters) made both training and inference expensive, and the short 16-frame input clip meant that only local temporal context was captured. Moreover, pre-training on Sports-1M, a relatively noisy and sports-dominated dataset, limited the generalisability of the learned features.

### 3.5.3 I3D: Inflated 3D ConvNets

The most influential 3D CNN architecture of the deep learning era was the Two-Stream Inflated 3D ConvNet (I3D) proposed by Carreira and Zisserman [34], which introduced two key innovations. First, I3D proposed "inflating" a pre-trained 2D CNN (specifically Inception-v1) into 3D by repeating the 2D convolutional filters N times along the temporal axis and dividing by N to preserve activation magnitudes. This inflation procedure allowed the full weight initialisation advantage of ImageNet pre-training to be transferred to the 3D architecture, bypassing the need to train from scratch on video data alone. Second, in parallel with I3D, Carreira and Zisserman introduced the Kinetics-400 dataset [34] â€” a large-scale video dataset of 400 human action classes with approximately 240,000 training clips sourced from YouTube. Kinetics rapidly superseded Sports-1M as the standard pre-training dataset for video recognition, and pre-training on Kinetics followed by fine-tuning on UCF101/HMDB51 became the canonical transfer learning pipeline. I3D with two-stream (RGB + optical flow) pre-trained on Kinetics achieved 98.0% on UCF101 and 80.9% on HMDB51 [34], dramatically advancing the state of the art and establishing a high bar that subsequent methods were benchmarked against.

### 3.5.4 Pseudo-3D and Factorised Approaches

The computational cost of full 3D convolutions motivated several factorised approximations. Qiu et al. [35] proposed the Pseudo-3D (P3D) network, which decomposed a 3Ã—3Ã—3 3D convolution into a 2D spatial convolution (3Ã—3Ã—1) followed by a 1D temporal convolution (1Ã—1Ã—3) in series or in parallel. This decomposition reduced the parameter count and FLOPs while maintaining comparable accuracy, and the residual connections of the ResNet backbone could be adapted to accommodate the branched structure.

### 3.5.5 R(2+1)D: Factorised Spatiotemporal Convolutions

Tran et al. [36] conducted a systematic empirical study of spatiotemporal convolution factorisation strategies and proposed R(2+1)D as the optimal factorisation: a full 3D convolution is decomposed into a 2D spatial convolution followed by a 1D temporal convolution, applied sequentially. Crucially, Tran et al. demonstrated that this factorisation is strictly more expressive than a full 3D convolution of the same parameter budget, because the intermediate rectification between the 2D and 1D operations doubles the number of non-linearities per block. Furthermore, the decomposition improves gradient flow during training by separating spatial and temporal optimisation signals, leading to faster convergence and slightly better final accuracy than equivalent full-3D architectures. R(2+1)D pre-trained on Kinetics achieved 96.8% on UCF101 and 74.5% on HMDB51 [36]. The R(2+1)D-18 architecture (18-layer ResNet-style backbone with (2+1)D convolutions) is employed in this thesis as the 3D-CNN model, directly assessing the effect of explicit spatiotemporal feature learning relative to the 2D baseline and the Transformer baseline.

### 3.5.6 SlowFast Networks and X3D

Feichtenhofer et al. [37] proposed SlowFast Networks, which processed video through two parallel pathways operating at different temporal resolutions: a Slow pathway sampling frames at low frame rates but with high spatial resolution and large channel capacity, and a Fast pathway sampling at high frame rates but with low spatial resolution and few channels. The two pathways were connected by lateral connections allowing the fast pathway's motion information to inform the slow pathway's semantic representation. SlowFast achieved state-of-the-art results on Kinetics and AVA, demonstrating that dual-rate temporal modelling provides complementary information that neither pathway captures alone. Feichtenhofer [38] subsequently proposed X3D, a family of networks obtained by progressively expanding a minimal 2D image architecture along multiple dimensions (temporal duration, frame rate, spatial resolution, width, depth) guided by an efficiency-accuracy objective. X3D variants achieved competitive accuracy at significantly lower computational cost than SlowFast or I3D, establishing a new Pareto frontier for video recognition efficiency.

### 3.5.7 Limitations of 3D CNNs

Despite their strong performance, 3D CNN architectures have inherent limitations. The computational complexity of 3D convolution scales cubically with the spatiotemporal kernel extent, making very large receptive fields expensive. Training from scratch requires enormous amounts of video data due to the increased parameter count, and even with Kinetics pre-training, fine-tuning on small datasets risks overfitting. Furthermore, the fixed, local nature of convolutional receptive fields means that long-range temporal dependencies â€” actions defined by events separated by many seconds â€” are only indirectly captured through many stacked layers, which may not be optimal. These limitations motivated the application of global attention mechanisms, originally developed for natural language processing, to the video domain.

---

## 3.6 Attention Mechanisms and Transformers

The introduction of the Transformer architecture [39] and its subsequent extension to visual domains represents perhaps the most significant architectural shift in deep learning since the advent of residual connections. Transformer-based models have progressively supplanted or augmented CNN-based approaches across virtually every computer vision task, including action recognition.

### 3.6.1 Attention Mechanisms in Sequence Models

Bahdanau et al. [40] introduced soft attention for neural machine translation, allowing the decoder to attend selectively to different positions of the encoder's hidden state sequence when generating each output token. Rather than compressing the entire input into a fixed-length context vector, the attention mechanism computed a weighted sum of encoder states, where the weights were learned as a function of the current decoder state and each encoder state. This mechanism dramatically improved machine translation quality on long sentences and established attention as a general-purpose tool for selectively routing information in neural networks.

### 3.6.2 The Transformer Architecture

Vaswani et al. [39] proposed the Transformer, which eliminated recurrence and convolution entirely in favour of self-attention as the primary computational primitive. In self-attention, each position in the sequence computes a query, key, and value vector; attention weights are computed as the scaled dot-product of each query with all keys, and the output is a weighted sum of the values. Multi-Head Attention applies this operation in parallel across multiple learned subspaces and concatenates the results, allowing the model to attend to different aspects of the input simultaneously. Critically, self-attention has a constant path length between any two positions in the sequence â€” unlike RNNs where path length is proportional to sequence distance â€” providing a direct gradient path for learning long-range dependencies. The Transformer also introduced positional encodings to inject sequence order information, since the self-attention mechanism is inherently permutation-equivariant. Transformers were pre-trained at scale with masked language modelling (BERT [41]) and demonstrated remarkable transfer learning capabilities, spawning the modern era of foundation models.

### 3.6.3 Vision Transformer (ViT)

Dosovitskiy et al. [42] demonstrated that the Transformer architecture could be applied directly to images with minimal modifications, requiring no convolutional inductive biases, provided sufficient pre-training data was available. In ViT, an input image is divided into fixed-size patches (typically 16Ã—16 pixels), each flattened and linearly projected to form a sequence of tokens, which is then processed by a standard Transformer encoder. ViT pre-trained on the large JFT-300M dataset achieved competitive or superior performance to state-of-the-art CNNs on ImageNet, while showing inferior performance when trained only on ImageNet-scale data â€” demonstrating that the Transformer's weaker inductive biases require larger datasets to compensate. ViT established patches-as-tokens as the canonical interface between visual inputs and Transformer architectures, directly enabling the video extensions described below.

### 3.6.4 TimeSformer

Bertasius, Wang, and Torresani [43] proposed TimeSformer (Time-Space Transformer) for video understanding by extending ViT to the spatiotemporal domain. The key challenge in applying ViT to video is the dramatic increase in sequence length: a video of T frames with spatial patches of size 16Ã—16 from a 224Ã—224 frame produces TÃ—196 tokens, which is computationally intractable for full self-attention with T=8 (yielding 1,568 tokens and O(NÂ²) attention cost). TimeSformer addressed this through divided space-time attention: temporal self-attention and spatial self-attention were applied sequentially, with each token attending first to all tokens from the same spatial location across time, then to all tokens from the same frame. This factorisation reduced the attention complexity from O(TÂ²HÂ²WÂ²) to O(TÂ²+HÂ²WÂ²), making the model tractable while preserving the ability to model both temporal and spatial dependencies. TimeSformer achieved 78.0% on Kinetics-400 and demonstrated that pure attention-based models, without any convolutional layers, could compete with strong 3D-CNN baselines.

### 3.6.5 Video Swin Transformer

Liu et al. [44] proposed the Video Swin Transformer, extending the Swin Transformer's shifted-window self-attention mechanism to the spatiotemporal domain. Rather than computing global self-attention, Swin computes attention within local windows of fixed size and uses a shifting schedule across layers to enable cross-window connections. In the video variant, windows are defined over 3D spatiotemporal volumes (e.g., 8Ã—7Ã—7 patches), and shifted windows allow adjacent 3D windows to exchange information. This approach achieves linear computational complexity in the number of tokens (with respect to video size) while maintaining local and shifted-window connectivity sufficient to model the spatiotemporal patterns relevant to action recognition. Video Swin achieved 84.9% on Kinetics-400 [44], surpassing TimeSformer and establishing the shifted-window Transformer as a leading architecture for video.

### 3.6.6 VideoMAE: Masked Autoencoding for Video

The dominant paradigm for initialising large Transformer-based video models requires expensive supervised pre-training on Kinetics with millions of labelled examples. Tong et al. [45] proposed VideoMAE, which adapted the Masked Autoencoder (MAE) framework of He et al. [46] to video, enabling highly efficient self-supervised pre-training. In VideoMAE, an extremely high proportion (90%) of spatiotemporal patch tokens are masked, and the model is tasked with reconstructing the pixel values of the masked patches from the visible ones. The masking ratio of 90% is substantially higher than the 75% used in image MAE [46], motivated by the observation that consecutive video frames contain high spatiotemporal redundancy â€” a model can trivially reconstruct a masked patch by copying from adjacent unmasked frames. The high masking ratio forces the model to learn genuine spatiotemporal reasoning rather than exploiting local temporal interpolation. The encoder operates only on the visible tokens (roughly 10% of the total), dramatically reducing the computational cost of pre-training, while a lightweight decoder processes the full token set including mask tokens for reconstruction. VideoMAE with a ViT-Base backbone achieves 83.9% on Kinetics-400 [45] after fine-tuning, competitive with much more expressive architectures trained with supervised objectives on the same data. Crucially, VideoMAE demonstrates strong data efficiency: pre-training on domain-specific data such as UCF101 or HMDB51 itself, rather than Kinetics, can be effective due to the self-supervised objective, reducing dependence on large external labelled datasets. VideoMAE (ViT-Base fine-tuned from Kinetics-400 pre-training) serves as the Transformer baseline in this thesis, providing a strong representative of the modern attention-based paradigm.

### 3.6.7 Self-Supervised and Foundation Model Approaches

Beyond VideoMAE, a rich body of work has explored self-supervised pre-training for video representations. Contrastive methods including SimCLR [47], MoCo [48], and DINO [49] applied contrastive objectives to video by treating different temporal crops or augmentations of the same video as positive pairs. CLIP [50] demonstrated that image-text contrastive pre-training on web-scale data produced representations that transfer remarkably well to video recognition through zero-shot or few-shot inference, motivating subsequent work on video-language pre-training. These foundation model approaches represent the current frontier of the field, though they require resources that preclude direct comparison in the scope of this thesis.

---

## 3.7 Benchmark Datasets for Action Recognition

The progression of action recognition performance has been inextricably linked to the availability of increasingly large, diverse, and challenging benchmark datasets. Each new dataset has exposed limitations of existing methods and driven architectural innovation.

### 3.7.1 Early Benchmarks

The KTH dataset [51] provided an early standardised benchmark for action recognition, consisting of 2,391 video sequences of six simple actions (walking, jogging, running, boxing, waving, clapping) performed by 25 subjects in controlled indoor and outdoor environments. While KTH enabled systematic comparison of early methods, its small scale, limited action vocabulary, and controlled conditions meant that saturating its accuracy (above 90% was achieved by multiple early methods) did not predict real-world performance. The Weizmann dataset [52] similarly provided a controlled small-scale benchmark, useful for algorithm development but insufficient for evaluating generalisation.

### 3.7.2 HMDB51

Kuehne et al. [53] introduced HMDB51 (Human Motion DataBase), which collected 6,766 video clips across 51 action categories from diverse real-world sources including movies, YouTube, and web repositories. Three official train/test splits were provided, each with 70 training and 30 test clips per class, with performance typically reported as the mean accuracy across all three splits. HMDB51 was substantially more challenging than UCF101 for two reasons: inter-class similarity (many action pairs such as "climb" and "climb stairs", or "run" and "jog" shared very similar visual appearances), and within-class variability due to the diversity of source material, camera angles, and recording conditions. HMDB51 is used as one of the two primary evaluation benchmarks in this thesis, providing a challenging setting that discriminates model performance across architectures.

### 3.7.3 UCF101

Soomro et al. [54] introduced UCF101, which extended the earlier UCF50 dataset to 101 action classes with 13,320 video clips collected from YouTube. UCF101 covers five major action groups: human-object interaction, body motion, human-human interaction, playing musical instruments, and sports. As with HMDB51, three official train/test splits are provided, and accuracy is averaged across splits. UCF101 became the most widely cited action recognition benchmark for the 2012â€“2018 period due to its relatively large scale, clear class definitions, and the availability of pre-segmented, trimmed clips that simplified evaluation. UCF101 is used as the second evaluation benchmark in this thesis, complementing HMDB51 by providing a somewhat easier, more diverse test of recognition accuracy.

### 3.7.4 Kinetics

Kay et al. [55] introduced Kinetics-400 in 2017, a dataset of approximately 240,000 ten-second YouTube video clips across 400 action classes, with a separate 50,000-clip validation set and held-out test set. Kinetics was substantially larger and more diverse than UCF101 and HMDB51, and its scale enabled training of large-capacity 3D CNNs and Transformers from scratch or as a strong pre-training target. The release of Kinetics-400 enabled I3D [34] to demonstrate the importance of large-scale video pre-training and triggered a shift in the field toward Kinetics-pretrained models as the standard starting point for UCF101/HMDB51 experiments. Kinetics-600 [56] and Kinetics-700 [57] subsequently expanded the action vocabulary and clip count. In this thesis, R(2+1)D and VideoMAE both use Kinetics-400 pre-trained weights as initialisation before fine-tuning, and any comparison of reported results from the literature should note whether Kinetics pre-training was used, as it provides a large advantage.

### 3.7.5 Other Benchmarks

THUMOS [58] introduced untrimmed video benchmarks requiring temporal action localisation â€” not just clip-level classification â€” extending the difficulty of the recognition task. ActivityNet [59] further scaled temporal localisation to 200 action classes with over 20,000 videos averaging several minutes in duration. Something-Something v2 [60] provided a benchmark specifically designed to require temporal reasoning: 220,847 short clips of humans performing fine-grained object interactions where the correct action label depends on the temporal order of events (e.g., "moving something from left to right" vs. "moving something from right to left"). Models that rely solely on appearance statistics without modelling temporal order perform poorly on Something-Something, making it a diagnostic tool for temporal reasoning capabilities. AVA [61] introduced spatiotemporal action localisation across 80 atomic action classes, requiring both detection and recognition of multiple simultaneously occurring actions at the person level.

---

## 3.8 Comparative Studies and Efficiency Analysis

Accurate assessment of a model's practical value requires not only measuring its accuracy on benchmarks but also characterising its computational cost and memory requirements. The literature contains numerous comparative studies, though direct comparison across papers is complicated by inconsistent evaluation protocols.

### 3.8.1 Efficiency Benchmarks for Image Classification

Canziani et al. [62] conducted a systematic empirical comparison of image classification architectures along multiple dimensions including accuracy, computational cost (GFLOPs per inference), memory footprint, and inference time on a fixed GPU. Their analysis revealed that many high-accuracy architectures were highly inefficient in practice: VGGNet consumed disproportionate memory relative to its accuracy, while ResNets occupied a favourable region of the accuracy-efficiency trade-off. This style of multi-dimensional benchmarking â€” rather than ranking purely by accuracy â€” directly motivated the framing of this thesis, where accuracy versus efficiency trade-offs are the primary object of study.

### 3.8.2 Video Model Efficiency Studies

Fan et al. [63] released PySlowFast, an open-source framework for video understanding, along with a systematic study of video model efficiency. Their analysis characterised the GFLOPs and parameter counts of SlowFast variants, X3D, and other architectures under standardised conditions, demonstrating that X3D achieved accuracy competitive with SlowFast at a fraction of the computational cost. However, such analyses are sensitive to the number of input frames, spatial resolution, and whether multi-crop or single-crop evaluation is used â€” parameters that vary substantially across papers and make direct comparisons misleading.

### 3.8.3 The Difficulty of Fair Comparison

A persistent challenge in the video recognition literature is the difficulty of performing truly fair comparisons across architectures. Different papers use different numbers of input frames (8, 16, 32, 64), different spatial resolutions (224Ã—224, 256Ã—256, 320Ã—320), different numbers of test clips and crops per video (1 clip Ã— 1 crop to 10 clips Ã— 3 crops), different pre-training datasets and durations, and different data augmentation strategies. Bertasius et al. [43] noted in their TimeSformer paper that architecture comparisons without controlling for these variables are unreliable, as a factor of 2Ã— in frame count or test crops can account for several percentage points of accuracy. The Something-Something survey by Tran et al. [36] similarly observed that reported numbers are often not directly comparable across methods due to these confounds. This observation directly motivates the contribution of this thesis: by training and evaluating all four architectures (ResNet-50 TSN, LRCN, R(2+1)D-18, and VideoMAE ViT-B) under a single, controlled experimental protocol with identical pre-processing, frame counts, and evaluation procedures, the accuracy versus efficiency trade-offs can be assessed without the confounding factors that plague cross-paper comparisons.

### 3.8.4 The Architecture Landscape circa 2024

The progression from hand-crafted features through 2D CNNs, RNNs, 3D CNNs, and Transformers reflects a consistent trend: each generation addressed a specific limitation of its predecessor. Hand-crafted methods were replaced by learned features capable of adapting to training data. Two-stream and TSN-style 2D CNNs processed spatial appearance efficiently but required explicit optical flow for motion. RNNs added sequential temporal modelling but were limited by sequential computation and vanishing gradients. 3D CNNs jointly learned spatiotemporal features but were computationally expensive and dependent on large-scale pre-training. Transformers modelled global spatiotemporal dependencies without the spatial locality bias of CNNs, but required even larger pre-training datasets to overcome their lack of inductive biases. Self-supervised pre-training, exemplified by VideoMAE, partially addressed the data requirement by learning from unlabelled video.

What remains less well understood in the literature is the systematic accuracy-versus-efficiency characterisation of representative models from each generation under controlled conditions on standard benchmarks. Existing surveys [64] provide broad coverage but typically aggregate results from disparate papers rather than measuring under a single experimental protocol. The present thesis addresses this gap directly, providing an empirical comparison of four architectures spanning three generations â€” 2D-CNN (ResNet-50 TSN), RNN (LRCN), 3D-CNN (R(2+1)D-18), and Transformer (VideoMAE ViT-B) â€” measured under identical conditions on UCF101 and HMDB51, with runtime profiling to characterise inference cost alongside accuracy.

---

**References**

[1] N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in *Proc. CVPR*, 2005.

[2] N. Dalal, B. Triggs, and C. Schmid, "Human detection using oriented histograms of flow and appearance," in *Proc. ECCV*, 2006.

[3] H. Wang, A. Klaser, C. Schmid, and C.-L. Liu, "Action recognition by dense trajectories," in *Proc. CVPR*, 2011.

[4] H. Wang and C. Schmid, "Action recognition with improved trajectories," in *Proc. ICCV*, 2013.

[5] I. Laptev, "On space-time interest points," *Int. J. Comput. Vis.*, vol. 64, no. 2â€“3, pp. 107â€“123, 2005.

[6] P. Dollar, V. Rabaud, G. Cottrell, and S. Belongie, "Behavior recognition via sparse spatio-temporal features," in *Proc. VS-PETS*, 2005.

[7] G. Csurka, C. Dance, L. Fan, J. Willamowski, and C. Bray, "Visual categorization with bags of keypoints," in *Proc. ECCV Workshops*, 2004.

[8] F. Perronnin, J. Sanchez, and T. Mensink, "Improving the Fisher kernel for large-scale image classification," in *Proc. ECCV*, 2010.

[9] G. Farneback, "Two-frame motion estimation based on polynomial expansion," in *Proc. SCIA*, 2003.

[10] C. Zach, T. Pock, and H. Bischof, "A duality based approach for realtime TV-L1 optical flow," in *Proc. DAGM*, 2007.

[11] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," *Proc. IEEE*, vol. 86, no. 11, pp. 2278â€“2324, 1998.

[12] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," in *Proc. NeurIPS*, 2012.

[13] O. Russakovsky et al., "ImageNet large scale visual recognition challenge," *Int. J. Comput. Vis.*, vol. 115, no. 3, pp. 211â€“252, 2015.

[14] V. Nair and G. E. Hinton, "Rectified linear units improve restricted Boltzmann machines," in *Proc. ICML*, 2010.

[15] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, pp. 1929â€“1958, 2014.

[16] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *Proc. ICLR*, 2015.

[17] C. Szegedy et al., "Going deeper with convolutions," in *Proc. CVPR*, 2015.

[18] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. ICML*, 2015.

[19] S. J. Pan and Q. Yang, "A survey on transfer learning," *IEEE Trans. Knowl. Data Eng.*, vol. 22, no. 10, pp. 1345â€“1359, 2010.

[20] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How transferable are features in deep neural networks?" in *Proc. NeurIPS*, 2014.

[21] K. Simonyan and A. Zisserman, "Two-stream convolutional networks for action recognition in videos," in *Proc. NeurIPS*, 2014.

[22] L. Wang et al., "Temporal segment networks: Towards good practices for deep action recognition," in *Proc. ECCV*, 2016.

[23] R. Girdhar, D. Ramanan, A. Gupta, J. Sivic, and B. Russell, "ActionVLAD: Learning spatio-temporal aggregation for action classification," in *Proc. CVPR*, 2017.

[24] B. Zhou, A. Andonian, A. Oliva, and A. Torralba, "Temporal relational reasoning in videos," in *Proc. ECCV*, 2018.

[25] Y. Bengio, P. Simard, and P. Frasconi, "Learning long-term dependencies with gradient descent is difficult," *IEEE Trans. Neural Netw.*, vol. 5, no. 2, pp. 157â€“166, 1994.

[26] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Comput.*, vol. 9, no. 8, pp. 1735â€“1780, 1997.

[27] F. A. Gers, J. Schmidhuber, and F. Cummins, "Learning to forget: Continual prediction with LSTM," *Neural Comput.*, vol. 12, no. 10, pp. 2451â€“2471, 2000.

[28] M. Schuster and K. K. Paliwal, "Bidirectional recurrent neural networks," *IEEE Trans. Signal Process.*, vol. 45, no. 11, pp. 2673â€“2681, 1997.

[29] K. Cho et al., "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in *Proc. EMNLP*, 2014.

[30] J. Donahue et al., "Long-term recurrent convolutional networks for visual recognition and description," in *Proc. CVPR*, 2015.

[31] S. Ji, W. Xu, M. Yang, and K. Yu, "3D convolutional neural networks for human action recognition," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 35, no. 1, pp. 221â€“231, 2013.

[32] M. Baccouche, F. Mamalet, C. Wolf, C. Garcia, and A. Baskurt, "Sequential deep learning for human action recognition," in *Proc. HBU*, 2011.

[33] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, "Learning spatiotemporal features with 3D convolutional networks," in *Proc. ICCV*, 2015.

[34] J. Carreira and A. Zisserman, "Quo vadis, action recognition? A new model and the Kinetics dataset," in *Proc. CVPR*, 2017.

[35] Z. Qiu, T. Yao, and T. Mei, "Learning spatio-temporal representation with pseudo-3D residual networks," in *Proc. ICCV*, 2017.

[36] D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri, "A closer look at spatiotemporal convolutions for video understanding," in *Proc. CVPR*, 2018.

[37] C. Feichtenhofer, H. Fan, J. Malik, and K. He, "SlowFast networks for video recognition," in *Proc. ICCV*, 2019.

[38] C. Feichtenhofer, "X3D: Expanding architectures for efficient video recognition," in *Proc. CVPR*, 2020.

[39] A. Vaswani et al., "Attention is all you need," in *Proc. NeurIPS*, 2017.

[40] D. Bahdanau, K. Cho, and Y. Bengio, "Neural machine translation by jointly learning to align and translate," in *Proc. ICLR*, 2015.

[41] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL*, 2019.

[42] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. ICLR*, 2021.

[43] G. Bertasius, H. Wang, and L. Torresani, "Is space-time attention all you need for video understanding?" in *Proc. ICML*, 2021.

[44] Z. Liu et al., "Video Swin Transformer," in *Proc. CVPR*, 2022.

[45] Z. Tong, Y. Song, J. Wang, and L. Wang, "VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training," in *Proc. NeurIPS*, 2022.

[46] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, "Masked autoencoders are scalable vision learners," in *Proc. CVPR*, 2022.

[47] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A simple framework for contrastive learning of visual representations," in *Proc. ICML*, 2020.

[48] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, "Momentum contrast for unsupervised visual representation learning," in *Proc. CVPR*, 2020.

[49] M. Caron et al., "Emerging properties in self-supervised vision transformers," in *Proc. ICCV*, 2021.

[50] A. Radford et al., "Learning transferable visual models from natural language supervision," in *Proc. ICML*, 2021.

[51] C. Schuldt, I. Laptev, and B. Caputo, "Recognizing human actions: A local SVM approach," in *Proc. ICPR*, 2004.

[52] M. Blank, L. Gorelick, E. Shechtman, M. Irani, and R. Basri, "Actions as space-time shapes," in *Proc. ICCV*, 2005.

[53] H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre, "HMDB: A large video database for human motion recognition," in *Proc. ICCV*, 2011.

[54] K. Soomro, A. R. Zamir, and M. Shah, "UCF101: A dataset of 101 human actions classes from videos in the wild," arXiv:1212.0402, 2012.

[55] W. Kay et al., "The Kinetics human action video dataset," arXiv:1705.06950, 2017.

[56] J. Carreira et al., "A short note about Kinetics-600," arXiv:1808.01340, 2018.

[57] L. Smaira et al., "A short note on the Kinetics-700-2020 human action dataset," arXiv:2010.10864, 2020.

[58] Y. Jiang et al., "THUMOS challenge: Action recognition with a large number of classes," in *Proc. ECCV Workshops*, 2014.

[59] F. Caba Heilbron, V. Escorcia, B. Ghanem, and J. C. Niebles, "ActivityNet: A large-scale video benchmark for human activity understanding," in *Proc. CVPR*, 2015.

[60] R. Goyal et al., "The 'something something' video database for learning and evaluating visual common sense," in *Proc. ICCV*, 2017.

[61] C. Gu et al., "AVA: A video dataset of spatio-temporally localized atomic visual actions," in *Proc. CVPR*, 2018.

[62] A. Canziani, A. Paszke, and E. Culurciello, "An analysis of deep neural network models for practical applications," arXiv:1605.07678, 2016.

[63] H. Fan et al., "PySlowFast: An open source video understanding codebase," arXiv:2012.07669, 2020.

[64] Z. Kong and W. Jing, "A comprehensive study of deep video action recognition," arXiv:2012.06567, 2022.


---

# Chapter 4: Project Presentation

## 4.1 System Architecture Overview

The experimental system developed for this thesis is designed around three core engineering principles: reproducibility, comparability, and extensibility. Every architectural decision at the system level — from how raw video is loaded to how training metrics are logged — is made in service of ensuring that the four model families being compared are evaluated under rigorously identical conditions.

The pipeline can be understood as a linear chain of transformations: raw media on disk (either video files or pre-extracted JPEG frame sequences) is consumed by the data pipeline, which produces batched clip tensors in a canonical shape `(B, T, C, H, W)` (batch size, temporal frames, colour channels, height, width). These tensors are passed to a model, which maps them to logits `(B, num_classes)`. During training, the logits are consumed by a cross-entropy loss; during inference, the argmax over the class dimension yields the predicted action label.

The system is model-agnostic by design. The training loop in `src/train.py`, the benchmarking harness in `src/benchmark.py`, and the evaluation routine in `src/evaluate.py` contain no model-specific code. Each of the four architectures implements the same Python interface: a subclass of `torch.nn.Module` whose `forward` method accepts `(B, T, C, H, W)` float tensors and returns `(B, num_classes)` logits. Any adapter logic required to reconcile the architecture's native input convention with this shared contract — such as the `(B,C,T,H,W)` permutation needed by torchvision's 3D CNN — is encapsulated within the model's own `forward` method and is transparent to the rest of the system.

The global configuration in `src/config.py` is the single source of truth for all shared parameters: the number of frames sampled per clip (`NUM_FRAMES = 16`), the spatial resolution (`IMG_SIZE = 224`), and the ImageNet normalisation statistics applied uniformly to all models. This single-source discipline guarantees that every model receives inputs produced by precisely the same transformation chain — a prerequisite for any comparative study that aims to isolate architectural differences rather than pipeline differences.

The end-to-end data flow is as follows. At dataset instantiation, `build_dataset()` scans the dataset root and constructs a list of `(relative_path, label_index)` tuples representing each clip in the training or test split. These are wrapped in a `VideoClipDataset` instance, which the PyTorch `DataLoader` iterates in parallel across worker processes. Within each worker, `__getitem__` is called for a given index: the appropriate clip is loaded from disk (either via `decord.VideoReader` for video files or PIL/NumPy for frame directories), `NUM_FRAMES` frames are sampled at uniformly-spaced temporal positions, spatial augmentation and normalisation are applied, and the resulting `(T, C, H, W)` tensor is returned alongside its class index. The DataLoader collates individual tensors into a `(B, T, C, H, W)` batch, which is transferred to the GPU and passed to the model. After forward pass and loss computation, gradients flow backwards through the model (and through the GradScaler in the mixed-precision case), the optimiser updates parameters, and the learning rate scheduler advances its internal counter. At the end of each epoch, `evaluate_model()` runs the model in inference mode over the test set and returns top-1 and top-5 accuracy. Results are appended to a per-model CSV file and, if the accuracy has improved, model weights are checkpointed to disk.

This architecture cleanly separates the concerns of data engineering, model definition, optimisation, and evaluation, making it straightforward to add new architectures or datasets without modifying any existing module.

---

## 4.2 Datasets

### 4.2.1 UCF101

UCF101 [Soomro et al., 2012] is a widely used benchmark dataset for human action recognition in realistic video conditions. It consists of 13,320 video clips distributed across 101 action classes, collected from YouTube. The dataset spans five broad activity groups: body motion only (e.g., Walking, Jumping), human-object interaction (e.g., Playing Guitar, Archery), human-human interaction (e.g., Handshaking, Boxing), playing musical instruments (e.g., Playing Piano, Violin), and sports (e.g., Basketball, Surfing). This diversity of action types — including both fine-grained hand and arm movements and gross whole-body motions — makes UCF101 a challenging and representative benchmark.

Clips are encoded at 25 frames per second with a typical duration of 6 to 10 seconds per clip, yielding between 150 and 250 raw frames per clip before subsampling. Spatial resolution varies across clips, as the dataset was assembled from naturally heterogeneous web sources, but clips span a range roughly centred on 320×240 pixels. Intra-class variation arises from viewpoint changes, illumination differences, camera motion (both handheld jitter and deliberate camera panning), and background clutter. Inter-class confusion is present in some groups — distinguishing "Biking" from "Rowing" at a single frame level, for instance, can be ambiguous without temporal context.

The dataset provides three official train/test splits, each partitioning the 13,320 clips into approximately 9,500 training and 3,800 test clips in a stratified manner. The splits are defined by three files: `classInd.txt`, which maps each of the 101 class names to a one-based integer index; `trainlistNN.txt`, containing relative paths of training clips for split NN; and `testlistNN.txt`, containing test clip paths. In this study, split 1 is used exclusively, consistent with the single-split evaluation protocol adopted by a number of prominent prior works.

The pipeline's `build_dataset()` function first attempts to locate `classInd.txt` and the appropriate split list files via recursive search under the dataset root. When found, it parses the split file directly, mapping class names to zero-based indices and constructing the list of `(relative_path, label_index)` tuples that define the dataset. This is the preferred code path for UCF101, as the official splits are deterministic and reproducible across different download mirrors. If the split files are absent (as may occur with some Kaggle dataset packages), the pipeline falls back to the layout-agnostic scan described in Section 4.3.

### 4.2.2 HMDB51

HMDB51 [Kuehne et al., 2011] is a complementary benchmark containing 6,766 video clips across 51 action classes. It was assembled from a diverse range of sources including Hollywood films, online video platforms, Prelinger Archive historical footage, and other publicly available video repositories. This heterogeneity of source material is both a strength and a challenge: the dataset captures a wide range of recording conditions, compression artefacts, aspect ratios, and cinematic conventions that are absent from YouTube-only datasets.

The 51 classes are drawn from categories including facial actions (e.g., Smile, Chew), body movement (e.g., Cartwheel, Handstand), human interaction (e.g., Hug, Kiss), general movements (e.g., Wave, Clap), and sport or leisure activities (e.g., Golf, Shoot Ball). The average number of clips per class is approximately 133, making this a comparatively small dataset where generalisation from limited per-class examples is the primary challenge. Several classes are particularly difficult to discriminate due to high visual similarity: "Drink" and "Eat" share similar arm and hand kinematics; "Kick" and "Punch" involve similar whole-body postures.

HMDB51 poses additional challenges relative to UCF101. Camera motion is more prevalent, as a significant proportion of clips were originally captured in cinematographic contexts involving deliberate panning, zooming, and tracking shots. Viewpoint variation is substantial, with the same action class appearing from dramatically different camera angles across clips. Inter-class visual similarity — for instance between actions involving hand-to-mouth contact — means that accurate classification requires precise temporal reasoning rather than static appearance alone.

The dataset provides three official train/test splits, each assigning 70 clips per class to training and 30 per class to test. However, the mirror of HMDB51 used in this study provides clips in the "RawFrames" format (pre-extracted JPEG sequences rather than video files) without the accompanying split annotation files. Consequently, this study constructs a deterministic 70/30 per-class random split using a fixed seed (seed = 42). The split is applied per class to ensure a balanced distribution, and is reproducible given the same dataset layout. This constitutes a deviation from the standard three-fold evaluation protocol and is acknowledged as a limitation of the experimental setup: results on this dataset are not directly comparable to published figures obtained with the official HMDB51 splits, and should be interpreted as indicative of relative model performance rather than absolute benchmark scores.

---

## 4.3 Data Pipeline

### 4.3.1 Video Decoding

For UCF101, raw video files are decoded using `decord` [Chen et al., 2020], a specialised video decoding library designed for machine learning workflows. Unlike general-purpose video processing frameworks such as FFmpeg (which must be invoked via subprocess) or OpenCV's `VideoCapture` (which does not support efficient random-access seeking), decord's `VideoReader` class provides Python-accessible random-access frame retrieval. This is critical for the uniform temporal sampling strategy: rather than decoding the entire video sequentially and discarding unwanted frames, `VideoReader` decodes only the specific frame indices requested, substantially reducing I/O and CPU cost.

The bridge between decord and PyTorch is configured at module import time with `decord.bridge.set_bridge("torch")`, which instructs decord to return `torch.Tensor` objects directly from `get_batch()` rather than NumPy arrays. The raw output of `vr.get_batch(idx)` is a tensor of shape `(T, H, W, C)` with unsigned 8-bit integer values in the range [0, 255] and channels in BGR order from the underlying FFMPEG decoder. A permutation operation converts this to `(T, C, H, W)` — the convention expected by torchvision's functional transforms — and the tensor is cast to float and scaled to the range [0, 1] by dividing by 255.

For HMDB51 in its RawFrames format, each action clip is stored as a directory of JPEG images named by frame number (e.g., `frame_000001.jpg`, `frame_000002.jpg`, ...). The `_load_frames_from_dir()` function handles this case: it first lists all files in the directory whose extension belongs to the set `{.jpg, .jpeg, .png, .bmp}`, sorts them lexicographically (which preserves temporal order given zero-padded numeric filenames), and then loads the subset of files indicated by the uniform index set computed by `_uniform_indices()`. Each frame is opened with PIL's `Image.open()`, converted to RGB (ensuring consistent three-channel output regardless of source encoding), converted to a NumPy array, and then to a uint8 tensor via `torch.from_numpy()`. The explicit NumPy intermediate is necessary to produce a writable tensor — PIL's memory buffer is read-only. Frames are stacked along dimension 0 and scaled to [0, 1], producing the same `(T, C, H, W)` float tensor format as the decord code path.

Both paths converge to the same representation before the transform pipeline, making the remainder of the data loading code format-agnostic.

### 4.3.2 Frame Sampling Strategy

A fundamental decision in video understanding is how many frames to sample from each clip and at which temporal positions. This study employs uniform temporal sampling, implemented by the `_uniform_indices()` function:

```python
def _uniform_indices(total, n):
    if total <= 0:
        return [0] * n
    if total < n:
        pad = [total - 1] * (n - total)
        return list(range(total)) + pad
    return [int(round(x)) for x in np.linspace(0, total - 1, n)]
```

For a clip of `total` frames, `np.linspace(0, total - 1, n)` generates `n` floating-point values equally spaced between the first and last frame index. Rounding these to integers yields the selected frame indices. This strategy samples frames from the beginning, middle, and end of the clip, ensuring that the entire temporal extent is represented even when only a small number of frames is extracted. Edge cases are handled explicitly: if the clip is shorter than the requested frame count, the available frames are taken in full and the final frame is repeated to pad the sequence to length `n`; if `total` is zero (an invalid or corrupt clip), a sequence of `n` zero-indices is returned and the resulting black clip is handled gracefully by the corrupt-clip fallback in `__getitem__`.

Uniform sparse sampling stands in contrast to dense sampling strategies, in which a contiguous window of consecutive frames is extracted from a randomly chosen temporal position within the clip. Dense sampling captures fine-grained local motion — the precise trajectory of a hand during a throwing action, for instance — but limits the temporal field of view to a window of a few seconds. Sparse sampling, by contrast, provides a global view of the clip's temporal evolution at the cost of missing high-frequency motion signals. For action categories defined primarily by their overall motion pattern and activity type (as in UCF101 and HMDB51), sparse sampling is generally the more informative strategy, as it exposes all four models to evidence from the full clip duration rather than an arbitrary sub-segment.

The value `NUM_FRAMES = 16` was chosen as a practical balance between temporal coverage and computational budget. It is also the exact input length required by the VideoMAE-base architecture — which was pre-trained with 16-frame clips — making this choice particularly appropriate for the Transformer model. For the 2D-CNN models, any number of frames is acceptable since each frame is processed independently, but a shared frame count ensures that the total amount of temporal information presented to every model is identical.

### 4.3.3 Data Augmentation

The augmentation pipeline is deliberately minimal and follows the standard practice for fine-tuning pre-trained video models. The training augmentation consists of three operations applied identically across all `T` frames of a clip:

1. **Resize**: the shorter spatial edge of each frame is rescaled to 256 pixels using bilinear interpolation with anti-aliasing (`TF.resize(clip, 256, antialias=True)`). The longer edge is scaled proportionally, preserving the original aspect ratio.
2. **Random crop**: a 224×224 patch is randomly cropped from the resized frame. The crop parameters (top-left corner coordinates) are sampled once for the entire clip — all `T` frames receive the same crop — ensuring that the spatial relationship between frames is preserved.
3. **Random horizontal flip**: with probability 0.5, all frames in the clip are horizontally mirrored. The flip is applied consistently across all frames to preserve optical flow consistency.

For evaluation, only the resize and a centre crop (extracting the central 224×224 patch) are applied; no random transforms are used, ensuring deterministic and reproducible test-time behaviour.

All frames are then normalised per channel using ImageNet statistics: mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]` in RGB order. This normalisation is applied to all four models, including R(2+1)D and VideoMAE which were pre-trained on Kinetics-400 rather than ImageNet. Kinetics-400 was itself collected from YouTube and displays broadly similar colour statistics to the ImageNet distribution; in practice, using ImageNet normalisation for Kinetics-pretrained models introduces negligible degradation and is standard practice in the community.

The choice of minimal augmentation is deliberate. More aggressive augmentation strategies — such as temporal jittering, colour jitter, Mixup, or RandAugment — might improve absolute accuracy and would be appropriate for a study aiming to maximise benchmark performance. However, the goal of this study is a controlled comparison of architectural inductive biases, not a search for optimal data augmentation hyperparameters. Using the same minimal augmentation policy for all four models ensures that observed accuracy differences are attributable to the architectures rather than to differing sensitivity to augmentation strength.

---

## 4.4 Model Architectures

### 4.4.1 ResNet-50 Temporal Segment Network (2D-CNN Baseline)

The simplest possible extension of a pre-trained image classifier to video is to apply it independently to each frame and combine the resulting per-frame predictions. The Temporal Segment Network (TSN) framework, proposed by Wang et al. (2016), formalises this idea as the consensus function: segment-level predictions (here, per-frame logits) are aggregated by averaging. This model serves as the 2D-CNN baseline in this study.

The architecture consists of a standard ResNet-50 backbone, pre-trained on ImageNet-1K, with the original 1000-class linear classification head replaced by a new `nn.Linear(2048, num_classes)` layer. The implementation is as follows:

```python
class ResNet50TSN(nn.Module):
    """2D-CNN baseline: per-frame ResNet-50, logits averaged over time (TSN consensus)."""
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):                       # (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.backbone(x)                    # (B*T, num_classes)
        return x.view(b, t, -1).mean(dim=1)     # temporal average
```

The forward pass is implemented efficiently by reshaping the `(B, T, C, H, W)` input to `(B*T, C, H, W)`, effectively treating all `T` frames of all `B` clips in the batch as a single large batch of images. The backbone processes this concatenated batch in a single forward pass, yielding `(B*T, num_classes)` logits. These are reshaped back to `(B, T, num_classes)` and averaged over the temporal dimension, producing the final `(B, num_classes)` logit tensor. This reshaping trick is computationally equivalent to processing each frame independently but takes advantage of GPU parallelism to do so in a single kernel call.

ResNet-50 follows the standard residual architecture: an initial 7×7 convolution with stride 2, followed by a 3×3 max-pooling layer, then four stages of residual bottleneck blocks (with 3, 4, 6, and 3 blocks respectively), a global average pooling layer, and the classification head. Bottleneck blocks consist of a 1×1 convolution that reduces channel dimensionality, a 3×3 convolution at the reduced dimension, and a 1×1 convolution that expands back to the block's output width, with a shortcut connection bypassing all three operations. The total parameter count is 25.6 million for the standard architecture; with 101 or 51 output classes (replacing the original 1000-class head), the counts in this study are 23.71 million (UCF101) for the complete model.

This model is chosen as the baseline because it makes no assumptions about temporal structure whatsoever: the consensus function is a simple mean, which discards all ordering information among frames. Any improvement observed in the CNN-RNN, 3D-CNN, or Transformer models therefore reflects the value added by explicit temporal modelling relative to this frame-level baseline.

### 4.4.2 ResNet-50 Bidirectional LSTM (CNN-RNN)

The CNN-RNN architecture, often called Long-term Recurrent Convolutional Network (LRCN) following Donahue et al. (2015), combines a convolutional feature extractor with a recurrent sequence model. Per-frame CNN features are extracted and arranged as a temporal sequence, which is then processed by an LSTM to model temporal dependencies explicitly. Unlike the TSN baseline, which performs a commutative mean over frames, the LSTM maintains a hidden state that is updated sequentially (or in this case, bidirectionally), allowing it to capture the ordering and dynamics of the feature sequence.

A critical design choice in this study is that the CNN-RNN uses the same ResNet-50 backbone as the TSN baseline. The fully connected classification layer is removed and replaced with `nn.Identity()`, causing the backbone to output a 2048-dimensional feature vector per frame. This feature sequence is fed to a bidirectional LSTM, and the result is passed through a final linear layer:

```python
class CNNLSTM(nn.Module):
    """CNN-RNN (LRCN): per-frame ResNet-50 features fed to bidirectional LSTM."""
    def __init__(self, num_classes, hidden=512, layers=1):
        super().__init__()
        self.backbone = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )
        feat_dim = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Identity()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)  # *2: bidirectional

    def forward(self, x):                       # (B, T, C, H, W)
        b, t, c, h, w = x.shape
        feats = self.backbone(x.reshape(b * t, c, h, w))  # (B*T, 2048)
        feats = feats.view(b, t, -1)                       # (B, T, 2048)
        out, _ = self.lstm(feats)                          # (B, T, 2*hidden)
        return self.fc(out.mean(dim=1))                    # temporal pool -> logits
```

The ResNet-50 backbone processes all frames in a single reshaping pass (identical to the TSN implementation) to extract `(B, T, 2048)` feature sequences. The bidirectional LSTM processes this sequence in both temporal directions: the forward pass reads frames from position 1 to T, accumulating hidden state; the backward pass reads from T to 1. At each time step t, the forward hidden state `h_t^forward` and backward hidden state `h_t^backward` are concatenated to produce a 1024-dimensional representation. The mean over T of these bidirectional hidden states is taken before the final linear projection to class logits.

The bidirectional design enables each time step's representation to incorporate context from both its past (forward pass) and its future (backward pass), which is beneficial for offline classification of pre-recorded clips where the full temporal context is available at inference time. The LSTM hidden dimension is 512, yielding 1024 dimensions after concatenation in the bidirectional case.

By sharing the exact same ResNet-50 backbone as the TSN baseline, this architecture forms a controlled experiment pair: any performance difference between TSN and CNN-LSTM is attributable solely to the presence and quality of recurrent temporal modelling, with no confounding from backbone architecture, pretraining, or feature representation differences. The LSTM adds approximately 10.4 million parameters (from the LSTM cell and the classification head), bringing the total to 34.11 million parameters. At inference time, the FLOPs attributable to the LSTM are not captured by fvcore's `FlopCountAnalysis` (which does not support `nn.LSTM`), making the reported 65.75 GFLOPs identical to the TSN figure; the LSTM contributes an additional approximately 0.3 GFLOPs per forward pass.

### 4.4.3 R(2+1)D-18 (3D-CNN)

R(2+1)D is a 3D CNN architecture proposed by Tran et al. (2018) that factorises full spatiotemporal convolutions into separate spatial and temporal components. A conventional 3D convolution with a kernel of size `d×d×t` operates across both spatial and temporal dimensions simultaneously. R(2+1)D instead performs a 2D spatial convolution with kernel `d×d×1` (which operates only in the spatial plane at each time step), followed by a 1D temporal convolution with kernel `1×1×t` (which aggregates information across time), with a rectified linear unit (ReLU) non-linearity applied between the two. This factorised structure provides two advantages: it separates spatial and temporal learning, facilitating optimisation; and it introduces an additional non-linearity between the two convolution operations, making the factorised version strictly more expressive than a monolithic 3D convolution of equivalent parameter count.

The network follows an 18-layer residual architecture analogous to ResNet-18, with each convolutional layer replaced by the `(2+1)D` factorised equivalent. Torchvision provides a pre-trained implementation initialised with weights trained on Kinetics-400 (a large-scale video dataset with 400 action classes and over 300,000 clips). The classification head is replaced with a new linear layer:

```python
class R2Plus1D(nn.Module):
    """3D-CNN: R(2+1)D-18, Kinetics-400 pretrained."""
    def __init__(self, num_classes):
        super().__init__()
        self.net = torchvision.models.video.r2plus1d_18(
            weights=torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1
        )
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x):                       # (B, T, C, H, W) -> (B, C, T, H, W)
        return self.net(x.permute(0, 2, 1, 3, 4))
```

The torchvision video model convention requires input in `(B, C, T, H, W)` format — channels before the temporal dimension — rather than the `(B, T, C, H, W)` convention used by the shared pipeline. The `permute(0, 2, 1, 3, 4)` call in the forward method adapts between these conventions without copying data. This permutation is the entirety of the adapter logic required to integrate the 3D-CNN into the model-agnostic pipeline.

Kinetics-400 pretraining is particularly important for 3D-CNN architectures. Unlike 2D-CNNs, which can be initialised from ImageNet weights (as the 2D convolutional filters transfer directly), 3D CNNs require that the temporal convolutional kernels also be initialised meaningfully. Training R(2+1)D-18 from random initialisation on UCF101 or HMDB51 alone — with fewer than 10,000 clips — would lead to severe overfitting and substantially degraded performance. Kinetics-400 pretraining provides the temporal feature representations with the necessary statistical grounding, which can then be refined for the target datasets through fine-tuning.

The total parameter count of R(2+1)D-18 is 31.33 million, with a computational cost of 162.58 GFLOPs per clip — the highest among the four architectures. This elevated compute cost reflects the expense of applying learned spatial and temporal convolutions across every spatiotemporal position in the input volume.

### 4.4.4 VideoMAE (Vision Transformer with Masked Autoencoding)

VideoMAE [Tong et al., 2022] extends the Vision Transformer (ViT) architecture to video by adopting a spatiotemporal patch tokenisation scheme, and is pre-trained using Masked Autoencoding (MAE) — a self-supervised pretraining paradigm in which a large proportion of input tokens are masked and the model is trained to reconstruct the original pixel values from the remaining visible tokens.

The architecture is ViT-Base, a 12-block transformer encoder with model dimension `d_model = 768`, 12 attention heads, and a feed-forward MLP dimension of 3072. Video input of shape `(B, 16, C, 224, 224)` is tokenised as follows: the 16 frames are grouped into temporal patches of 2 frames each (8 temporal groups), and each frame's spatial extent is divided into non-overlapping 16×16 patches (14×14 patches per frame). The resulting spatiotemporal patch tokens number `8 temporal_patches × 14 × 14 spatial_patches = 1,568` tokens. These tokens are projected from their raw pixel dimension to the model dimension `d_model = 768` by a linear patch embedding, positional embeddings are added, and the full sequence is processed by 12 successive transformer encoder blocks.

The model used in this study is initialised from the HuggingFace checkpoint `MCG-NJU/videomae-base-finetuned-kinetics`, which represents VideoMAE-base after the complete pretraining and fine-tuning pipeline: self-supervised masked autoencoding pretraining on Kinetics-400 with a 90% token masking ratio, followed by supervised fine-tuning on the full Kinetics-400 400-class classification task.

```python
class VideoMAE(nn.Module):
    """Transformer: VideoMAE-base, Kinetics-400 pretrained (HuggingFace)."""
    def __init__(self, num_classes):
        super().__init__()
        from transformers import VideoMAEForVideoClassification
        self.net = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):                       # (B, T, C, H, W)
        return self.net(pixel_values=x).logits
```

The `ignore_mismatched_sizes=True` argument is required because the pre-trained checkpoint contains a classification head projecting to 400 Kinetics classes, whereas the fine-tuning target in this study has either 101 (UCF101) or 51 (HMDB51) classes. Setting this flag causes HuggingFace's `from_pretrained` to discard the mismatched head weights and randomly initialise a new head of the correct size, while loading all other weights (the patch embedding, positional embeddings, and 12 transformer blocks) from the checkpoint. The HuggingFace `VideoMAEForVideoClassification` wrapper internally applies the mean-pool of the ViT encoder's output token sequence to obtain a clip-level representation before projecting to class logits.

The 90% masking ratio during pretraining is substantially higher than the 75% ratio used in the original image-domain MAE. This difference is motivated by the high spatiotemporal redundancy of video: adjacent frames share most of their pixel content, so models can trivially reconstruct masked tokens from neighbouring unmasked tokens if the masking ratio is low. A 90% ratio forces the model to learn genuinely long-range spatiotemporal dependencies in order to reconstruct masked regions — precisely the kind of representation that is beneficial for action recognition.

VideoMAE has by far the largest parameter count (86.28 million) of the four architectures in this study, but its computational cost of 135.17 GFLOPs is notably lower than R(2+1)D-18's 162.58 GFLOPs. This non-obvious efficiency advantage of the Transformer over the 3D-CNN arises from the fact that ViT's attention mechanism computes interactions between patch tokens rather than applying dense convolutional kernels across every spatial position at every layer.

---

## 4.5 Training Procedure

### 4.5.1 Optimizer Configuration

All four models are trained with the AdamW optimiser [Loshchilov and Hutter, 2019] with a learning rate of `1e-4` and weight decay of `0.01`. AdamW is preferred over the original Adam optimiser for fine-tuning pre-trained models because of how weight decay is applied. In standard Adam, weight decay is incorporated into the gradient estimate, effectively coupling it to the adaptive learning rate scale — meaning parameters with small gradient variance (common in fine-tuning from a converged pre-trained model) are penalised more aggressively. AdamW implements weight decay in a decoupled manner as a direct subtraction from the parameter value after the gradient update, restoring the intended L2 regularisation behaviour independently of the gradient scale. This distinction is particularly important when fine-tuning from Kinetics-400 pre-trained weights, as many parameters begin near their optimal values and should be penalised conservatively.

The learning rate schedule is cosine annealing implemented via `torch.optim.lr_scheduler.CosineAnnealingLR` with `T_max` equal to the total number of training epochs. Under cosine annealing, the learning rate decays from its initial value following a half-cosine curve, reaching a minimum (near zero) at epoch `T_max`. This schedule is preferred over step-wise decay because it provides a smooth, continuous reduction in learning rate, reducing the risk of large, destabilising parameter updates in the final training epochs. For Transformer-based models such as VideoMAE, which are sensitive to large learning rates in the early training phases due to the high dimensionality and mutual dependence of attention weight matrices, a linear warmup of a few hundred gradient steps from zero to the target learning rate would typically be applied. In this study's simplified schedule, cosine decay without explicit warmup is used uniformly across all models, which represents a minor pragmatic simplification for the non-Transformer architectures.

### 4.5.2 Mixed-Precision Training

All training runs employ automatic mixed precision (AMP) via PyTorch's `torch.amp.GradScaler` and `torch.amp.autocast`. Within an `autocast` context, PyTorch automatically selects between float32 (FP32) and float16 (FP16) for individual operations according to a set of precision-safety rules: operations whose numerical stability is sensitive to precision (such as batch normalisation's running-statistics accumulation) remain in FP32, while matrix multiplications and convolutions — which dominate the computational budget and are well-conditioned — are executed in FP16.

The benefits of mixed-precision training in this experimental context are threefold. First, FP16 tensors occupy exactly half the memory of their FP32 equivalents, reducing the GPU memory footprint of activations by approximately 50% and enabling larger effective batch sizes within a fixed VRAM budget. Second, modern GPUs (including the NVIDIA L4 used in this study) contain dedicated tensor core hardware that executes FP16 matrix multiplications at 2–4 times the throughput of equivalent FP32 operations, directly accelerating the training loop. Third, for VideoMAE specifically, the large activation volume arising from the 1,568-token transformer sequence makes AMP a practical necessity: without it, fitting even a single clip in 22 GB of VRAM during the backward pass would require reducing the spatial resolution or temporal depth below their standard values.

Gradient underflow — a failure mode in which small FP16 gradient values round to zero before accumulating — is mitigated by the `GradScaler`, which multiplies the loss by a dynamically maintained scale factor before the backward pass, then divides parameter gradients by the same factor before the optimiser step. If any gradient values overflow to infinity in FP16 (indicating that the scale factor is too large), the scaler reduces it automatically; otherwise, it is gradually increased to maximise numerical precision.

### 4.5.3 Skip-Guard and Resumability

A practical challenge of conducting multi-model experiments on Google Colab (which enforces session time limits of 12 hours for Pro users) is that a single run of four models on two datasets would normally require approximately 20–30 hours of GPU time, far exceeding any single session. The pipeline addresses this through a skip-guard mechanism in `src/train.py`.

At the start of each training run, the driver checks whether a results CSV file for the specific `(model, dataset)` combination already exists at the expected path in `RESULTS_DIR` and contains at least as many rows as the requested number of training epochs. If this condition is met, the training run is skipped entirely, printing a diagnostic message. Otherwise, training proceeds normally, appending a row to the CSV at the end of each epoch.

Persistence across sessions is achieved by setting `RESULTS_DIR` to a directory on Google Drive via the `RESULTS_DIR` environment variable, which `src/config.py` reads at import time. Since Google Drive persists across Colab sessions, CSV files and checkpoints written in one session are visible to the next session, and the skip-guard prevents redundant recomputation. This design allows `run_all.py` to be re-run idempotently at the start of each new session: it will skip all already-completed training runs and resume only those that were interrupted.

### 4.5.4 Batch Size and Memory Management

A batch size of 8 was selected as the maximum that fits within the 22 GB VRAM available on the NVIDIA L4 GPU used for experiments, subject to the constraint that the largest model (R(2+1)D-18) must fit alongside its gradient and optimiser state tensors. The peak GPU memory allocated during a single forward-backward pass for R(2+1)D-18 with batch size 8 is approximately 582 MB per clip for activations, scaling to approximately 4.6 GB for a batch of 8; additional memory is required for gradients (matching the parameter count) and AdamW's first- and second-moment estimates (two copies of the parameter count). Empirically, batch size 8 is the largest power-of-two that does not trigger out-of-memory errors on the L4 for any of the four architectures.

A batch size guard is implemented in `train.py` via the `_safe_batch_size()` function, which checks the available GPU memory at runtime and caps the batch size to 8 for GPUs with fewer than 18 GB of VRAM (such as the Kaggle T4 with approximately 15 GB). This ensures that the Colab notebook can be run on Kaggle without modification in cases where a larger batch size is requested via command-line argument.

Gradient accumulation — in which gradients from multiple smaller batches are summed before a parameter update, effectively simulating a larger batch — was not used in this study, as the primary goal was simplicity and uniformity across models rather than maximising absolute accuracy.

---

## 4.6 Efficiency Measurement Protocol

Efficiency measurements are conducted by `src/benchmark.py`, which evaluates each model in inference mode (no gradient tracking) on a single GPU. All four models are measured on the same hardware (NVIDIA L4, 22 GB VRAM) under the same conditions, ensuring comparability.

**Parameter count** is computed using `fvcore.nn.parameter_count`, which counts the number of learnable scalar values across all named `nn.Module` parameters. For the `VideoMAEForVideoClassification` wrapper, this includes the patch embedding, positional embedding, all 12 transformer blocks, and the classification head.

**Computational cost (GFLOPs)** is measured using `fvcore.nn.FlopCountAnalysis`, which performs static analysis of the computational graph by tracing the model with a dummy input tensor of shape `(1, 16, 3, 224, 224)`. fvcore counts floating-point multiply-add operations for convolutional and linear layers. One known limitation is that `nn.LSTM` operations are not supported by fvcore's operator registry and are therefore silently excluded from the count, resulting in a slight undercount for the CNN-LSTM model (approximately 0.3 GFLOPs for the bidirectional LSTM over 16 frames with hidden size 512 and input dimension 2048). The reported GFLOPs for ResNet-50 TSN and CNN-LSTM are therefore identical at 65.75, with the latter's true cost slightly higher. Warnings from fvcore about unsupported operations are suppressed to avoid console clutter, but the limitation is acknowledged.

**Latency** is measured as the wall-clock time per clip for a batch of 1, averaged over a specified number of iterations. A warmup phase of 5 forward passes is performed before timing begins; these warmup iterations allow the GPU to reach thermal steady state, prime CUDA kernel caches, and complete any deferred JIT compilation. The timed measurement phase runs for `--iters` forward passes (30 in the default configuration, 100 as stated in the experimental protocol for final results); `torch.cuda.synchronize()` is called before timing begins and after it ends to ensure that all GPU kernel launches are complete and that wall-clock time accurately reflects GPU computation time rather than CPU submission time. Latency per clip is computed as `(total_wall_clock / iters) / batch_size`.

**Throughput** is the inverse of latency scaled by batch size: `batch_size / (total_wall_clock / iters)`, representing the number of clips the model can process per second.

**Peak GPU memory** is measured using `torch.cuda.max_memory_allocated()` immediately after the timed inference loop. Prior to the timing loop, `torch.cuda.reset_peak_memory_stats()` is called to reset the peak tracker to zero, ensuring that the measured peak reflects the inference pass alone rather than any prior allocations.

All efficiency measurements are performed at batch size 1, which is the standard convention for reporting per-clip inference cost in the action recognition literature. Batch inference would amortise some fixed costs (such as CUDA kernel launch overhead) across multiple clips, potentially changing the relative ordering of architectures in terms of throughput; however, single-clip latency is the relevant metric for applications requiring low-latency real-time classification.

---

## 4.7 Software Architecture and Implementation

The codebase is structured as a Python package (`src/`) with a thin driver script (`run_all.py`) that orchestrates the full experimental pipeline by invoking training, benchmarking, and reporting as subprocess calls:

```
F:\Dizertatie\
├── run_all.py              # driver: train→benchmark→report for all models
├── src/
│   ├── config.py           # NUM_FRAMES=16, IMG_SIZE=224, RESULTS_DIR, DATASETS dict
│   ├── datasets.py         # VideoClipDataset, build_dataset, find_data_root
│   ├── models.py           # ResNet50TSN, CNNLSTM, R2Plus1D, VideoMAE, BUILDERS
│   ├── train.py            # training loop with AMP, skip-guard, CSV logging
│   ├── benchmark.py        # fvcore FLOPs, CUDA timer, efficiency.csv
│   └── report.py           # accuracy table, Pareto scatter plot
└── notebooks/
    └── colab_run.ipynb     # thin wrapper for Google Colab Pro
```

The `BUILDERS` dictionary in `models.py` provides a registry mapping string identifiers to model constructor classes:

```python
BUILDERS = {
    "resnet50_tsn": ResNet50TSN,
    "cnn_lstm": CNNLSTM,
    "r2plus1d_18": R2Plus1D,
    "videomae": VideoMAE,
}
```

The `build_model(name, num_classes)` function is the single entry point used by `train.py` and `benchmark.py` to instantiate any model by name. Adding a new architecture to the comparison requires only implementing a `(B,T,C,H,W) → (B,C)` forward method and adding an entry to `BUILDERS` — no changes to training, benchmarking, or reporting code are required.

Dataset auto-detection is handled by `find_data_root()`, which scans the Kaggle `/kaggle/input/` directory and scores each subdirectory according to the presence of official split files (`classInd.txt`), the presence of video files, and keyword matches in the directory name. This heuristic reliably selects the correct dataset directory in Kaggle's multi-dataset input environment without requiring manual path configuration.

Reproducibility is ensured through several mechanisms. Random seeds are fixed at the dataset split stage: the HMDB51 fallback split uses `random.Random(42)`, a seeded instance that does not affect the global random state, and the seed is fixed to 42 uniformly. Frame sampling and spatial transforms are deterministic given the same input clip. GPU training involves inherent non-determinism from operations such as atomicAdd in cuDNN convolutions, which cannot be avoided without substantial performance penalty; accordingly, exact numerical reproducibility of training runs is not guaranteed, but the training procedure is otherwise fixed (same hyperparameters, same data order given deterministic splits and no additional seed variation).

The `report.py` module consumes the per-model training CSV files and the consolidated `efficiency.csv` to produce a formatted accuracy table and a Pareto scatter plot (accuracy versus GFLOPs, with bubble size proportional to parameter count). The Pareto plot is the primary visualisation tool for interpreting the accuracy-efficiency trade-off across the four model families.

---

# Appendix A: Training Hyperparameter Configuration

The following table summarises all hyperparameters used across all training experiments in this study. Values are identical for all four model architectures and both datasets unless otherwise noted.

| Hyperparameter | Value | Notes |
|---|---|---|
| **Optimizer** | AdamW | Decoupled weight decay |
| **Learning rate** | 1e-4 | Applied to all parameter groups uniformly |
| **Weight decay** | 0.01 | Applied via decoupled AdamW formulation |
| **LR schedule** | Cosine annealing | `CosineAnnealingLR`, `T_max = num_epochs` |
| **Warmup** | None | Cosine annealing applied from epoch 1 |
| **Training epochs** | 10 | Same for all models and datasets |
| **Batch size** | 8 | Maximum fitting 22 GB VRAM for R(2+1)D |
| **Number of frames (T)** | 16 | Uniformly sampled per clip |
| **Spatial resolution** | 224 × 224 | After resize-to-256 and crop |
| **Resize (shorter edge)** | 256 px | Bilinear with anti-aliasing |
| **Train crop** | Random 224 × 224 | Same crop applied across all T frames |
| **Train flip** | Horizontal, p=0.5 | Same flip applied across all T frames |
| **Eval crop** | Centre 224 × 224 | No random transforms at evaluation |
| **Normalisation mean** | [0.485, 0.456, 0.406] | ImageNet RGB mean |
| **Normalisation std** | [0.229, 0.224, 0.225] | ImageNet RGB std |
| **Mixed precision** | FP16 (AMP) | `torch.amp.autocast` + `GradScaler` |
| **Loss function** | Cross-entropy | Standard multi-class classification loss |
| **DataLoader workers** | 4 | Parallel data loading processes |
| **Pin memory** | True | Accelerates CPU-to-GPU transfer |
| **Drop last batch** | True (train only) | Avoids variable-size batches |
| **Random seed (splits)** | 42 | Seeded `random.Random` for HMDB51 split |
| **UCF101 split** | Split 1 | Official split (classInd + trainlist01.txt) |
| **HMDB51 split** | 70/30 random | Per-class, seed=42, due to no official files |
| **Benchmark warmup iters** | 5 | Forward passes before timing |
| **Benchmark timing iters** | 30–100 | Forward passes used for latency estimate |
| **Benchmark batch size** | 1 | Single-clip inference latency |
| **LSTM hidden size** | 512 | Per direction; 1024 total (bidirectional) |
| **LSTM layers** | 1 | Single-layer bidirectional LSTM |

---

# Appendix B: Mathematical Foundations

## B.1 LSTM Equations

The Long Short-Term Memory unit [Hochreiter and Schmidhuber, 1997] was designed to address the vanishing gradient problem in recurrent neural networks when learning long-range temporal dependencies. At each time step t, the LSTM maintains two state vectors: the cell state c_t (which carries long-range information) and the hidden state h_t (which is the output at each time step). The dynamics are governed by four gating mechanisms:

**Input gate** i_t controls which elements of the cell state to update:

$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$

**Forget gate** f_t controls which elements of the previous cell state to retain:

$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$

**Output gate** o_t controls which elements of the cell state to expose as the hidden state:

$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$

**Candidate cell update** g_t (sometimes written as c̃_t) computes new candidate values for the cell state:

$$g_t = \tanh(W_g x_t + U_g h_{t-1} + b_g)$$

The cell state and hidden state are then updated as:

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$

$$h_t = o_t \odot \tanh(c_t)$$

where σ denotes the sigmoid activation function, ⊙ denotes element-wise multiplication, W_i, W_f, W_o, W_g are weight matrices applied to the current input x_t, and U_i, U_f, U_o, U_g are recurrent weight matrices applied to the previous hidden state h_{t-1}. The biases b_i, b_f, b_o, b_g complete the affine transformation.

The forget gate f_t is the critical innovation enabling long-range memory: when f_t ≈ 1 for some dimension, the corresponding element of the cell state c_{t-1} is preserved unchanged; when f_t ≈ 0, it is erased. This gated mechanism allows the network to selectively retain information over many time steps, alleviating the exponential gradient decay that afflicts vanilla recurrent networks.

**Bidirectional LSTM**: In the bidirectional variant [Schuster and Paliwal, 1997], two separate LSTM networks process the input sequence in opposite temporal directions:

- The forward LSTM processes the sequence x_1, x_2, ..., x_T and produces forward hidden states h_1^→, h_2^→, ..., h_T^→.
- The backward LSTM processes the sequence x_T, x_{T-1}, ..., x_1 and produces backward hidden states h_T^←, h_{T-1}^←, ..., h_1^←.

At each time step t, the bidirectional representation is the concatenation:

$$\tilde{h}_t = [h_t^{\rightarrow}; h_t^{\leftarrow}]$$

In this study, hidden_size = 512 per direction, yielding ‖h̃_t‖ = 1024. The bidirectional representation is beneficial for offline video classification because future frames provide contextual evidence that can disambiguate the interpretation of past frames — for example, identifying that an ambiguous arm motion is part of a throwing action rather than a catching action requires observing the subsequent trajectory.

## B.2 (2+1)D Convolution Factorisation

A standard 3D convolution with kernel size d×d×t, M_i input channels, and M_o output channels applied to a spatiotemporal volume of size H×W×L (height, width, temporal length) has a computational cost proportional to:

$$\text{FLOPs}_{3D} \propto d \cdot d \cdot t \cdot M_i \cdot M_o \cdot H \cdot W \cdot L$$

The R(2+1)D decomposition replaces this with two successive operations. First, a 2D spatial convolution with kernel d×d×1, M_i input channels, and M_d intermediate channels:

$$\text{FLOPs}_{spatial} \propto d \cdot d \cdot 1 \cdot M_i \cdot M_d \cdot H \cdot W \cdot L$$

Second, a 1D temporal convolution with kernel 1×1×t, M_d intermediate channels, and M_o output channels:

$$\text{FLOPs}_{temporal} \propto 1 \cdot 1 \cdot t \cdot M_d \cdot M_o \cdot H \cdot W \cdot L$$

The total cost of the factorised operation is:

$$\text{FLOPs}_{(2+1)D} \propto (d^2 \cdot M_i \cdot M_d + t \cdot M_d \cdot M_o) \cdot H \cdot W \cdot L$$

Comparing with the monolithic 3D convolution, the ratio of factorised to full cost is:

$$\frac{\text{FLOPs}_{(2+1)D}}{\text{FLOPs}_{3D}} = \frac{d^2 M_i M_d + t M_d M_o}{d^2 t M_i M_o}$$

Tran et al. (2018) choose M_d such that the two operations have approximately equal parameter counts, setting M_d = floor(t · M_i · M_o · d² / (M_i · d² + t · M_o)). Under this choice, the factorised version uses approximately the same number of parameters as the full 3D convolution, but achieves a different (and empirically superior) inductive bias through the separate spatial-temporal treatment.

The critical advantage identified by Tran et al. is not parameter reduction but the additional non-linearity. The (2+1)D decomposition places a ReLU activation between the spatial convolution and the temporal convolution:

spatial\_conv → **ReLU** → temporal\_conv

A full 3D convolution contains no such intermediate non-linearity and thus has a strictly linear input-output relationship within each convolutional block (before the block's own output non-linearity). The intermediate ReLU in the (2+1)D decomposition doubles the number of non-linear transformations within the equivalent computational budget, increasing the function class expressible by the network. Empirically, this results in lower training error and consistently better generalisation on video benchmarks.

## B.3 Multi-Head Self-Attention

The self-attention mechanism [Vaswani et al., 2017] computes pairwise interactions between all positions in a sequence simultaneously. Given an input sequence represented as a matrix X ∈ ℝ^{n×d_model} (n tokens, each of dimension d_model), queries, keys, and values are computed by linear projections:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

where W_Q, W_K, W_V ∈ ℝ^{d_model × d_k} and d_k is the per-head key/query dimension. The scaled dot-product attention is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

The scaling factor 1/√d_k is applied before the softmax because the dot products Q K^T grow in magnitude with d_k (since they are sums of d_k products of unit-variance values). Without scaling, large dot products push the softmax into regions of very small gradient, effectively preventing the attention from being uniformly distributed — a pathological condition in which one or few tokens receive all attention weight. Scaling by 1/√d_k restores the dot products to a unit-variance regime, stabilising gradients through the softmax.

Multi-head attention computes h independent attention functions in parallel, each operating in a lower-dimensional subspace:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

where:

$$\text{head}_i = \text{Attention}(Q W_i^Q, \, K W_i^K, \, V W_i^V)$$

and W_i^Q, W_i^K, W_i^V ∈ ℝ^{d_model × d_k}, W_O ∈ ℝ^{h d_k × d_model}. With d_model = 768 and h = 12 attention heads, each head operates in d_k = 64 dimensions. The multiple heads allow the model to attend to different aspects of the sequence simultaneously: some heads may specialise in local spatial relationships within a frame, others in long-range temporal dependencies across frames.

In VideoMAE, the input to the self-attention layers consists of spatiotemporal patch tokens derived from the 16-frame input. With a temporal patch size of 2 frames and a spatial patch size of 16×16, the 16 frames at 224×224 yield `(16/2) × (224/16) × (224/16) = 8 × 14 × 14 = 1,568` patch tokens. Each token is a linear projection of the flattened raw pixel values within its spatiotemporal patch volume (2 × 16 × 16 × 3 = 1,536 values). The ViT-Base attention operates over all 1,568 tokens jointly, allowing every spatial position in every frame to attend to every other position — enabling truly global spatiotemporal reasoning without any locally-constrained receptive field.

## B.4 Cosine Annealing Learning Rate Schedule

The cosine annealing schedule [Loshchilov and Hutter, 2017] decays the learning rate from an initial maximum value lr_max to a minimum value lr_min following a half-cosine curve. At training step t (with T_max total steps), the learning rate is:

$$\text{lr}(t) = \text{lr}_{\min} + \frac{1}{2}(\text{lr}_{\max} - \text{lr}_{\min})\left(1 + \cos\!\left(\frac{\pi t}{T_{\max}}\right)\right)$$

At t = 0, cos(0) = 1, so lr(0) = lr_max. At t = T_max, cos(π) = -1, so lr(T_max) = lr_min. Between these endpoints, the learning rate follows a smooth S-shaped decline: it decreases slowly at first (when cos is near its maximum and changes slowly), then more rapidly through the middle of training, and slowly again near the end. This profile prevents large, destabilising updates in the final epochs while maintaining adequate gradient step sizes in the middle of training.

In this study, lr_max = 1e-4, lr_min ≈ 0 (the default PyTorch implementation sets lr_min to 0 unless the `eta_min` parameter is specified), and T_max = 10 (the total number of training epochs). The schedule is applied at epoch granularity (once per epoch) via `sched.step()` called after the final batch of each epoch.

A warmup phase linearly ramps the learning rate from 0 to lr_max over the first N_warmup steps before the cosine decay begins. Warmup is particularly beneficial for Transformer-based models such as VideoMAE: at initialisation, the attention weights have large, randomly distributed values, and a full-magnitude learning rate in the first step can produce a large, poorly-directed gradient update that takes many subsequent epochs to correct. A gradual linear ramp allows the model to first find a direction of consistent gradient descent before committing to the full learning rate. In this study's simplified implementation, explicit warmup is omitted and the full learning rate is applied from epoch 1; this is a pragmatic simplification that may slightly reduce the final accuracy of VideoMAE but does not affect the relative comparison between architectures.

---

# Appendix C: Key Source Code Listings

## C.1 Model Architecture Definitions

The `CNNLSTM` class, reproduced below with extended commentary, illustrates the design decisions made in constructing the CNN-RNN architecture:

```python
class CNNLSTM(nn.Module):
    """CNN-RNN (LRCN, Donahue et al. 2015): per-frame ResNet-50 features fed to a
    bidirectional LSTM over time, then classified.

    Shares the SAME ResNet-50 backbone as ResNet50TSN on purpose: TSN averages the
    per-frame features, this models their temporal order with recurrence -- so the
    pair isolates exactly what recurrence adds over naive temporal pooling.
    """

    def __init__(self, num_classes, hidden=512, layers=1):
        super().__init__()
        # Use the same ImageNet-pretrained ResNet-50 as the TSN baseline.
        # This is the controlled comparison: backbone is identical; only the
        # temporal aggregation mechanism differs.
        self.backbone = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )
        feat_dim = self.backbone.fc.in_features  # 2048: dimension of penultimate layer

        # Replace the classification head with an identity mapping so the backbone
        # outputs a 2048-dim feature vector rather than class logits.
        # nn.Identity() passes its input through unchanged; this avoids any change
        # to the backbone's forward computation graph.
        self.backbone.fc = nn.Identity()

        # Bidirectional LSTM: processes the feature sequence in both temporal
        # directions simultaneously. hidden=512 per direction; output dim = 1024
        # after forward and backward hidden states are concatenated.
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True)

        # Final linear classifier: maps from 1024-dim (hidden*2 for bidirectional)
        # to num_classes output logits.
        self.fc = nn.Linear(hidden * 2, num_classes)  # *2: bidirectional

    def forward(self, x):                               # (B, T, C, H, W)
        b, t, c, h, w = x.shape

        # Efficient frame feature extraction: reshape to (B*T, C, H, W) so all
        # T frames of all B clips are processed in a single CNN forward pass,
        # exploiting GPU parallelism. Logically equivalent to a loop over frames.
        feats = self.backbone(x.reshape(b * t, c, h, w))  # (B*T, 2048)

        # Reshape back to per-clip sequences: (B, T, 2048).
        # This is the input sequence that the LSTM will process.
        feats = feats.view(b, t, -1)

        # LSTM forward pass over the T-frame sequence.
        # out shape: (B, T, 2*hidden) = (B, T, 1024)
        # The second return value (h_n, c_n) contains the final hidden/cell states;
        # these are discarded here as we pool over all time steps.
        out, _ = self.lstm(feats)                          # (B, T, 1024)

        # Temporal mean pooling: average the LSTM outputs across all T time steps
        # to obtain a single 1024-dim clip representation.
        # Alternative: take only the final hidden state out[:, -1, :], but
        # mean pooling is more robust when frame count varies or clips are padded.
        return self.fc(out.mean(dim=1))                    # (B, num_classes)
```

## C.2 Data Pipeline: Frame Sampling

The `_uniform_indices` function provides the shared temporal sampling logic for both video files and frame-folder datasets:

```python
def _uniform_indices(total, n):
    """n frame indices spread uniformly across a clip of `total` frames."""
    if total <= 0:
        # Degenerate case: clip reports zero or negative length (corrupt file).
        # Return n copies of index 0; the corrupt-clip handler in __getitem__
        # will catch the subsequent read failure.
        return [0] * n
    if total < n:
        # Clip is shorter than the requested number of frames.
        # Take all available frames and repeat the last frame to reach length n.
        # This avoids invalid index access and preserves the clip's true content.
        pad = [total - 1] * (n - total)
        return list(range(total)) + pad
    # Standard case: n < total. Use np.linspace to generate n float values
    # equally spaced from 0 to (total-1) inclusive, then round to integers.
    # This produces indices that span the full temporal extent of the clip,
    # from the first frame to the last frame, with even spacing between samples.
    return [int(round(x)) for x in np.linspace(0, total - 1, n)]
```

With `total = 250` (a typical UCF101 clip of 10 seconds at 25 fps) and `n = 16`, `np.linspace(0, 249, 16)` produces the values `[0.0, 16.6, 33.2, 49.9, 66.5, 83.2, 99.8, 116.4, 133.1, 149.7, 166.4, 182.9, 199.7, 216.3, 232.9, 249.0]`, which round to `[0, 17, 33, 50, 67, 83, 100, 116, 133, 150, 166, 183, 200, 216, 233, 249]`. These 16 indices are approximately evenly spaced across the 250-frame clip, sampling roughly one frame every 15.6 frames (every 0.63 seconds at 25 fps).

## C.3 Data Pipeline: Dual-Format Loader

The `VideoClipDataset.__getitem__` method demonstrates the pipeline's ability to handle both video files and pre-extracted frame directories transparently:

```python
def __getitem__(self, i):
    rel, label = self.samples[i]
    path = self.videos_root / rel

    try:
        if path.is_dir():
            # HMDB51 RawFrames path: the clip "file" is actually a directory
            # containing sorted JPEG images (one per frame).
            # _load_frames_from_dir handles uniform sampling from the sorted
            # file list and returns (T, C, H, W) float tensor in [0, 1].
            clip = _load_frames_from_dir(path, NUM_FRAMES)
        else:
            # UCF101 / standard video path: use decord for random-access
            # frame decoding without reading the full video stream.
            vr = VideoReader(str(path), ctx=cpu(0))
            idx = _uniform_indices(len(vr), NUM_FRAMES)
            # get_batch decodes only the requested frames (with decord's
            # torch bridge enabled, returns a (T, H, W, C) uint8 tensor).
            frames = vr.get_batch(idx)
            # Permute from decord's (T, H, W, C) to torchvision's (T, C, H, W)
            # and scale uint8 [0, 255] to float [0, 1].
            clip = frames.permute(0, 3, 1, 2).float() / 255.0

        # Apply the shared spatial transform (resize, crop, flip, normalise).
        # self.train determines whether random or deterministic transforms are used.
        return _transform(clip, self.train), label

    except Exception as e:
        # Tolerate individual corrupt or unreadable clips by returning a
        # black (all-zero) clip with correct normalisation applied.
        # A warning is printed only on the first failure per dataset instance
        # to avoid flooding the log.
        if not self._warned:
            print(f"[datasets] failed to read {path}: {e} (using black clip)")
            self._warned = True
        black = torch.zeros(NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE)
        return TF.normalize(black, MEAN, STD), label
```

The dual-format detection is a simple `path.is_dir()` check: if the path constructed from `videos_root / relative_path` is a directory, the clip is in RawFrames format; if it is a file, decord is used. This design means that the same `VideoClipDataset` class and the same `build_dataset()` call correctly handles both UCF101 video files and HMDB51 frame directories without any dataset-specific branching in the caller.

## C.4 Training Loop (Abbreviated)

The following excerpt shows the core training loop structure from `src/train.py`, illustrating the AMP integration, skip-guard check, and CSV logging:

```python
# --- Skip-guard: check if this (model, dataset) training run is already done ---
log_path = RESULTS_DIR / f"{args.model}_{args.dataset}_train.csv"
if log_path.exists():
    done = max(0, sum(1 for _ in open(log_path)) - 1)  # subtract header row
    if done >= args.epochs:
        print(f"[skip] {log_path.name} already has {done} epochs logged; skipping.")
        return  # exit train.py main() immediately; driver moves to next model

# --- Optimizer and AMP setup ---
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
criterion = torch.nn.CrossEntropyLoss()
# GradScaler manages the dynamic loss scale for FP16 gradient stability.
# enabled=False on CPU (no AMP benefit on CPU; avoids overhead).
scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

# --- Training loop ---
for epoch in range(1, args.epochs + 1):
    model.train()
    running_loss = 0.0

    for clips, labels in train_loader:
        clips, labels = clips.to(device), labels.to(device)
        opt.zero_grad()

        # autocast automatically selects FP16 for matmuls/convolutions
        # and keeps FP32 for numerically sensitive operations.
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(clips)               # (B, num_classes)
            loss = criterion(logits, labels)    # scalar

        # Scale the loss before backward pass to prevent FP16 gradient underflow.
        scaler.scale(loss).backward()
        # Unscale gradients, then step (skips step if any gradient is inf/nan).
        scaler.step(opt)
        # Update the scale factor for the next iteration.
        scaler.update()
        running_loss += loss.item() * clips.size(0)

    sched.step()  # advance cosine annealing schedule by one epoch

    # Evaluate on test set (no_grad is set inside evaluate_model).
    acc1, acc5 = evaluate_model(model, test_loader, device)
    train_loss = running_loss / len(train_dataset)

    # Append results to CSV (creates file with header on first write).
    _append_csv(log_path,
                ["epoch", "train_loss", "top1", "top5", "sec"],
                [epoch, f"{train_loss:.4f}", f"{acc1:.2f}", f"{acc5:.2f}", f"{elapsed:.0f}"])

    # Save best checkpoint.
    if acc1 > best_acc:
        best_acc = acc1
        torch.save(model.state_dict(),
                   CHECKPOINT_DIR / f"{args.model}_{args.dataset}_best.pth")
```

The AMP context manager (`autocast`) and the gradient scaler (`GradScaler`) are applied uniformly to all four architectures without any model-specific modifications. The skip-guard at the top prevents re-running expensive training jobs in a resumed Colab session, enabling fault-tolerant multi-day experiments across session boundaries.


---

# Chapter 5: Theoretical and Experimental Results

## 5.1 Experimental Setup Summary

### 5.1.1 Hardware and Software Environment

All experiments were conducted on a single NVIDIA L4 GPU with 22 GB of VRAM, accessed through Google Colab Pro. The L4 is a data-center-grade Ada Lovelace GPU optimized for inference and training workloads, providing a consistent and reproducible compute environment across all model evaluations. No multi-GPU parallelism was employed; all models were trained and evaluated on a single device to ensure comparability of latency and throughput measurements.

The software stack comprised PyTorch 2.x as the primary deep learning framework, HuggingFace Transformers 4.40+ for the VideoMAE implementation, torchvision for ResNet-50, R(2+1)D-18, and data pipeline utilities, decord for efficient video decoding, and fvcore for computational cost profiling (GFLOPs and parameter counts). Mixed-precision training (FP16) was enabled via PyTorch's automatic mixed precision (AMP) facility [35] to reduce memory pressure and accelerate training without affecting convergence.

### 5.1.2 Datasets

Two benchmark datasets were used throughout the study.

**UCF101** [24] is a widely used action recognition benchmark comprising 13,320 video clips distributed across 101 action categories collected from YouTube. The dataset covers a broad range of activity types including sports, musical instruments, body motions, and human-object interactions. For this study, the standard 70/30 train/test split was applied, yielding approximately 9,537 training clips and 3,783 test clips.

**HMDB51** [25] is a more challenging benchmark consisting of 6,766 video clips drawn from 51 action classes, collected from digitized movies, the Prelinger archive, and YouTube. Its difficulty arises from higher inter-class similarity, greater intra-class variation, camera motion, and lower clip quality relative to UCF101. A 70/30 random split was applied, as the official HMDB51 three-fold cross-validation splits were not used in this study (this is acknowledged as a limitation in Section 6.3).

### 5.1.3 Hyperparameter Uniformity

A fundamental methodological requirement of this comparative study is that all four models are evaluated under strictly identical training conditions, so that any observed performance differences can be attributed to architectural choices rather than tuning disparities. Table 5.1 summarizes the shared hyperparameter configuration.

**Table 5.1: Shared hyperparameter configuration for all models and both datasets.**

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW [34] |
| Learning rate | 1 Ã— 10â»â´ |
| Weight decay | 1 Ã— 10â»Â² |
| LR scheduler | Cosine annealing with warmup |
| Training epochs | 10 |
| Batch size | 8 clips |
| Input frames per clip | 16 |
| Spatial resolution | 224 Ã— 224 |
| Input normalization | ImageNet mean/std |
| Mixed precision | FP16 via PyTorch AMP |

The same frame sampling strategy (uniform temporal sampling across the full clip duration) was applied to all models. No test-time augmentation (TTA) was performed; all evaluation used single-clip, center-crop inference. This conservative evaluation protocol slightly disadvantages models such as VideoMAE that typically benefit from multi-clip ensembling, but it allows direct latency and throughput comparisons under identical conditions.

### 5.1.4 Pretrained Weight Initialization

All models were initialized from publicly available pretrained weights rather than trained from scratch, consistent with the modern standard of transfer learning for action recognition. The specific pretraining sources are:

- **ResNet-50 TSN**: ResNet-50 backbone pretrained on ImageNet-1K [2], with the temporal segment network (TSN) head trained from random initialization.
- **ResNet-50 + LSTM**: ResNet-50 backbone pretrained on ImageNet-1K, with the LSTM and fully connected layers trained from random initialization.
- **R(2+1)D-18**: Pretrained on the Kinetics-400 video dataset [26] using the torchvision model zoo checkpoint.
- **VideoMAE**: Pretrained on Kinetics-400 with masked autoencoder self-supervised pre-training [22], using the `MCG-NJU/videomae-base-finetuned-kinetics` checkpoint from HuggingFace.

The asymmetry in pretraining source (ImageNet for CNN-based models versus Kinetics for 3D-CNN and Transformer) is a known confound that is analyzed in Section 5.5 and acknowledged in the limitations (Section 6.3). This choice reflects real-world constraints: Kinetics pretrained ResNet-50 checkpoints suitable for TSN-style temporal segment aggregation are not standardly available in the torchvision or HuggingFace ecosystems, and forcing all models to ImageNet pretraining would disadvantage the 3D architectures that inherently require video pretraining to reach their published performance levels.

### 5.1.5 Efficiency Measurement Protocol

Computational efficiency was measured along four dimensions. **Parameter count** and **GFLOPs per clip** were computed using fvcore's `FlopCountAnalysis` and `parameter_count` utilities with a dummy input tensor of shape (1, 3, 16, 224, 224). A known limitation is that fvcore does not register a FLOP handler for PyTorch's `nn.LSTM` module; consequently, the reported 65.75 GFLOPs for ResNet-50 + LSTM reflects only the ResNet-50 backbone. The LSTM computation adds approximately 0.3 GFLOPs (estimated analytically), which is negligible relative to the backbone and does not affect any ranking or conclusion. **Inference latency** (milliseconds per clip) and **throughput** (clips per second) were measured by timing 100 forward passes on the GPU after a 10-step warmup, using `torch.cuda.synchronize()` barriers to ensure accurate wall-clock measurement. **Peak GPU memory** was recorded via `torch.cuda.max_memory_allocated()` during a single forward pass.

---

## 5.2 Accuracy Results â€” UCF101

### 5.2.1 Results Table

Table 5.2 presents the top-1 accuracy and efficiency profile of all four models on UCF101 after 10 epochs of fine-tuning.

**Table 5.2: UCF101 results (101 classes, 10 epochs, batch 8, NVIDIA L4 GPU).**

| Model | Top-1 (%) | Params (M) | GFLOPs/clip | Latency (ms) | Throughput (clips/s) | Peak Mem (MB) |
|---|---|---|---|---|---|---|
| VideoMAE (Transformer) | 98.14 | 86.28 | 135.17 | 51.85 | 19.28 | 432 |
| R(2+1)D-18 (3D-CNN) | 98.00 | 31.33 | 162.58 | 45.29 | 22.08 | 582 |
| ResNet-50 + LSTM (CNN-RNN) | 97.91 | 34.11 | 65.75* | 19.33 | 51.72 | 323 |
| ResNet-50 TSN (2D-CNN) | 97.59 | 23.71 | 65.75 | 18.39 | 54.39 | 281 |

*fvcore LSTM undercount; actual cost approximately 0.3 GFLOPs higher (see Section 5.1.5).

### 5.2.2 Saturation Phenomenon and Architectural Indistinguishability

The defining feature of the UCF101 results is the remarkable compression of accuracy scores into a 0.55 percentage-point band (97.59%â€“98.14%). Despite spanning four architecturally distinct paradigms â€” from a simple 2D-CNN temporal segment average to a masked-autoencoder Vision Transformer â€” the models achieve essentially identical accuracy on this benchmark. The absolute gap between the best model (VideoMAE, 98.14%) and the worst (TSN, 97.59%) is smaller than the variance one would expect from re-running a single model with a different random seed.

This finding points directly to a saturation phenomenon driven by the intersection of two factors. First, UCF101's 101-class taxonomy, while broad, contains many visually distinctive activities (archery, billiards, hammering) where a single well-chosen frame is sufficient for classification. Strong single-frame classifiers based on ImageNet-pretrained CNNs have been known to achieve above 95% on UCF101 for nearly a decade [11], leaving little room for temporal modeling to contribute additional signal. Second, and more critically, all four models in this study benefit from substantial transfer from large-scale pretraining â€” either ImageNet (1.28 million images) or Kinetics-400 (up to 306,245 video clips). The combined representational capacity of modern pretrained backbones effectively bridges the distributional gap to UCF101, rendering the benchmark unable to discriminate between architectures.

From a methodological standpoint, these UCF101 results illustrate a broader lesson for the field: benchmark saturation invalidates performance-based architecture comparison. When all models cluster within the test-set measurement noise, observed rankings are unreliable. UCF101 should be considered a validation checkpoint â€” a confirmation that a model is correctly integrated and fine-tuning is working â€” rather than a discriminative evaluation surface for architecture comparison under modern pretraining regimes.

### 5.2.3 Comparison with Published Results

The obtained results are consistent with the published literature. VideoMAE-Base fine-tuned on UCF101 has been reported at approximately 96.1â€“98.9% across various implementations and fine-tuning settings [22], and the 98.14% obtained here falls comfortably within that range. For R(2+1)D-18, torchvision's pretrained model has been reported at 96.2â€“97.8% on UCF101 with full-resolution evaluation [15]; the 98.00% achieved here reflects the additional benefit of per-experiment fine-tuning rather than zero-shot transfer from Kinetics. For TSN with a ResNet-50 backbone and ImageNet pretraining, the original TSN paper [10] reports 94.2% on UCF101 using optical flow in addition to RGB; RGB-only results in the 97â€“98% range are consistent with subsequent re-implementations using stronger backbones and longer pretraining. The results therefore provide a sound empirical basis for subsequent analysis.

### 5.2.4 Convergence Behavior

All four models converged rapidly under the 10-epoch protocol, a direct consequence of strong pretrained initialization. Models initialized from Kinetics-400 (R(2+1)D, VideoMAE) typically reached above 95% validation accuracy within the first two epochs, as the pretrained features already encode video-relevant representations. Models initialized from ImageNet (TSN, CNN-LSTM) required two to three additional epochs to reach a comparable plateau, consistent with the greater distribution shift from still images to video clips. By epoch 5, all four models had reached their approximate plateau, suggesting that 10 epochs is sufficient under this fine-tuning regime and that additional epochs would not substantially alter the rankings. This convergence pattern also implies that the 10-epoch results fairly reflect the architectures' fine-tuning efficiency under the given hyperparameters.

---

## 5.3 Accuracy Results â€” HMDB51

### 5.3.1 Results Table

Table 5.3 presents the top-1 accuracy and efficiency profile of all four models on HMDB51 after 10 epochs of fine-tuning.

**Table 5.3: HMDB51 results (51 classes, 10 epochs, batch 8, NVIDIA L4 GPU).**

| Model | Top-1 (%) | Params (M) | GFLOPs/clip | Latency (ms) | Throughput (clips/s) | Peak Mem (MB) |
|---|---|---|---|---|---|---|
| VideoMAE (Transformer) | 85.48 | 86.28 | 135.17 | 51.56 | 19.39 | 432 |
| R(2+1)D-18 (3D-CNN) | 79.63 | 31.33 | 162.58 | 45.25 | 22.10 | 581 |
| ResNet-50 + LSTM (CNN-RNN) | 75.05 | 34.05 | 65.75* | 19.41 | 51.52 | 323 |
| ResNet-50 TSN (2D-CNN) | 74.27 | 23.61 | 65.75 | 18.46 | 54.16 | 281 |

*fvcore LSTM undercount; see Section 5.1.5.

### 5.3.2 Paradigm Separation and the Four-Tier Hierarchy

In sharp contrast to the UCF101 results, HMDB51 produces a clear four-tier accuracy hierarchy that spans 11.21 percentage points from top to bottom. The ranking â€” VideoMAE (85.48%) >> R(2+1)D-18 (79.63%) > CNN-LSTM (75.05%) â‰ˆ TSN (74.27%) â€” maps precisely onto the temporal modeling complexity of each paradigm: attention-based full spatiotemporal modeling, then factored spatiotemporal convolution, then sequential recurrence over frame features, then independent frame averaging. The gaps are structurally meaningful: VideoMAE leads R(2+1)D by 5.85 percentage points, R(2+1)D leads CNN-LSTM by 4.58 points, and CNN-LSTM leads TSN by only 0.78 points.

This hierarchy supports the central thesis: architectural choices for temporal modeling matter substantially, but their importance is only observable on benchmarks that genuinely require temporal reasoning. The 5.85-point advantage of VideoMAE over R(2+1)D is particularly noteworthy because it establishes that attention-based temporal modeling â€” which can in principle relate any two frames across the entire clip regardless of temporal distance â€” provides a material advantage over locally factored 3D convolutions when the task requires understanding subtle distinctions between similar actions. HMDB51 contains classes such as "climb" versus "climb stairs," "run" versus "walk," and "punch" versus "hit," where the distinguishing information is distributed across the full clip duration and cannot be captured by local 3D convolution windows alone.

### 5.3.3 Why HMDB51 Is the Discriminative Benchmark

HMDB51's discriminative power relative to UCF101 arises from three structural properties. First, its 51 classes are densely packed in semantic and visual space: many pairs of classes share the same body parts, motion types, and scene backgrounds, requiring fine-grained temporal cues to distinguish them. Second, the dataset was collected from diverse sources including digitized films and archival footage, introducing realistic variation in camera motion, video quality, lighting, and viewpoint â€” conditions that prevent any single pretrained appearance representation from generalizing perfectly. Third, with only 6,766 clips total (approximately 133 clips per class on average), the dataset provides limited training signal per class, which amplifies the importance of pretraining quality and temporal modeling capacity: models that cannot leverage temporal information effectively cannot compensate with additional appearance supervision.

These properties make HMDB51 the appropriate surface for the primary conclusions of this study. The 11.21-point accuracy range across four architectures provides statistically meaningful differentiation and validates the hypothesis that temporal modeling complexity correlates with action recognition performance on sufficiently challenging data.

### 5.3.4 Comparison with Published Results

The HMDB51 results align well with the published literature. For VideoMAE-Base, Tong et al. [22] report 86.6% on HMDB51 with Kinetics-400 pretraining and multi-clip evaluation; the 85.48% obtained here with single-clip evaluation is consistent, with the small gap attributable to the single-clip protocol's disadvantage relative to ensembled multi-clip inference. For R(2+1)D-18, published results vary substantially with pretraining source: Tran et al. [15] report approximately 73% when training from scratch on HMDB51 and approximately 74â€“76% with Sports-1M pretraining; Kinetics-400 pretraining as used here pushes the ceiling higher, and the 79.63% result is at the upper end of the expected range with Kinetics pretraining. For TSN with RGB input and ResNet-50 backbone, the original TSN paper [10] reports 69.4% on HMDB51 with optical flow and only approximately 66% with RGB alone; the 74.27% obtained here reflects the stronger ImageNet pretraining available in more recent torchvision checkpoints compared to those available in 2016. The CNN-LSTM result (75.05%) is consistent with the LRCN-style architecture results reported by Donahue et al. [9], adjusted for the stronger ResNet-50 backbone and more recent ImageNet checkpoint.

---

## 5.4 Efficiency Analysis

### 5.4.1 Computational Cost: GFLOPs per Clip

The GFLOPs analysis reveals a counterintuitive ordering. R(2+1)D-18, despite having the fewest parameters among the video-pretrained models and less than half the parameters of VideoMAE, is the most computationally expensive architecture at 162.58 GFLOPs per clip. VideoMAE requires 135.17 GFLOPs, placing it 17% below R(2+1)D despite its substantially larger parameter count and superior accuracy. TSN and CNN-LSTM share a FLOP count of 65.75 GFLOPs (excluding the negligible LSTM contribution), reflecting the fact that both process each of the 16 frames through the same ResNet-50 backbone independently.

The high GFLOPs of R(2+1)D arise from the architecture's use of 3D feature maps throughout the network. Factored spatiotemporal convolutions â€” a (1Ã—dÃ—d) spatial convolution followed by a (tÃ—1Ã—1) temporal convolution â€” must be applied to full-resolution 3D feature tensors at each stage. Although the factorization reduces the parameter count relative to full 3D convolutions, the number of multiply-add operations remains high because the feature maps retain a temporal dimension throughout the forward pass. This architectural property explains the seemingly anomalous result that a model with 31.33 million parameters consumes more GFLOPs than one with 86.28 million: parameter count and FLOP count measure fundamentally different quantities, and the relationship between them is architecture-dependent.

VideoMAE's comparatively lower FLOP count reflects the efficiency of the Vision Transformer's patch-based processing. After extracting 3D spatiotemporal patches (2 frames Ã— 16Ã—16 spatial patches) and masking a large fraction of them during training, the inference-time forward pass processes a fixed sequence of 1568 patch tokens through 12 transformer blocks with multi-head self-attention and feed-forward layers. While each attention operation is quadratic in sequence length, the relatively modest sequence length (compared to dense per-pixel processing) keeps the total FLOPs lower than the repeated 3D convolution operations in R(2+1)D.

### 5.4.2 Inference Latency and Throughput

Latency measurements show a consistent two-tier structure. TSN and CNN-LSTM achieve latencies of 18.39 ms and 19.33 ms respectively, yielding throughputs of 54.39 and 51.72 clips/second. R(2+1)D and VideoMAE are substantially slower at 45.29 ms and 51.85 ms respectively, yielding throughputs of 22.08 and 19.28 clips/second. The 2D-CNN-based models therefore offer approximately 2.7Ã— higher throughput than their video-specialized counterparts.

The latency gap between R(2+1)D (45.29 ms) and VideoMAE (51.85 ms) is moderate (14% slower for VideoMAE), even though VideoMAE has 2.75Ã— more parameters. This reflects the high degree of parallelism in transformer attention operations on modern GPU hardware: matrix-matrix multiplications in the attention and feed-forward layers are highly amenable to tensor core acceleration, whereas the sequential 3D convolutions in R(2+1)D involve more memory-bandwidth-bound operations at smaller tile sizes.

It is worth noting that these latency figures correspond to single-clip batch-size-1 inference. In throughput-oriented deployment scenarios where clips can be batched, the relative ordering may shift, as transformer architectures with large parameter counts benefit disproportionately from batching due to their compute-bound rather than memory-bandwidth-bound profile. However, for real-time single-stream inference (the typical edge deployment scenario), the 2Ã— latency advantage of TSN and CNN-LSTM is directly actionable.

### 5.4.3 Memory Efficiency

Peak GPU memory consumption during inference follows the ordering TSN (281 MB) < CNN-LSTM (323 MB) < VideoMAE (432 MB) < R(2+1)D (581â€“582 MB). Several aspects of this ordering deserve attention.

The most striking finding is that R(2+1)D consumes substantially more memory (581 MB) than VideoMAE (432 MB) despite having 2.75Ã— fewer parameters. This inversion of the parameter-to-memory relationship is explained by the 3D feature map storage requirement. During the forward pass of R(2+1)D, intermediate activations at each convolutional layer retain the full temporal dimension, meaning that feature maps of shape (batch, channels, T, H, W) must be kept in memory simultaneously for backpropagation (during training) or for the duration of the forward pass (during inference with activation checkpointing disabled). As the network deepens, the number of channels grows while the spatial dimensions shrink, but the temporal dimension T remains at 16 until late temporal pooling, resulting in a large aggregate activation footprint. VideoMAE, by contrast, operates on a sequence of 1D patch token embeddings after the initial patch extraction step; the memory footprint is dominated by the weight matrices of the 12 transformer blocks rather than by stored activations, and it scales more favorably.

The TSN memory figure (281 MB) reflects the model's fundamental simplicity: each frame is processed independently through a ResNet-50 forward pass, the per-frame features are aggregated, and no multi-frame activations are retained simultaneously. CNN-LSTM's 42 MB additional memory over TSN (323 vs. 281 MB) comes from storing the 16 per-frame feature vectors and the LSTM hidden state, which is negligible relative to the backbone activation cost.

### 5.4.4 Parameters versus Accuracy Trade-off

Plotting parameter count against HMDB51 accuracy reveals that the four models do not lie on a simple monotonic curve. VideoMAE achieves the highest accuracy (85.48%) with the most parameters (86.28 M). R(2+1)D achieves the second-highest accuracy (79.63%) with the second-fewest parameters (31.33 M). CNN-LSTM is the third most accurate (75.05%) with 34.11 M parameters â€” marginally more than R(2+1)D. TSN is the least accurate (74.27%) with the fewest parameters (23.71 M).

This distribution illustrates that parameter count is a poor proxy for model capacity in the context of video understanding. R(2+1)D's 31.33 M parameters provide substantially more discriminative temporal modeling than CNN-LSTM's 34.11 M parameters, because the 3D convolutional architecture is inherently designed to learn joint spatiotemporal features rather than sequential appearance features. The parameter budget matters less than how those parameters are organized to capture temporal structure.

---

## 5.5 Multi-Dimensional Trade-off Analysis

### 5.5.1 Pareto Frontier Characterization

A Pareto analysis formalizes the observation that no single architecture dominates all others on all dimensions simultaneously. An architecture is Pareto-optimal if there exists no other architecture that is at least as good on all measured dimensions and strictly better on at least one. When considering the two primary trade-off axes â€” accuracy (HMDB51 Top-1) and computational cost (GFLOPs/clip) â€” the Pareto structure is as follows.

VideoMAE (85.48%, 135.17 GFLOPs) strictly Pareto-dominates R(2+1)D-18 (79.63%, 162.58 GFLOPs): VideoMAE achieves higher accuracy with fewer GFLOPs. This is a strong result because it implies that, on the accuracy-versus-compute trade-off axis, R(2+1)D-18 offers no advantage over VideoMAE. If computational budget (in GFLOPs) is the binding constraint, deploying VideoMAE is the rational choice over R(2+1)D for any GFLOPs budget above 135 GFLOPs.

TSN and CNN-LSTM (both approximately 65.75 GFLOPs, 74.27% and 75.05% respectively) define the efficient frontier at lower compute budgets: they achieve approximately 74â€“75% accuracy at less than half the GFLOPs of either VideoMAE or R(2+1)D, making them Pareto-optimal at the low-compute end of the spectrum. The transition from the efficient models to VideoMAE involves spending approximately 2Ã— more GFLOPs to gain approximately 10â€“11 accuracy points â€” a trade-off that is worthwhile in high-accuracy applications but prohibitive in compute-constrained ones.

When the Pareto analysis is extended to include latency, memory, and parameter count as additional objectives, the picture becomes more nuanced. R(2+1)D-18, while Pareto-dominated on the accuracy-vs-GFLOPs axis, becomes non-dominated when latency is considered: it achieves 45.29 ms versus VideoMAE's 51.85 ms, 31.33 M parameters versus 86.28 M, and still delivers 79.63% accuracy. For deployment scenarios constrained by model file size (e.g., on-device storage) or number of parameters (e.g., embedded systems with limited weight memory), R(2+1)D's substantially smaller parameter count is a genuine advantage.

### 5.5.2 Deployment Scenario Recommendations

Based on the multi-dimensional Pareto analysis, Table 5.4 summarizes deployment recommendations across representative real-world scenarios.

**Table 5.4: Deployment scenario recommendations.**

| Scenario | Recommended Model | Rationale |
|---|---|---|
| Maximum accuracy, unconstrained resources | VideoMAE | Best HMDB51 accuracy (85.48%), Pareto-dominant on accuracy-vs-GFLOPs |
| Server-side high-throughput batch processing | ResNet-50 TSN | 54 clips/s throughput, 65.75 GFLOPs, adequate accuracy for coarse recognition |
| Real-time single-stream, accuracy important | R(2+1)D-18 | 45 ms latency, 79.63% HMDB51, smaller model than VideoMAE |
| Edge/mobile deployment | ResNet-50 TSN | 281 MB memory, 18 ms latency, 65.75 GFLOPs, simplest deployment graph |
| Memory-constrained embedded systems | ResNet-50 TSN | Smallest peak memory (281 MB), fewest parameters (23.71 M) |
| Research: temporal modeling study | VideoMAE or R(2+1)D | Both show clear temporal modeling; large HMDB51 gap validates temporal contribution |

### 5.5.3 The Role of Pretraining Source as a Confound

A critical confound in this comparison is the asymmetry in pretraining source: TSN and CNN-LSTM are initialized from ImageNet (still images), while R(2+1)D and VideoMAE are initialized from Kinetics-400 (video clips). Kinetics-400 pretraining exposes the model to video motion patterns, camera dynamics, and appearance-motion co-occurrences that are absent from ImageNet. This pretraining advantage may account for a portion of the observed accuracy gap between the two tiers, independent of architectural differences.

Quantifying the pretraining contribution precisely would require an ablation study in which all four architectures are pretrained on the same dataset â€” ideally Kinetics-400 for all, or alternatively ImageNet for all with random temporal components. This ablation was not performed in the present study. Consequently, the 5.85-point gap between R(2+1)D (Kinetics pretrained, 79.63%) and CNN-LSTM (ImageNet pretrained, 75.05%) should be interpreted as reflecting a combination of both the architectural advantage of spatiotemporal convolutions and the data advantage of video pretraining. Similarly, part of VideoMAE's superiority over R(2+1)D may reflect the quality of the masked autoencoder pretraining on Kinetics-400 rather than the attention mechanism alone. The controlled recurrence experiment in Section 5.7 partially addresses this concern by comparing TSN and CNN-LSTM, which share the same ImageNet-pretrained backbone, thereby isolating the architectural contribution of recurrence.

---

## 5.6 Cross-Dataset Consistency

### 5.6.1 Rank Stability Across Benchmarks

One of the most important validity checks for the experimental findings is whether the accuracy ranking of architectures is stable across datasets. If the ranking reversed between UCF101 and HMDB51, it would suggest that the observed differences are driven by dataset-specific artifacts rather than fundamental architectural properties. Table 5.5 presents the accuracy rankings on both datasets.

**Table 5.5: Accuracy ranking consistency across UCF101 and HMDB51.**

| Rank | UCF101 | HMDB51 |
|---|---|---|
| 1st | VideoMAE (98.14%) | VideoMAE (85.48%) |
| 2nd | R(2+1)D-18 (98.00%) | R(2+1)D-18 (79.63%) |
| 3rd | ResNet-50 + LSTM (97.91%) | ResNet-50 + LSTM (75.05%) |
| 4th | ResNet-50 TSN (97.59%) | ResNet-50 TSN (74.27%) |

The ranking is perfectly preserved across both datasets: VideoMAE > R(2+1)D > CNN-LSTM > TSN on both UCF101 and HMDB51. This rank stability provides strong evidence that the observed hierarchy reflects genuine architectural properties rather than dataset-specific confounds. The conclusion that attention-based temporal modeling outperforms factored 3D convolution, which in turn outperforms recurrence over sparse frame features, which marginally outperforms temporal averaging, is not an artifact of the particular choice of evaluation dataset.

### 5.6.2 Gap Magnitude Difference: The Diagnostic Power of Dataset Choice

While the ranking is consistent, the magnitude of the gaps differs dramatically between the two datasets. On UCF101, the total range across all four models is 0.55 percentage points (97.59%â€“98.14%). On HMDB51, the range is 11.21 percentage points (74.27%â€“85.48%). This 20-fold amplification of the inter-model gap on HMDB51 is the empirical basis for the recommendation that HMDB51 serves as the primary discriminative benchmark in this study.

The gap amplification is not uniform across model pairs. The VideoMAEâ€“R(2+1)D gap expands from 0.14% (UCF101) to 5.85% (HMDB51). The R(2+1)Dâ€“CNN-LSTM gap expands from 0.09% to 4.58%. The CNN-LSTMâ€“TSN gap expands from 0.32% to 0.78%. Each tier boundary becomes more visible on the harder dataset, with the largest amplification occurring at the boundaries involving the most temporally complex models. This pattern is consistent with the hypothesis that temporal modeling capacity matters most when the task places genuine demands on temporal reasoning.

---

## 5.7 The Controlled Experiment: Recurrence versus Averaging

### 5.7.1 Experimental Design and Motivation

The comparison between ResNet-50 TSN and ResNet-50 + LSTM constitutes the cleanest controlled experiment in this thesis. Both models share an identical ResNet-50 backbone pretrained on ImageNet-1K, process the same 16 uniformly sampled frames at 224Ã—224 resolution, and are fine-tuned under the same hyperparameter configuration. The sole variable is the temporal aggregation mechanism: TSN averages per-frame prediction scores (or equivalently, per-frame feature vectors before the classifier), while CNN-LSTM passes the sequence of per-frame feature vectors through a two-layer bidirectional LSTM before classification. This design isolates the contribution of sequential recurrence over sparse frame features from all other confounds.

### 5.7.2 Results and Cost-Benefit Analysis

The results of this controlled comparison are summarized in Table 5.6.

**Table 5.6: Controlled recurrence experiment â€” TSN vs. CNN-LSTM.**

| Dimension | ResNet-50 TSN | ResNet-50 + LSTM | Delta (LSTM âˆ’ TSN) |
|---|---|---|---|
| UCF101 Top-1 (%) | 97.59 | 97.91 | +0.32 |
| HMDB51 Top-1 (%) | 74.27 | 75.05 | +0.78 |
| Parameters (M) | 23.71 | 34.11 | +10.40 |
| GFLOPs/clip | 65.75 | ~66.05 | ~+0.30 |
| Latency (ms) â€” UCF101 | 18.39 | 19.33 | +0.94 |
| Latency (ms) â€” HMDB51 | 18.46 | 19.41 | +0.95 |
| Peak Memory (MB) | 281 | 323 | +42 |

The recurrent model (CNN-LSTM) consistently outperforms the averaging model (TSN), but the advantage is small: +0.32% on UCF101 and +0.78% on HMDB51. The cost of this accuracy improvement is +10.4 million parameters (a 44% parameter increase), +0.95 ms latency (a 5% latency increase), and +42 MB peak memory (a 15% memory increase). On the HMDB51 axis where the accuracy gains are largest, the CNN-LSTM gains 0.78 accuracy points while spending 44% more parameters. In contrast, upgrading from CNN-LSTM to R(2+1)D-18 gains 4.58 accuracy points at a cost of approximately 3 fewer million parameters but with much higher GFLOPs and memory.

### 5.7.3 Why Sparse Temporal Sampling Limits Recurrence

The small benefit of recurrence in this experiment can be explained by the interaction between the uniform frame sampling strategy and the nature of LSTM temporal modeling. The 16 frames are sampled uniformly across the full clip duration, which for a typical 10-second UCF101 or HMDB51 clip corresponds to one frame roughly every 0.625 seconds. At this temporal resolution, consecutive sampled frames share substantial visual content (they are far apart in time relative to typical motion speeds) but are too distantly spaced to reveal the fine-grained optical flow patterns that distinguish similar actions.

LSTM architectures are most beneficial when the input sequence contains coherent temporal dependencies at the sampled time scale â€” for example, recognizing that a sequence of postures forms a throwing motion requires frames spaced at approximately 50â€“100 ms intervals. At 625 ms intervals, the per-frame ResNet features already contain sufficient appearance information for the averaging model to make a correct prediction, and the LSTM has limited additional information to extract from the sequence ordering.

This analysis is supported by findings in the broader literature. Donahue et al. [9] demonstrated that LRCN architectures provide meaningful improvements over static CNNs when applied to dense optical flow sequences or fine-grained action recognition tasks. However, subsequent work [13, 15] showed that 3D convolutions applied to dense temporal clips (typically 8â€“16 consecutive frames at 25â€“30 fps, corresponding to 0.27â€“0.64 second windows) substantially outperform recurrence-based approaches because they directly capture short-range motion patterns in the feature extraction stage rather than at the aggregation stage. The Temporal Segment Network design [10], originally intended to model long-range temporal structure through sparse sampling, is effective for recognizing activities defined by their global structure (e.g., a complete golf swing) but less suited for capturing the fine-grained motion cues that distinguish similar activities on HMDB51.

### 5.7.4 Implications for Architecture Design

The controlled experiment provides a clear design implication: the temporal aggregation mechanism is a meaningful but secondary contributor to action recognition performance when the temporal resolution of the input is low. The major accuracy gains observed in this study come from replacing the independent frame processing paradigm (both TSN and CNN-LSTM) with architectures that jointly model spatial and temporal information during feature extraction â€” either through 3D convolutions (R(2+1)D) or through spatiotemporal self-attention (VideoMAE). This finding supports the field's trajectory from LRCN-style architectures (circa 2015) through C3D and I3D-style architectures (circa 2015â€“2017) to the current generation of transformer-based models (circa 2021â€“2022), each step representing an advance in the depth and density of spatiotemporal feature learning rather than in the sophistication of temporal aggregation over independently extracted frame features.

---

# Chapter 6: Conclusions

## 6.1 Summary of Contributions

This thesis presents a controlled comparative study of four major paradigms for video-based human action recognition: 2D-CNN temporal aggregation (ResNet-50 TSN), CNN-RNN recurrence (ResNet-50 + LSTM), 3D convolutional networks (R(2+1)D-18), and masked autoencoder Vision Transformers (VideoMAE-Base). The study makes the following contributions.

**Reproducible open-source evaluation framework.** A complete, end-to-end pipeline was implemented and made publicly available, covering data loading from raw video files via decord, model fine-tuning with unified hyperparameters via PyTorch and HuggingFace Transformers, and efficiency profiling via fvcore and PyTorch CUDA timers. The framework runs on a single GPU in a cloud environment (Google Colab Pro) and supports both UCF101 and HMDB51 without modification.

**Strictly controlled four-paradigm comparison.** To the authors' knowledge, no prior published study has compared all four of these specific model families (TSN, CNN-LSTM, R(2+1)D, VideoMAE) under completely identical training conditions â€” same hardware, same hyperparameters, same frame sampling, same evaluation protocol, same number of epochs â€” across both UCF101 and HMDB51. The strict control of experimental conditions is essential for attributing performance differences to architecture rather than to training regimen.

**Pareto frontier characterization.** The study provides a multi-dimensional efficiency analysis covering parameters, GFLOPs, latency, throughput, and memory across both datasets. The finding that VideoMAE Pareto-dominates R(2+1)D-18 on the accuracy-versus-GFLOPs axis is a practically significant result for deployment decisions. The characterization of the two-tier latency/memory structure (TSN and CNN-LSTM at 18â€“19 ms and 281â€“323 MB versus R(2+1)D and VideoMAE at 45â€“52 ms and 432â€“582 MB) provides a quantitative foundation for edge-versus-server deployment decisions.

**Controlled recurrence ablation.** The TSN-versus-CNN-LSTM comparison, which holds the backbone fixed and varies only the temporal aggregation mechanism, constitutes a methodologically clean ablation that isolates the contribution of sequential recurrence from pretraining effects, backbone capacity, and other confounds. This ablation clarifies the limited marginal value of recurrence over averaging at 16 sparse frames, a finding with direct implications for practical architecture selection.

## 6.2 Key Conclusions

**The temporal modeling hierarchy is real but benchmark-dependent.** The accuracy ranking Transformer > 3D-CNN > CNN-RNN â‰ˆ 2D-CNN is perfectly consistent across both UCF101 and HMDB51. However, the hierarchy is only clearly visible on benchmarks that require genuine temporal reasoning. On UCF101, the gap between the best and worst architecture is 0.55 percentage points â€” effectively indistinguishable under single-seed evaluation. On HMDB51, the same gap is 11.21 percentage points â€” unambiguously significant. This finding has a methodological implication for the field: architecture comparisons conducted solely on UCF101 under modern pretraining regimes cannot reliably distinguish temporal modeling approaches.

**VideoMAE's spatiotemporal attention is the most effective temporal modeling approach tested.** VideoMAE achieves the highest accuracy on both datasets (98.14% UCF101, 85.48% HMDB51) while simultaneously requiring fewer GFLOPs than R(2+1)D-18 (135 vs. 163 GFLOPs). Its global self-attention mechanism, which can relate spatiotemporal patches from any two points in the clip regardless of temporal distance, provides substantially better temporal reasoning than locally factored 3D convolutions. The combination of masked autoencoder pretraining on Kinetics-400 and the transformer architecture's capacity for long-range temporal dependencies explains this superiority.

**The efficiency trade-off is multi-dimensional and deployment-context-dependent.** No single architecture dominates all others on all efficiency dimensions. VideoMAE leads on accuracy and GFLOPs but has the largest parameter count (86 M). R(2+1)D has moderate latency and the smallest parameter footprint among video-pretrained models but the highest GFLOPs and memory. TSN and CNN-LSTM offer the lowest latency (18â€“19 ms), smallest memory (281â€“323 MB), and lowest GFLOPs at the cost of approximately 10â€“11 accuracy points on HMDB51. For resource-constrained deployment, ResNet-50 TSN at 65.75 GFLOPs, 18.39 ms latency, and 281 MB memory represents a pragmatic choice, delivering 97.59% on UCF101 and 74.27% on HMDB51 from a model that requires no video pretraining.

**Recurrence adds marginally over averaging at 16 sparse frames.** The controlled TSN-vs-CNN-LSTM ablation demonstrates that adding 10.4 million parameters in the form of a bidirectional LSTM over 16 uniformly sampled frames yields only +0.32% on UCF101 and +0.78% on HMDB51. The improvement is real and consistent, but it is small relative to the cost in parameters and relative to the gains achieved by moving to 3D convolutions or attention. This result quantitatively confirms the field's empirical observation that the key to better video understanding lies in learning spatiotemporal features jointly during feature extraction, not in more sophisticated aggregation of independently extracted frame features.

**3D-CNN and Transformer pretraining on video data is a substantial confound.** Part of the advantage of R(2+1)D and VideoMAE over TSN and CNN-LSTM is attributable to the availability of Kinetics-400 pretrained checkpoints, which encode video-specific motion priors unavailable from ImageNet. Disentangling the pretraining contribution from the architectural contribution requires an ablation not performed in this study, and any interpretation of the inter-tier gaps should account for this confound.

## 6.3 Limitations

**Pretraining source asymmetry.** The two video-pretrained models (R(2+1)D, VideoMAE) benefit from Kinetics-400 pretraining while the two CNN-based models (TSN, CNN-LSTM) use ImageNet pretraining. This confounds architecture-versus-architecture comparisons across the two tiers. A fully controlled comparison would require either Kinetics-400 pretrained weights for all four architectures or a from-scratch training regime on a sufficiently large video dataset.

**Non-standard HMDB51 split.** The official HMDB51 evaluation protocol specifies three predefined train/test splits for three-fold cross-validation [25]. This study used a single 70/30 random split rather than the official splits, which means the reported HMDB51 numbers are not directly comparable with published results that use the official protocol. While the relative ranking of models is unlikely to change with official splits, the absolute accuracy values may differ by a few percentage points.

**Short training regime.** All models were trained for 10 epochs. While this was sufficient for convergence to a plateau under strong pretrained initialization, longer fine-tuning (30â€“50 epochs with learning rate decay) might yield higher absolute accuracy, potentially with different relative performance gaps. It is possible that some architectures benefit more from extended fine-tuning than others.

**Single-clip, no test-time augmentation.** All evaluation was performed using a single center-cropped clip. The published results for VideoMAE and R(2+1)D typically use multi-clip, multi-crop evaluation (e.g., 5 clips Ã— 3 crops = 15 views per video), which can add 1â€“3 percentage points. The single-clip protocol was chosen for latency comparability but slightly penalizes the architectures that benefit most from multi-view ensembling.

**LSTM GFLOPs undercount.** As noted in Section 5.1.5, fvcore does not register a FLOP handler for nn.LSTM, resulting in an undercount of approximately 0.3 GFLOPs for the CNN-LSTM model. While negligible in magnitude (less than 0.5% of total GFLOPs), this should be noted in any formal benchmarking context.

**Single random seed.** All experiments were conducted with a single random seed. Given the small accuracy differences observed (particularly on UCF101), results averaged over multiple seeds with standard deviation reporting would provide a more statistically robust comparison.

## 6.4 Future Work

**Official evaluation splits.** Wiring the official UCF101 and HMDB51 split files into the evaluation pipeline and reporting results across all three official folds with mean and standard deviation would make the results directly comparable with the published literature and would improve statistical reliability.

**Pretraining equalization ablation.** The most important methodological extension is a controlled ablation in which all four architectures are fine-tuned from the same pretrained source â€” either Kinetics-400 checkpoints for all models (which would require a TSN-style Kinetics checkpoint for the ResNet-50 backbone) or a common from-scratch regime. This ablation would quantify the architecture contribution independently of pretraining source.

**Extended model family.** The current four-model comparison covers the major paradigms but omits several influential architectures. Natural extensions include SlowFast [16], which uses dual temporal pathways at different frame rates; TimeSformer [20], which applies divided space-time attention with lower computational cost than VideoMAE; Video Swin Transformer [21], which introduces local window attention for efficiency; and MViT [36], which uses multiscale attention to reduce the sequence length progressively through the network.

**Temporal action localization.** This study addresses clip-level classification, which is a simpler task than temporal action localization, where the model must also predict the start and end time of each action. Extending the framework to localization would increase practical relevance for applications such as surveillance, sports analytics, and medical procedure monitoring.

**Efficient transformer variants and deployment optimization.** For deployment research, the pipeline could be extended to evaluate efficient video transformer variants such as X3D [17] and MViT-V2, and to incorporate ONNX export, TensorRT optimization, and INT8 quantization to characterize model performance under deployment-realistic conditions. This would complete the efficiency characterization from theoretical GFLOPs to practical inference performance on target hardware.

**Longer training and learning rate search.** Training all models for 30â€“50 epochs with a proper learning rate sweep would establish whether the 10-epoch rankings are representative of the fully trained performance and whether the accuracy gaps between tiers change under extended optimization.

---

# Bibliography

[1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," *Proceedings of the IEEE*, vol. 86, no. 11, pp. 2278â€“2324, Nov. 1998.

[2] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 25, pp. 1097â€“1105, 2012.

[3] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *Proc. International Conference on Learning Representations (ICLR)*, San Diego, CA, USA, 2015.

[4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Las Vegas, NV, USA, pp. 770â€“778, 2016.

[5] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. International Conference on Machine Learning (ICML)*, Lille, France, pp. 448â€“456, 2015.

[6] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735â€“1780, Nov. 1997.

[7] K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in *Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)*, Doha, Qatar, pp. 1724â€“1734, 2014.

[8] M. Schuster and K. K. Paliwal, "Bidirectional recurrent neural networks," *IEEE Transactions on Signal Processing*, vol. 45, no. 11, pp. 2673â€“2681, Nov. 1997.

[9] J. Donahue, L. Anne Hendricks, S. Guadarrama, M. Rohrbach, S. Venugopalan, K. Saenko, and T. Darrell, "Long-term recurrent convolutional networks for visual recognition and description," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Boston, MA, USA, pp. 2625â€“2634, 2015.

[10] L. Wang, Y. Xiong, Z. Wang, Y. Qiao, D. Lin, X. Tang, and L. Van Gool, "Temporal segment networks: Towards good practices for deep action recognition," in *Proc. European Conference on Computer Vision (ECCV)*, Amsterdam, Netherlands, pp. 20â€“36, 2016.

[11] K. Simonyan and A. Zisserman, "Two-stream convolutional networks for action recognition in videos," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 27, pp. 568â€“576, 2014.

[12] S. Ji, W. Xu, M. Yang, and K. Yu, "3D convolutional neural networks for human action recognition," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 35, no. 1, pp. 221â€“231, Jan. 2013.

[13] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, "Learning spatiotemporal features with 3D convolutional networks," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, Santiago, Chile, pp. 4489â€“4497, 2015.

[14] J. Carreira and A. Zisserman, "Quo vadis, action recognition? A new model and the Kinetics dataset," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Honolulu, HI, USA, pp. 6299â€“6308, 2017.

[15] D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri, "A closer look at spatiotemporal convolutions for action recognition," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Salt Lake City, UT, USA, pp. 6450â€“6459, 2018.

[16] C. Feichtenhofer, H. Fan, J. Malik, and K. He, "SlowFast networks for video recognition," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, Seoul, South Korea, pp. 6202â€“6211, 2019.

[17] C. Feichtenhofer, "X3D: Expanding architectures for efficient video recognition," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Seattle, WA, USA, pp. 203â€“213, 2020.

[18] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Å. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 5998â€“6008, 2017.

[19] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16Ã—16 words: Transformers for image recognition at scale," in *Proc. International Conference on Learning Representations (ICLR)*, 2021.

[20] G. Bertasius, H. Wang, and L. Torresani, "Is space-time attention all you need for video understanding?" in *Proc. International Conference on Machine Learning (ICML)*, pp. 813â€“824, 2021.

[21] Z. Liu, J. Ning, Y. Cao, Y. Wei, Z. Zhang, S. Lin, and H. Hu, "Video Swin Transformer," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, New Orleans, LA, USA, pp. 3202â€“3211, 2022.

[22] Z. Tong, Y. Song, J. Wang, and L. Wang, "VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, pp. 10078â€“10093, 2022.

[23] K. He, X. Chen, S. Xie, Y. Li, P. DollÃ¡r, and R. Girshick, "Masked autoencoders are scalable vision learners," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, New Orleans, LA, USA, pp. 16000â€“16009, 2022.

[24] K. Soomro, A. Roshan Zamir, and M. Shah, "UCF101: A dataset of 101 human actions classes from videos in the wild," *arXiv preprint arXiv:1212.0402*, 2012.

[25] H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre, "HMDB: A large video database for human motion recognition," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, Barcelona, Spain, pp. 2556â€“2563, 2011.

[26] W. Kay, J. Carreira, K. Simonyan, B. Zhang, C. Hillier, S. Vijayanarasimhan, F. Viola, T. Green, T. Back, P. Natsev, M. Suleyman, and A. Zisserman, "The Kinetics human action video dataset," *arXiv preprint arXiv:1705.06950*, 2017.

[27] N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, San Diego, CA, USA, vol. 1, pp. 886â€“893, 2005.

[28] H. Wang, A. KlÃ¤ser, C. Schmid, and C.-L. Liu, "Dense trajectories and motion boundary descriptors for action recognition," *International Journal of Computer Vision*, vol. 103, no. 1, pp. 60â€“79, May 2013.

[29] I. Laptev, "On space-time interest points," *International Journal of Computer Vision*, vol. 64, no. 2â€“3, pp. 107â€“123, Sep. 2005.

[30] Y. Bengio, P. Simard, and P. Frasconi, "Learning long-term dependencies with gradient descent is difficult," *IEEE Transactions on Neural Networks*, vol. 5, no. 2, pp. 157â€“166, Mar. 1994.

[31] D. Bahdanau, K. Cho, and Y. Bengio, "Neural machine translation by jointly learning to align and translate," in *Proc. International Conference on Learning Representations (ICLR)*, San Diego, CA, USA, 2015.

[32] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)*, Minneapolis, MN, USA, pp. 4171â€“4186, 2019.

[33] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala, "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, pp. 8024â€“8035, 2019.

[34] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. International Conference on Learning Representations (ICLR)*, New Orleans, LA, USA, 2019.

[35] P. Micikevicius, S. Narang, J. Alben, G. Diamos, E. Elsen, D. Garcia, B. Ginsburg, M. Houston, O. Kuchaiev, G. Venkatesh, and H. Wu, "Mixed precision training," in *Proc. International Conference on Learning Representations (ICLR)*, Vancouver, BC, Canada, 2018.

[36] H. Fan, B. Xiong, K. Mangalam, Y. Li, Z. Yan, J. Malik, and C. Feichtenhofer, "Multiscale vision transformers," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, Montreal, QC, Canada, pp. 6824â€“6835, 2021.

[37] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. Le Scao, S. Gugger, M. Drame, Q. Lhoest, and A. M. Rush, "Transformers: State-of-the-art natural language processing," in *Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP): System Demonstrations*, Online, pp. 38â€“45, 2020.

[38] R. Goyal, S. Ebrahimi Kahou, V. Michalski, J. Materzynska, S. Westphal, H. Kim, V. Haenel, I. Fruend, P. Yianilos, M. Mueller-Freitag, F. Hoppe, C. Thurau, I. Bax, and R. Memisevic, "The 'something something' video database for learning and evaluating visual common sense," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, Venice, Italy, pp. 5842â€“5850, 2017.
