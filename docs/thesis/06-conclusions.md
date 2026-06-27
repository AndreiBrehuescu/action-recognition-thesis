# Chapter 6: Conclusions

## 6.1 Summary of Contributions

This thesis presents a controlled comparative study of four major paradigms for video-based human action recognition: 2D-CNN temporal aggregation (ResNet-50 TSN), CNN-RNN recurrence (ResNet-50 + LSTM), 3D convolutional networks (R(2+1)D-18), and masked autoencoder Vision Transformers (VideoMAE-Base). The study makes the following contributions.

**Reproducible open-source evaluation framework.** A complete, end-to-end pipeline was implemented and made publicly available, covering data loading from raw video files via decord, model fine-tuning with unified hyperparameters via PyTorch and HuggingFace Transformers, and efficiency profiling via fvcore and PyTorch CUDA timers. The framework runs on a single GPU in a cloud environment (Google Colab Pro) and supports both UCF101 and HMDB51 without modification.

**Strictly controlled four-paradigm comparison.** To the authors' knowledge, no prior published study has compared all four of these specific model families (TSN, CNN-LSTM, R(2+1)D, VideoMAE) under completely identical training conditions — same hardware, same hyperparameters, same frame sampling, same evaluation protocol, same number of epochs — across both UCF101 and HMDB51. The strict control of experimental conditions is essential for attributing performance differences to architecture rather than to training regimen.

**Pareto frontier characterization.** The study provides a multi-dimensional efficiency analysis covering parameters, GFLOPs, latency, throughput, and memory across both datasets. The finding that VideoMAE Pareto-dominates R(2+1)D-18 on the accuracy-versus-GFLOPs axis is a practically significant result for deployment decisions. The characterization of the two-tier latency/memory structure (TSN and CNN-LSTM at 18–19 ms and 281–323 MB versus R(2+1)D and VideoMAE at 45–52 ms and 432–582 MB) provides a quantitative foundation for edge-versus-server deployment decisions.

**Controlled recurrence ablation.** The TSN-versus-CNN-LSTM comparison, which holds the backbone fixed and varies only the temporal aggregation mechanism, constitutes a methodologically clean ablation that isolates the contribution of sequential recurrence from pretraining effects, backbone capacity, and other confounds. This ablation clarifies the limited marginal value of recurrence over averaging at 16 sparse frames, a finding with direct implications for practical architecture selection.

## 6.2 Key Conclusions

**The temporal modeling hierarchy is real but benchmark-dependent.** The accuracy ranking Transformer > 3D-CNN > CNN-RNN ≈ 2D-CNN is perfectly consistent across both UCF101 and HMDB51. However, the hierarchy is only clearly visible on benchmarks that require genuine temporal reasoning. On UCF101, the gap between the best and worst architecture is 0.55 percentage points — effectively indistinguishable under single-seed evaluation. On HMDB51, the same gap is 11.21 percentage points — unambiguously significant. This finding has a methodological implication for the field: architecture comparisons conducted solely on UCF101 under modern pretraining regimes cannot reliably distinguish temporal modeling approaches.

**VideoMAE's spatiotemporal attention is the most effective temporal modeling approach tested.** VideoMAE achieves the highest accuracy on both datasets (98.14% UCF101, 85.48% HMDB51) while simultaneously requiring fewer GFLOPs than R(2+1)D-18 (135 vs. 163 GFLOPs). Its global self-attention mechanism, which can relate spatiotemporal patches from any two points in the clip regardless of temporal distance, provides substantially better temporal reasoning than locally factored 3D convolutions. The combination of masked autoencoder pretraining on Kinetics-400 and the transformer architecture's capacity for long-range temporal dependencies explains this superiority.

**The efficiency trade-off is multi-dimensional and deployment-context-dependent.** No single architecture dominates all others on all efficiency dimensions. VideoMAE leads on accuracy and GFLOPs but has the largest parameter count (86 M). R(2+1)D has moderate latency and the smallest parameter footprint among video-pretrained models but the highest GFLOPs and memory. TSN and CNN-LSTM offer the lowest latency (18–19 ms), smallest memory (281–323 MB), and lowest GFLOPs at the cost of approximately 10–11 accuracy points on HMDB51. For resource-constrained deployment, ResNet-50 TSN at 65.75 GFLOPs, 18.39 ms latency, and 281 MB memory represents a pragmatic choice, delivering 97.59% on UCF101 and 74.27% on HMDB51 from a model that requires no video pretraining.

**Recurrence adds marginally over averaging at 16 sparse frames.** The controlled TSN-vs-CNN-LSTM ablation demonstrates that adding 10.4 million parameters in the form of a bidirectional LSTM over 16 uniformly sampled frames yields only +0.32% on UCF101 and +0.78% on HMDB51. The improvement is real and consistent, but it is small relative to the cost in parameters and relative to the gains achieved by moving to 3D convolutions or attention. This result quantitatively confirms the field's empirical observation that the key to better video understanding lies in learning spatiotemporal features jointly during feature extraction, not in more sophisticated aggregation of independently extracted frame features.

**3D-CNN and Transformer pretraining on video data is a substantial confound.** Part of the advantage of R(2+1)D and VideoMAE over TSN and CNN-LSTM is attributable to the availability of Kinetics-400 pretrained checkpoints, which encode video-specific motion priors unavailable from ImageNet. Disentangling the pretraining contribution from the architectural contribution requires an ablation not performed in this study, and any interpretation of the inter-tier gaps should account for this confound.

## 6.3 Limitations

**Pretraining source asymmetry.** The two video-pretrained models (R(2+1)D, VideoMAE) benefit from Kinetics-400 pretraining while the two CNN-based models (TSN, CNN-LSTM) use ImageNet pretraining. This confounds architecture-versus-architecture comparisons across the two tiers. A fully controlled comparison would require either Kinetics-400 pretrained weights for all four architectures or a from-scratch training regime on a sufficiently large video dataset.

**Non-standard HMDB51 split.** The official HMDB51 evaluation protocol specifies three predefined train/test splits for three-fold cross-validation [25]. This study used a single 70/30 random split rather than the official splits, which means the reported HMDB51 numbers are not directly comparable with published results that use the official protocol. While the relative ranking of models is unlikely to change with official splits, the absolute accuracy values may differ by a few percentage points.

**Short training regime.** All models were trained for 10 epochs. While this was sufficient for convergence to a plateau under strong pretrained initialization, longer fine-tuning (30–50 epochs with learning rate decay) might yield higher absolute accuracy, potentially with different relative performance gaps. It is possible that some architectures benefit more from extended fine-tuning than others.

**Single-clip, no test-time augmentation.** All evaluation was performed using a single center-cropped clip. The published results for VideoMAE and R(2+1)D typically use multi-clip, multi-crop evaluation (e.g., 5 clips × 3 crops = 15 views per video), which can add 1–3 percentage points. The single-clip protocol was chosen for latency comparability but slightly penalizes the architectures that benefit most from multi-view ensembling.

**LSTM GFLOPs undercount.** As noted in Section 5.1.5, fvcore does not register a FLOP handler for nn.LSTM, resulting in an undercount of approximately 0.3 GFLOPs for the CNN-LSTM model. While negligible in magnitude (less than 0.5% of total GFLOPs), this should be noted in any formal benchmarking context.

**Single random seed.** All experiments were conducted with a single random seed. Given the small accuracy differences observed (particularly on UCF101), results averaged over multiple seeds with standard deviation reporting would provide a more statistically robust comparison.

## 6.4 Future Work

**Official evaluation splits.** Wiring the official UCF101 and HMDB51 split files into the evaluation pipeline and reporting results across all three official folds with mean and standard deviation would make the results directly comparable with the published literature and would improve statistical reliability.

**Pretraining equalization ablation.** The most important methodological extension is a controlled ablation in which all four architectures are fine-tuned from the same pretrained source — either Kinetics-400 checkpoints for all models (which would require a TSN-style Kinetics checkpoint for the ResNet-50 backbone) or a common from-scratch regime. This ablation would quantify the architecture contribution independently of pretraining source.

**Extended model family.** The current four-model comparison covers the major paradigms but omits several influential architectures. Natural extensions include SlowFast [16], which uses dual temporal pathways at different frame rates; TimeSformer [20], which applies divided space-time attention with lower computational cost than VideoMAE; Video Swin Transformer [21], which introduces local window attention for efficiency; and MViT [36], which uses multiscale attention to reduce the sequence length progressively through the network.

**Temporal action localization.** This study addresses clip-level classification, which is a simpler task than temporal action localization, where the model must also predict the start and end time of each action. Extending the framework to localization would increase practical relevance for applications such as surveillance, sports analytics, and medical procedure monitoring.

**Efficient transformer variants and deployment optimization.** For deployment research, the pipeline could be extended to evaluate efficient video transformer variants such as X3D [17] and MViT-V2, and to incorporate ONNX export, TensorRT optimization, and INT8 quantization to characterize model performance under deployment-realistic conditions. This would complete the efficiency characterization from theoretical GFLOPs to practical inference performance on target hardware.

**Longer training and learning rate search.** Training all models for 30–50 epochs with a proper learning rate sweep would establish whether the 10-epoch rankings are representative of the fully trained performance and whether the accuracy gaps between tiers change under extended optimization.

---
