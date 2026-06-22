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
