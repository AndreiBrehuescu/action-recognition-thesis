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
