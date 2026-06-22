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
