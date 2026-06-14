"""The three architecture families, each wrapped to a common interface:

    forward(clip) where clip is (B, T, C, H, W)  ->  logits (B, num_classes)

so `train.py` / `evaluate.py` / `benchmark.py` stay completely model-agnostic.
"""
import torch
import torch.nn as nn
import torchvision


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


class VideoMAE(nn.Module):
    """Transformer: VideoMAE-base, Kinetics-400 pretrained (HuggingFace)."""

    def __init__(self, num_classes):
        super().__init__()
        from transformers import VideoMAEForVideoClassification
        self.net = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,       # swap the K400 head for ours
        )

    def forward(self, x):                       # VideoMAE expects (B, T, C, H, W)
        return self.net(pixel_values=x).logits


BUILDERS = {
    "resnet50_tsn": ResNet50TSN,
    "r2plus1d_18": R2Plus1D,
    "videomae": VideoMAE,
}


def build_model(name, num_classes):
    if name not in BUILDERS:
        raise ValueError(f"unknown model '{name}'. choices: {list(BUILDERS)}")
    return BUILDERS[name](num_classes)
