"""Accuracy evaluation: top-1 / top-5 and an optional confusion matrix.

    python -m src.evaluate --model videomae --dataset ucf101 --confusion
"""
import argparse

import torch
from torch.utils.data import DataLoader

from .config import DATA_ROOT, CHECKPOINT_DIR, RESULTS_DIR
from .datasets import build_dataset
from .models import build_model


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Returns (top1 %, top5 %)."""
    model.eval()
    top1 = top5 = total = 0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        out = model(clips)
        _, pred5 = out.topk(5, dim=1)
        correct = pred5.eq(labels.view(-1, 1))
        top1 += correct[:, :1].sum().item()
        top5 += correct.sum().item()
        total += labels.size(0)
    return 100 * top1 / total, 100 * top5 / total


def _save_confusion(model, loader, device, num_classes, tag):
    import numpy as np
    from sklearn.metrics import confusion_matrix
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for clips, labels in loader:
            ps.append(model(clips.to(device)).argmax(1).cpu().numpy())
            ys.append(labels.numpy())
    cm = confusion_matrix(np.concatenate(ys), np.concatenate(ps),
                          labels=list(range(num_classes)))
    out = RESULTS_DIR / f"{tag}_confusion.csv"
    np.savetxt(out, cm, fmt="%d", delimiter=",")
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet50_tsn", "r2plus1d_18", "videomae"])
    ap.add_argument("--dataset", default="ucf101")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--confusion", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_ds, num_classes = build_dataset(args.dataset, args.data_root, args.split, train=False)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model, num_classes).to(device)
    ckpt = args.checkpoint or str(CHECKPOINT_DIR / f"{args.model}_{args.dataset}_best.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    acc1, acc5 = evaluate_model(model, loader, device)
    print(f"{args.model} on {args.dataset}: top1={acc1:.2f}  top5={acc5:.2f}")
    if args.confusion:
        _save_confusion(model, loader, device, num_classes, f"{args.model}_{args.dataset}")


if __name__ == "__main__":
    main()
