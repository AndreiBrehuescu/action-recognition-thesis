"""Model-agnostic fine-tuning loop. Same code path for all three families.

    python -m src.train --model videomae --dataset ucf101 --epochs 5 --batch-size 8
"""
import argparse
import csv
import time

import torch
from torch.utils.data import DataLoader

from .config import RESULTS_DIR, CHECKPOINT_DIR, DATA_ROOT
from .datasets import build_dataset
from .models import build_model
from .evaluate import evaluate_model


def _append_csv(path, header, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def _safe_batch_size(model_name, requested, device):
    """Clamp the batch size to what fits on a smaller GPU (e.g. Kaggle's ~15 GB
    T4) at 16 frames / 224 px. Larger cards (the 24 GB local GPU) are untouched.
    Guards against an over-large --batch-size from the notebook causing a CUDA
    OOM -- enforced here, in code, because notebook cell values don't update via
    git pull."""
    if device != "cuda":
        return requested
    try:
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        return requested
    if total_gb < 18 and requested > 8:
        print(f"[batch-cap] {model_name}: requested batch {requested} is too large "
              f"for a {total_gb:.0f} GB GPU at 16 frames; capping to 8.", flush=True)
        return 8
    return requested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet50_tsn", "r2plus1d_18", "videomae"])
    ap.add_argument("--dataset", default="ucf101")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.batch_size = _safe_batch_size(args.model, args.batch_size, device)
    print(f"device={device} | model={args.model} | dataset={args.dataset} | batch={args.batch_size}", flush=True)

    print("scanning dataset (can take ~1 min on first run)...", flush=True)
    train_ds, num_classes = build_dataset(args.dataset, args.data_root, args.split, train=True)
    test_ds, _ = build_dataset(args.dataset, args.data_root, args.split, train=False)
    print(f"datasets ready: train={len(train_ds)} clips, test={len(test_ds)} clips, "
          f"classes={num_classes}", flush=True)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    print(f"building '{args.model}' (downloads pretrained weights on first use)...", flush=True)
    model = build_model(args.model, num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    log_path = RESULTS_DIR / f"{args.model}_{args.dataset}_train.csv"
    if log_path.exists():                                   # idempotent re-runs
        done = max(0, sum(1 for _ in open(log_path)) - 1)   # rows minus header
        if done >= args.epochs:
            print(f"[skip] {log_path.name} already has {done} epochs logged; skipping.")
            return
    best = 0.0
    print(f"starting training: {args.epochs} epochs x {len(train_ld)} batches/epoch", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, t0 = 0.0, time.time()
        n_batches = len(train_ld)
        for i, (clips, labels) in enumerate(train_ld, 1):
            clips, labels = clips.to(device), labels.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                loss = criterion(model(clips), labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * clips.size(0)
            if i == 1 or i % 20 == 0 or i == n_batches:     # frequent heartbeat
                print(f"  epoch {epoch} | batch {i}/{n_batches} | loss={loss.item():.3f}", flush=True)
        sched.step()

        train_loss = running / len(train_ds)
        acc1, acc5 = evaluate_model(model, test_ld, device)
        dt = time.time() - t0
        print(f"epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  "
              f"top1={acc1:.2f}  top5={acc5:.2f}  ({dt:.0f}s)", flush=True)
        _append_csv(log_path, ["epoch", "train_loss", "top1", "top5", "sec"],
                    [epoch, f"{train_loss:.4f}", f"{acc1:.2f}", f"{acc5:.2f}", f"{dt:.0f}"])

        if acc1 > best:
            best = acc1
            torch.save(model.state_dict(),
                       CHECKPOINT_DIR / f"{args.model}_{args.dataset}_best.pth")
    print(f"done. best top1={best:.2f}  (log: {log_path})")


if __name__ == "__main__":
    main()
