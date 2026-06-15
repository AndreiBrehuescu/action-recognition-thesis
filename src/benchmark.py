"""Efficiency side of the comparison: params / FLOPs / latency / peak memory.

    python -m src.benchmark --model videomae
Appends one row per model to results/efficiency.csv.
"""
import argparse
import csv
import time

import torch

from .config import NUM_FRAMES, IMG_SIZE, RESULTS_DIR, DATASETS
from .models import build_model


def _flops_g(model, x):
    """GFLOPs for a single clip (hardware-independent)."""
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return fca.total() / 1e9
    except Exception as e:
        print(f"FLOPs unavailable ({e})")
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet50_tsn", "r2plus1d_18", "videomae"])
    ap.add_argument("--dataset", default="ucf101")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--data-root", default=None)        # accepted & ignored (synthetic input)
    ap.add_argument("--num-workers", type=int, default=0)  # accepted & ignored
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = DATASETS[args.dataset]["classes"]
    model = build_model(args.model, num_classes).to(device).eval()

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    x = torch.randn(args.batch_size, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE, device=device)
    gflops = _flops_g(model, x[:1])

    with torch.no_grad():
        for _ in range(5):                              # warmup
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(args.iters):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    dt = (time.time() - t0) / args.iters
    latency_ms = dt * 1000 / args.batch_size
    throughput = args.batch_size / dt
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else float("nan")

    print(f"{args.model}: params={params_m:.1f}M  gflops={gflops:.1f}  "
          f"latency={latency_ms:.1f}ms/clip  throughput={throughput:.1f}clip/s  "
          f"peakmem={peak_mb:.0f}MB")

    out = RESULTS_DIR / "efficiency.csv"
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "params_M", "gflops", "latency_ms", "throughput", "peak_mem_MB"])
        w.writerow([args.model, f"{params_m:.2f}", f"{gflops:.2f}",
                    f"{latency_ms:.2f}", f"{throughput:.2f}", f"{peak_mb:.0f}"])
    print(f"appended -> {out}")


if __name__ == "__main__":
    main()
