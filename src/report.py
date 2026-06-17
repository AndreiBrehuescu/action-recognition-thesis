"""Summarize results and draw the accuracy-vs-GFLOPs Pareto scatter.

    python -m src.report --dataset ucf101

Reads results/<model>_<dataset>_train.csv (accuracy) and results/efficiency.csv
(params/GFLOPs/...) and writes results/pareto_<dataset>.png/.pdf.
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless (Kaggle/Colab)
import matplotlib.pyplot as plt

from .config import RESULTS_DIR

COLORS = {"resnet50_tsn": "#4C72B0", "cnn_lstm": "#C44E52",
          "r2plus1d_18": "#DD8452", "videomae": "#55A868"}
LABELS = {"resnet50_tsn": "ResNet-50 TSN (2D-CNN)",
          "cnn_lstm": "ResNet-50 + LSTM (CNN-RNN)",
          "r2plus1d_18": "R(2+1)D-18 (3D-CNN)",
          "videomae": "VideoMAE (Transformer)"}
ORDER = ("resnet50_tsn", "cnn_lstm", "r2plus1d_18", "videomae")


def _best_acc(results_dir, dataset):
    acc = {}
    for f in sorted(results_dir.glob(f"*_{dataset}_train.csv")):
        model = f.stem.replace(f"_{dataset}_train", "")
        df = pd.read_csv(f)
        if "top1" in df and len(df):
            acc[model] = float(df["top1"].astype(float).max())
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ucf101")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    results_dir = Path(args.results_dir)

    acc = _best_acc(results_dir, args.dataset)
    print("\n=== Accuracy (best top-1) ===")
    for m in ORDER:
        if m in acc:
            print(f"  {LABELS[m]:28s} {acc[m]:.2f}%")

    eff_path = results_dir / "efficiency.csv"
    if not eff_path.exists():
        print("\n(no efficiency.csv yet -- run the benchmark to get the Pareto plot)")
        return

    eff = pd.read_csv(eff_path).drop_duplicates("model", keep="last")
    eff["top1"] = eff["model"].map(acc)
    print("\n=== Efficiency ===")
    print(eff.to_string(index=False))

    plot_df = eff.dropna(subset=["top1"])
    if plot_df.empty:
        print("\n(no model has both accuracy and efficiency yet -- skipping plot)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in plot_df.iterrows():
        ax.scatter(row["gflops"], row["top1"], s=max(row["params_M"] * 2, 20),
                   color=COLORS.get(row["model"], "gray"),
                   alpha=0.85, edgecolors="k", linewidths=0.5, zorder=3)
        ax.annotate(LABELS.get(row["model"], row["model"]),
                    (row["gflops"], row["top1"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("GFLOPs / clip")
    ax.set_ylabel(f"{args.dataset.upper()} Top-1 Accuracy (%)")
    ax.set_title("Accuracy vs Computational Cost\n(bubble size proportional to parameter count)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = results_dir / f"pareto_{args.dataset}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved pareto_{args.dataset}.png / .pdf in {results_dir}")


if __name__ == "__main__":
    main()
