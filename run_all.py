"""End-to-end driver: train -> benchmark -> report for all three models.

All experiment settings live here so the Kaggle/Colab notebooks stay thin,
stable wrappers (no per-cell editing). Resumable: train.py skips any model whose
results CSV already has the requested epochs, and benchmark/report are cheap to
re-run, so a dropped session just continues where it left off (when RESULTS_DIR
points at persistent storage, e.g. Google Drive on Colab).

    python run_all.py --data-root /content/ucf101 --dataset ucf101 --epochs 10
"""
import argparse
import subprocess
import sys

from src.datasets import find_data_root

MODELS = ["videomae", "r2plus1d_18", "resnet50_tsn"]


def run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None, help="dataset path (else auto-detect)")
    ap.add_argument("--input-dir", default="/kaggle/input", help="where to auto-detect data")
    ap.add_argument("--dataset", default="ucf101")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--skip-train", action="store_true", help="benchmark + report only")
    args = ap.parse_args()

    data_root = args.data_root or find_data_root(args.input_dir)
    print(f"data_root = {data_root}\nmodels    = {args.models}\nepochs    = {args.epochs}")

    if not args.skip_train:
        for m in args.models:
            run([sys.executable, "-m", "src.train",
                 "--model", m, "--dataset", args.dataset,
                 "--data-root", data_root,
                 "--epochs", str(args.epochs),
                 "--batch-size", str(args.batch_size),
                 "--num-workers", str(args.num_workers)])

    for m in args.models:
        run([sys.executable, "-m", "src.benchmark",
             "--model", m, "--dataset", args.dataset])

    run([sys.executable, "-m", "src.report", "--dataset", args.dataset])
    print("\nrun_all complete.")


if __name__ == "__main__":
    main()
