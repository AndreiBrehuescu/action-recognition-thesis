"""Lay out dataset folders and fetch the official UCF101 train/test split lists.

    python scripts/setup_data.py --dataset ucf101 --splits-only

Expected final layout:
    data/ucf101/UCF-101/<ClassName>/<video>.avi      <- the videos (you add these)
    data/ucf101/<...>/classInd.txt, trainlist01.txt, testlist01.txt

On Kaggle: 'Add Data' -> UCF101, then pass --data-root /kaggle/input/<dataset-path>
to the training scripts (no download needed).
"""
import argparse
import urllib.request
import zipfile
from pathlib import Path

SPLIT_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--dataset", default="ucf101")
    ap.add_argument("--splits-only", action="store_true",
                    help="download the official UCF101 train/test split lists")
    args = ap.parse_args()

    root = Path(args.data_root) / args.dataset
    (root / "splits").mkdir(parents=True, exist_ok=True)
    print(f"created {root.resolve()}")

    if args.splits_only and args.dataset == "ucf101":
        zpath = root / "splits.zip"
        print(f"downloading split lists from {SPLIT_URL} ...")
        try:
            urllib.request.urlretrieve(SPLIT_URL, zpath)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(root / "splits")
            zpath.unlink(missing_ok=True)
            print("extracted official split lists into", root / "splits")
        except Exception as e:
            print(f"automatic download failed ({e}).")
            print("Get UCF101TrainTestSplits-RecognitionTask.zip from the UCF101 CRCV "
                  "page and unzip it under", root / "splits")

    print("\nNext: place the UCF101 videos under", root / "UCF-101",
          "\n(Kaggle users: skip this and point --data-root at the added dataset).")


if __name__ == "__main__":
    main()
