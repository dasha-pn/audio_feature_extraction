import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import ensure_dir

DEFAULT_FEATURES = [
    "zcr_mean",
    "centroid_mean",
    "rolloff_mean",
    "rms_mean",
    "mel_db_mean",
    "mfcc1_mean",
    "mfcc2_mean",
    "mfcc3_mean",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/features/features.csv")
    ap.add_argument("--feature", default=None, help="single feature to plot")
    args = ap.parse_args()

    ensure_dir("outputs/plots")

    df = pd.read_csv(args.csv)
    df = df[df["label"].isin(["speech", "music", "noise"])]

    features = [args.feature] if args.feature else DEFAULT_FEATURES

    for feat in features:
        if feat not in df.columns:
            print(f"Skip missing feature: {feat}")
            continue

        groups = [df[df["label"] == lab][feat].dropna() for lab in ["speech", "music", "noise"]]

        plt.figure()
        plt.boxplot(groups, labels=["speech", "music", "noise"])
        plt.title(f"Feature comparison: {feat}")
        plt.ylabel(feat)
        out_path = f"outputs/plots/compare_{feat}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
