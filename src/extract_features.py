import argparse
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from src.utils import list_audio_files, ensure_dir

def extract_features(file_path: str, sr: int = 22050, n_mfcc: int = 13):
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]

    zcr = librosa.feature.zero_crossing_rate(y)[0]

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # (n_mfcc, T)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    feats = {
        "duration_sec": float(duration),

        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),

        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),

        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),

        "rolloff_mean": float(np.mean(rolloff)),
        "rolloff_std": float(np.std(rolloff)),

        "mel_db_mean": float(np.mean(mel_db)),
        "mel_db_std": float(np.std(mel_db)),
    }

    for i in range(mfcc.shape[0]):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))

    for i in range(chroma.shape[0]):
        feats[f"chroma{i+1}_mean"] = float(np.mean(chroma[i]))
        feats[f"chroma{i+1}_std"] = float(np.std(chroma[i]))

    return feats

def infer_label_from_path(path: str):
    p = path.replace("\\", "/").lower()
    if "/speech/" in p:
        return "speech"
    if "/music/" in p:
        return "music"
    if "/noise/" in p:
        return "noise"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="audio_samples")
    ap.add_argument("--out_csv", default="outputs/features/features.csv")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_mfcc", type=int, default=13)
    args = ap.parse_args()

    ensure_dir("outputs/features")

    files = list_audio_files(args.input_dir)
    if not files:
        raise SystemExit(f"No audio files found in {args.input_dir}")

    rows = []
    for f in tqdm(files, desc="Extracting features"):
        feats = extract_features(f, sr=args.sr, n_mfcc=args.n_mfcc)
        feats["file"] = f
        feats["label"] = infer_label_from_path(f)
        rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv} (rows={len(df)})")

if __name__ == "__main__":
    main()
