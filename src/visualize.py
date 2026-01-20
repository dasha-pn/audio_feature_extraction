import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from src.utils import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--sr", type=int, default=22050)
    args = ap.parse_args()

    ensure_dir("outputs/plots")

    y, sr = librosa.load(args.file, sr=args.sr, mono=True)

    # Waveform
    plt.figure()
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig("outputs/plots/waveform.png", bbox_inches="tight")
    plt.close()

    # Mel-spectrogram (dB)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure()
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
    plt.title("Mel-Spectrogram (dB)")
    plt.colorbar(format="%+2.0f dB")
    plt.savefig("outputs/plots/mel_spectrogram.png", bbox_inches="tight")
    plt.close()

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure()
    librosa.display.specshow(mfcc, sr=sr, x_axis="time")
    plt.title("MFCC")
    plt.colorbar()
    plt.savefig("outputs/plots/mfcc.png", bbox_inches="tight")
    plt.close()

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    plt.figure()
    librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma")
    plt.title("Chroma")
    plt.colorbar()
    plt.savefig("outputs/plots/chroma.png", bbox_inches="tight")
    plt.close()

    print("Saved plots to outputs/plots/")

if __name__ == "__main__":
    main()
