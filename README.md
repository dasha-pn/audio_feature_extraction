# Audio Feature Extraction & Visualization Toolkit

A lightweight audio analysis toolkit for extracting, visualizing, and comparing
audio features from real-world audio signals.  
The project focuses on **audio understanding and signal processing**, rather than
training machine learning models.

---

## Project Overview

This project implements an end-to-end pipeline for audio analysis that includes:
- loading and preprocessing raw audio files,
- extracting common time-domain and spectral audio features,
- visualizing audio representations,
- comparing features across different sound categories.

It is designed as a **portfolio-ready project** for audio and signal processing roles.

---

## Goal

To analyze and understand audio signals by extracting and visualizing
their acoustic features.

### Sound categories
- Speech  
- Music  
- Noise  

---

## Extracted Audio Features

### Time-domain features
- Signal duration
- Root Mean Square (RMS) energy
- Zero-Crossing Rate (ZCR)

### Spectral features
- Spectral centroid
- Spectral roll-off

### Perceptual features
- Mel-spectrogram (log-scaled)
- Mel-Frequency Cepstral Coefficients (MFCCs)
- Chroma features

Frame-level features are aggregated into fixed-length representations
using **mean and standard deviation**.

---

## Dataset

Audio samples are organized manually into three folders:
- `audio_samples/speech/`
- `audio_samples/music/`
- `audio_samples/noise/`

Each audio file is a short clip (10–30 seconds) in `wav` or `mp3` format.

Audio samples were taken from **publicly available datasets**
(e.g. LibriSpeech, MUSAN, GTZAN, FreeSound) and are used for feature analysis only.

---

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Extract audio features

```bash
python -m src.extract_features \
  --input_dir audio_samples \
  --out_csv outputs/features/features.csv
```

This command extracts audio features from all audio files and saves them to a CSV file.

## Visualize features for a single audio file

```bash
python -m src.visualize --file audio_samples/speech/example.wav
```

Generated plots:

- waveform

- Mel-spectrogram

- MFCC

- chroma

Plots are saved to `outputs/plots/`.

## Compare features across sound classes

```bash
python -m src.compare_features --csv outputs/features/features.csv
```

This command generates boxplots comparing selected audio features
between speech, music, and noise.

## Example Outputs

- `features.csv` — extracted numerical audio features

- `waveform.png` — time-domain waveform

- `mel_spectrogram.png` — time–frequency representation

- `compare_centroid_mean.png` — feature comparison across classes
