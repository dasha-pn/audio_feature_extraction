import os
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

def list_audio_files(root_dir: str):
    root = Path(root_dir)
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(str(p))
    return sorted(files)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
