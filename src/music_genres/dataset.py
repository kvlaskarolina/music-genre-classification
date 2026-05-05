from pathlib import Path

import kagglehub
import soundfile as sf
import torch
import torchaudio.functional as F
from torch.utils.data import Dataset

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
SAMPLE_RATE = 22050
N_SAMPLES = SAMPLE_RATE * 30  # 30-second clips

# jazz.00054.wav is a known corrupted file in the GTZAN dataset
_SKIP = {"jazz.00054.wav"}


class GTZANDataset(Dataset):
    def __init__(self, files: list[tuple[Path, int]], transform=None):
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.files[idx]
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (samples, channels)
        waveform = torch.from_numpy(data.T)  # (channels, samples)
        if sr != SAMPLE_RATE:
            waveform = F.resample(waveform, sr, SAMPLE_RATE)
        waveform = waveform.mean(0, keepdim=True)  # mono
        if waveform.shape[1] < N_SAMPLES:
            waveform = torch.nn.functional.pad(waveform, (0, N_SAMPLES - waveform.shape[1]))
        else:
            waveform = waveform[:, :N_SAMPLES]
        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform, label


def load_files(root: Path) -> list[tuple[Path, int]]:
    genre_dir = root / "Data" / "genres_original"
    files = []
    for label, genre in enumerate(GENRES):
        for wav in sorted((genre_dir / genre).glob("*.wav")):
            if wav.name not in _SKIP:
                files.append((wav, label))
    return files


def download_dataset() -> Path:
    return Path(kagglehub.dataset_download(
        "andradaolteanu/gtzan-dataset-music-genre-classification"
    ))