import os
import numpy as np
import tensorflow as tf
import librosa


def load_dataset(duration=10, sr=22050, n_mels=128):
    mel_specs = []
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_count = len([f for f in os.listdir(data_dir) if f.endswith(".wav")])
    print(f"Processing {file_count} WAV files...")

    for i, filename in enumerate(os.listdir(data_dir)):
        if filename.endswith(".wav"):
            file_path = os.path.join(data_dir, filename)
            y, sr = librosa.load(file_path, sr=sr, duration=duration)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=sr // 100)
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            mel_specs.append(mel_spec)
            print(f"Processed {i + 1}/{file_count} files")

    if not mel_specs:
        return np.array([]), 1.0, 0.0

    mel_specs = np.array(mel_specs)
    mel_specs = np.clip(mel_specs, -80.0, 0.0)
    mel_specs = (mel_specs + 80.0) / 80.0
    max_val = 1.0
    min_val = 0.0
    return mel_specs, max_val, min_val