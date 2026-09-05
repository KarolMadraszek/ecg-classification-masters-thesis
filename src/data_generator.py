import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import h5py
import numpy as np
import tensorflow as tf
from pathlib import Path

class CWTDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, h5_path, split='train', batch_size=32, shuffle=True):
        self.h5_path = h5_path
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle

        with h5py.File(self.h5_path, 'r') as f:
            self.indexes = np.arange(f[self.split]['X'].shape[0])

        self.on_epoch_end()

    def __len__(self):
        with h5py.File(self.h5_path, 'r') as f:
            total_samples = f[self.split]['X'].shape[0]
        return int(np.ceil(total_samples / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.indexes))
        batch_indexes = self.indexes[start_idx:end_idx]

        batch_indexes = np.sort(batch_indexes)

        with h5py.File(self.h5_path, 'r') as f:
            group = f[self.split]
            X_batch = group['X'][batch_indexes].astype(np.float32)
            y_batch = group['y'][batch_indexes].astype(np.float32)

        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


if __name__ == '__main__':
    # Test dla wersji deweloperskiej
    base_dir = Path(__file__).resolve().parent.parent
    dev_h5 = base_dir / 'data' / 'processed' / 'cwt_scalograms_DEV.h5'

    if dev_h5.exists():
        generator = CWTDataGenerator(dev_h5, split='train', batch_size=16)
        print(f"Liczba partii w epoce: {len(generator)}")

        X_sample, y_sample = generator[0]
        print(f"Kształt partii X: {X_sample.shape} (Oczekiwane: (16, 39, 1000, 12))")
        print(f"Kształt partii y: {y_sample.shape} (Oczekiwane: (16, 5))")
        print("Generator działa poprawnie!")
    else:
        print("Najpierw proszę wygenerować plik CWT dla wersji DEV, aby przetestować generator.")