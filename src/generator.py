import h5py
import numpy as np
import tensorflow as tf

class H5MemorySafeGenerator(tf.keras.utils.Sequence):
    def __init__(self, h5_path, split='train', batch_size=64, crop=False, crop_range=(50, 950)):
        self.h5_path = h5_path
        self.split = split
        self.batch_size = batch_size
        self.crop = crop
        self.crop_range = crop_range

        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f[self.split]['y'])

    def __len__(self):
        return int(np.ceil(self.length / self.batch_size))

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            start = idx * self.batch_size
            end = min(start + self.batch_size, self.length)
            X_batch = f[self.split]['X'][start:end]
            y_batch = f[self.split]['y'][start:end]

        if self.crop:
            c_start, c_end = self.crop_range
            X_batch = X_batch[:, :, c_start:c_end, :]

        return X_batch, y_batch