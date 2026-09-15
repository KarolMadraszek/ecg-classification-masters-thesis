import h5py
import numpy as np
import tensorflow as tf

def create_tukey_window(length, alpha=0.1):
    if alpha <= 0:
        return np.ones(length)
    elif alpha >= 1:
        return np.hanning(length)

    x = np.linspace(0, 1, length)
    w = np.ones(length)

    # Lewy brzeg
    first_condition = x < alpha / 2
    w[first_condition] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[first_condition] - alpha / 2)))

    # Prawy brzeg
    third_condition = x >= 1 - alpha / 2
    w[third_condition] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[third_condition] - 1 + alpha / 2)))

    return w

class H5MemorySafeGenerator(tf.keras.utils.Sequence):
    def __init__(self, h5_path, split, batch_size=32, shuffle=False,
                 crop=False, crop_range=(50, 950),
                 taper_edges=False, freq_masking=False):
        self.h5_path = h5_path
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.crop = crop
        self.crop_range = crop_range
        self.taper_edges = taper_edges
        self.freq_masking = freq_masking

        with h5py.File(self.h5_path, 'r') as f:
            self.num_samples = f[self.split]['X'].shape[0]

        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)

        self.tukey_window = None
        if self.taper_edges:
            length = self.crop_range[1] - self.crop_range[0] if self.crop else 1000
            # (1, 1, czas, 1)
            self.tukey_window = create_tukey_window(length, alpha=0.1).reshape(1, 1, -1, 1)

    def __len__(self):
        return int(np.ceil(self.num_samples / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_indices = np.sort(batch_indices)

        with h5py.File(self.h5_path, 'r') as f:
            X_batch = np.array(f[self.split]['X'][batch_indices], dtype=np.float32)
            y_batch = np.array(f[self.split]['y'][batch_indices], dtype=np.float32)

        if self.crop:
            start, end = self.crop_range
            X_batch = X_batch[:, :, start:end, :]

        if self.taper_edges and self.tukey_window is not None:
            X_batch = X_batch * self.tukey_window

        if self.freq_masking and self.split == 'train':
            X_batch = self._apply_freq_mask(X_batch)

        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _apply_freq_mask(self, X_batch, max_mask_pct=0.15):
        # Założenie kształtu wejścia: (batch, freq, time, channels)
        n_freq = X_batch.shape[1]
        for i in range(X_batch.shape[0]):
            if np.random.rand() < 0.7:
                mask_size = np.random.randint(1, int(n_freq * max_mask_pct) + 1)
                mask_start = np.random.randint(0, n_freq - mask_size)
                X_batch[i, mask_start: mask_start + mask_size, :, :] = 0.0

        return X_batch