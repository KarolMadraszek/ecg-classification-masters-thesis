import h5py
import numpy as np
import tensorflow as tf

def create_tukey_window(length, alpha=0.1):
    n = np.arange(length)
    width = int(alpha * (length - 1) / 2.0)
    w = np.ones(length, dtype=np.float32)
    if width > 0:
        w[:width] = 0.5 * (1.0 + np.cos(np.pi * (-1.0 + 2.0 * n[:width] / (2.0 * width))))
        w[-width:] = 0.5 * (1.0 + np.cos(np.pi * (-2.0 + 2.0 * (length - 1 - n[-width:]) / (2.0 * width))))
    return w

class H5MemorySafeGenerator(tf.keras.utils.Sequence):
    def __init__(self, h5_path, split, batch_size=32, shuffle=False,
                 crop=False, crop_range=(50, 950),
                 taper_edges=False, freq_masking=False):
        self.h5_path = str(h5_path)
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.crop = crop
        self.crop_range = crop_range
        self.taper_edges = taper_edges
        self.freq_masking = freq_masking

        self.f = h5py.File(self.h5_path, 'r')
        self.num_samples = self.f[self.split]['X'].shape[0]

        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)

        self.tukey_window = None
        if self.taper_edges:
            length = self.crop_range[1] - self.crop_range[0] if self.crop else 1000
            self.tukey_window = create_tukey_window(length, alpha=0.1).reshape(1, 1, -1, 1)

    def __len__(self):
        return int(np.ceil(self.num_samples / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        sorted_order = np.argsort(batch_indices)
        sorted_indices = batch_indices[sorted_order]

        X_batch = np.array(self.f[self.split]['X'][sorted_indices], dtype=np.float32)
        y_batch = np.array(self.f[self.split]['y'][sorted_indices], dtype=np.float32)

        inverse_order = np.empty_like(sorted_order)
        inverse_order[sorted_order] = np.arange(len(sorted_order))
        X_batch = X_batch[inverse_order]
        y_batch = y_batch[inverse_order]

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
        n_freq = X_batch.shape[1]
        for i in range(X_batch.shape[0]):
            if np.random.rand() < 0.7:
                mask_size = np.random.randint(1, int(n_freq * max_mask_pct) + 1)
                mask_start = np.random.randint(0, n_freq - mask_size)
                X_batch[i, mask_start: mask_start + mask_size, :, :] = 0.0
        return X_batch

    def __del__(self):
        if hasattr(self, 'f') and self.f:
            try:
                self.f.close()
            except Exception:
                pass