import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import h5py
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

class H5MemorySafeGenerator(tf.keras.utils.Sequence):
    def __init__(self, h5_path, split='test', batch_size=64):
        self.h5_path = h5_path
        self.split = split
        self.batch_size = batch_size
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
        return X_batch, y_batch

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        elif hasattr(layer, 'output') and hasattr(layer.output, 'shape'):
            if len(layer.output.shape) == 4 and 'conv' in layer.name.lower():
                return layer.name
    raise ValueError("Brak warstwy konwolucyjnej.")

def compute_gradcam(sample_x, model, conv_layer, class_idx):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(sample_x)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val
    return heatmap.numpy()

def main():
    base_dir = Path(__file__).resolve().parent
    h5_path = base_dir / 'data' / 'processed' / 'cwt_scalograms_FULL.h5'
    models_dir = base_dir / 'models'
    out_dir = models_dir / 'error_analysis_plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        y_true = f['test']['y'][:]

    models = {
        'baseline': tf.keras.models.load_model(models_dir / "best_baseline.keras"),
        'densenet121': tf.keras.models.load_model(models_dir / "best_densenet121.keras")
    }

    conv_layers = {m: get_last_conv_layer(models[m]) for m in models}

    preds = {}
    test_gen = H5MemorySafeGenerator(h5_path, split='test', batch_size=64)
    for m in models:
        p = models[m].predict(test_gen, verbose=0)
        preds[m] = p[:len(y_true)]

    records = []
    for cls_idx, cls_name in enumerate(CLASSES):
        yt = y_true[:, cls_idx]
        p_base = preds['baseline'][:, cls_idx]
        p_dense = preds['densenet121'][:, cls_idx]

        err_base = np.abs(yt - p_base)
        err_dense = np.abs(yt - p_dense)

        joint_error = err_base * err_dense

        for idx in range(len(yt)):
            records.append({
                'sample_idx': idx,
                'class_name': cls_name,
                'class_idx': cls_idx,
                'y_true': yt[idx],
                'pred_baseline': p_base[idx],
                'pred_densenet121': p_dense[idx],
                'joint_error': joint_error[idx],
                'err_type': 'FP' if yt[idx] == 0 else 'FN'
            })

    df = pd.DataFrame(records)
    df.to_csv(models_dir / "detailed_error_analysis.csv", index=False)

    top_errors = df.sort_values(by='joint_error', ascending=False).drop_duplicates(subset=['sample_idx']).head(5)

    with h5py.File(h5_path, 'r') as f:
        for _, row in top_errors.iterrows():
            idx = int(row['sample_idx'])
            c_idx = int(row['class_idx'])
            c_name = row['class_name']

            sample_x = f['test']['X'][idx:idx + 1]
            bg_img = np.mean(sample_x[0], axis=-1)

            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            for i, m in enumerate(['baseline', 'densenet121']):
                h = compute_gradcam(sample_x, models[m], conv_layers[m], c_idx)
                h_resized = cv2.resize(h, (bg_img.shape[1], bg_img.shape[0]))

                axes[i].imshow(bg_img, cmap='gray', aspect='auto')
                axes[i].imshow(h_resized, cmap='jet', alpha=0.45, aspect='auto')
                p_val = row[f'pred_{m}']
                axes[i].set_title(
                    f"{m.upper()} | Cel: {c_name} (Prawda: {int(row['y_true'])}, Pred: {p_val * 100:.1f}%)")
                axes[i].set_ylabel("Skala CWT")

            axes[1].set_xlabel("Czas (próbki)")
            plt.tight_layout()
            plt.savefig(out_dir / f"error_sample_{idx}_{c_name}_{row['err_type']}.png", dpi=200)
            plt.close(fig)

    print(f"Zapisano CSV i wykresy Grad-CAM błędnych diagnoz w: {out_dir}")

if __name__ == '__main__':
    main()