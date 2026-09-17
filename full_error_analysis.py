import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import h5py
import cv2
import gc
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from src.generator import H5MemorySafeGenerator

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def make_gradcam_heatmap(img_array, model, class_idx):
    last_conv_output = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            last_conv_output = layer.input
            break

    if last_conv_output is None:
        raise ValueError("Nie znaleziono warstwy GlobalAveragePooling2D w modelu.")

    grad_model = tf.keras.models.Model(
        model.inputs,
        [last_conv_output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
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

def run_error_analysis(h5_path, models_dir, iter_name, crop, model_filenames):
    print("\n" + "=" * 80)
    print(f"ROZPOCZĘCIE ANALIZY BŁĘDÓW: {iter_name.upper()}")
    print("=" * 80)

    out_dir = models_dir / f'error_analysis_plots_{iter_name}'
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        y_true = f['test']['y'][:]

    loaded_models = {}
    for m_name, filename in model_filenames.items():
        model_path = models_dir / filename
        if not model_path.exists():
            print(f" -> Brak modelu: {model_path}. Pominięto.")
            continue
        print(f" -> Ładowanie wag: {m_name}")
        loaded_models[m_name] = tf.keras.models.load_model(model_path, compile=False)

    if not loaded_models:
        print(f"Nie załadowano modeli dla {iter_name}.")
        return

    if crop:
        test_gen = H5MemorySafeGenerator(h5_path, split='test', batch_size=64, crop=True, crop_range=(50, 950))
    else:
        test_gen = H5MemorySafeGenerator(h5_path, split='test', batch_size=64)

    preds = {}
    print("Generowanie predykcji")
    for m_name, model in loaded_models.items():
        p = model.predict(test_gen, verbose=1)
        preds[m_name] = p[:len(y_true)]

    print("Obliczanie błędów wspólnych")
    records = []
    for cls_idx, cls_name in enumerate(CLASSES):
        yt = y_true[:, cls_idx]

        model_errors = {m_name: np.abs(yt - preds[m_name][:, cls_idx]) for m_name in loaded_models}

        # Błąd wspólny (ang. joint error) to iloczyn błędów wszystkich analizowanych modeli
        joint_error = np.prod(list(model_errors.values()), axis=0)

        for idx in range(len(yt)):
            record = {
                'sample_idx': idx,
                'class_name': cls_name,
                'class_idx': cls_idx,
                'y_true': yt[idx],
                'joint_error': joint_error[idx],
                'err_type': 'FP' if yt[idx] == 0 else 'FN'
            }
            for m_name in loaded_models:
                record[f'pred_{m_name}'] = preds[m_name][idx, cls_idx]

            records.append(record)

    df = pd.DataFrame(records)
    csv_out = models_dir / f"detailed_error_analysis_{iter_name}.csv"
    df.to_csv(csv_out, index=False)
    print(f"Zapisano CSV z błędami w: {csv_out}")

    top_errors = df.sort_values(by='joint_error', ascending=False).drop_duplicates(subset=['sample_idx']).head(5)

    print("Generowanie wizualizacji Grad-CAM dla top 5 błędów")
    for _, row in top_errors.iterrows():
        idx = int(row['sample_idx'])
        c_idx = int(row['class_idx'])
        c_name = row['class_name']

        with h5py.File(h5_path, 'r') as f:
            sample_x = f['test']['X'][idx:idx + 1]

        if crop:
            sample_x = sample_x[:, :, 50:950, :]

        bg_img = np.mean(sample_x[0], axis=-1)

        fig, axes = plt.subplots(len(loaded_models), 1, figsize=(12, 5 * len(loaded_models)))
        if len(loaded_models) == 1:
            axes = [axes]

        for ax, (m_name, model) in zip(axes, loaded_models.items()):
            h = make_gradcam_heatmap(sample_x, model, c_idx)
            h_resized = cv2.resize(h, (bg_img.shape[1], bg_img.shape[0]))

            ax.imshow(bg_img, cmap='gray', aspect='auto')
            ax.imshow(h_resized, cmap='jet', alpha=0.45, aspect='auto')

            p_val = row[f'pred_{m_name}']
            ax.set_title(f"{m_name.upper()} | Cel: {c_name} (Prawda: {int(row['y_true'])}, Pred: {p_val * 100:.1f}%)",
                         fontsize=14, fontweight='bold')
            ax.set_ylabel("Skala CWT")

        axes[-1].set_xlabel("Czas (próbki)")
        plt.tight_layout()
        plot_out = out_dir / f"error_sample_{idx}_{c_name}_{row['err_type']}.png"
        plt.savefig(plot_out, dpi=200)
        plt.close(fig)
        print(f" -> Zapisano przypadek {idx} ({c_name})")

    del loaded_models
    tf.keras.backend.clear_session()
    gc.collect()


def main():
    base_dir = Path(__file__).resolve().parent
    kaggle_paths = list(Path('/kaggle/input').rglob('cwt_scalograms_FULL.h5'))
    h5_path = kaggle_paths[0] if kaggle_paths else base_dir / 'data' / 'processed' / 'cwt_scalograms_FULL.h5'

    if not h5_path.exists():
        print(f"Błąd: Brak pliku danych: {h5_path}")
        return

    models_iter1 = {
        'Baseline': 'best_baseline.keras',
        'DenseNet121': 'best_densenet121.keras'
    }

    models_iter2 = {
        'Baseline': 'best_baseline_iter2.keras',
        'DenseNet121': 'best_densenet121_iter2.keras',
        'MobileNetV2': 'best_mobilenetv2_iter2.keras'
    }

    models_iter3 = {
        'Baseline': 'best_baseline_iter3.keras',
        'DenseNet121': 'best_densenet121_iter3.keras',
        'MobileNetV2': 'best_mobilenetv2_iter3.keras'
    }

    # ==========================================
    # WYKONANIE DLA ITERACJI 3
    # ==========================================
    run_error_analysis(
        h5_path=h5_path,
        models_dir=base_dir / 'models' / 'iter3',
        iter_name='iter3',
        crop=True,
        model_filenames=models_iter3
    )

    # ==========================================
    # WYKONANIE DLA ITERACJI 2
    # ==========================================
    """
    run_error_analysis(
        h5_path=h5_path,
        models_dir=base_dir / 'models' / 'iter2',
        iter_name='iter2',
        crop=True,
        model_filenames=models_iter2
    )
    """

    # ==========================================
    # WYKONANIE DLA ITERACJI 1
    # ==========================================
    """
    run_error_analysis(
        h5_path=h5_path,
        models_dir=base_dir / 'models',
        iter_name='iter1',
        crop=False,
        model_filenames=models_iter1
    )
    """

if __name__ == '__main__':
    main()