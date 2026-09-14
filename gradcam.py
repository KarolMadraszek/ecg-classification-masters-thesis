import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import cv2

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def make_gradcam_heatmap(img_array, model, pred_index=None):
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
        conv_outputs, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_heat = tf.math.reduce_max(heatmap)
    if max_heat > 0:
        heatmap = heatmap / max_heat

    return heatmap.numpy()

def generate_gradcam_for_iteration(h5_path, models_dir, iter_name, crop, model_filenames):
    print("\n" + "=" * 80)
    print(f"GENEROWANIE GRAD-CAM: {iter_name.upper()}")
    print("=" * 80)

    models_dir.mkdir(parents=True, exist_ok=True)
    loaded_models = {}

    print("Inicjalizacja modeli w pamięci...")
    for m_name, filename in model_filenames.items():
        model_path = models_dir / filename
        if not model_path.exists():
            print(f" -> Brak modelu: {model_path}. Pominięto.")
            continue

        print(f" -> Ładowanie wag: {m_name}")
        model = tf.keras.models.load_model(model_path)
        loaded_models[m_name] = model

    if not loaded_models:
        print(f"Nie załadowano żadnych modeli dla {iter_name}.")
        return

    with h5py.File(h5_path, 'r') as f:
        y_test = f['test']['y'][:]
        x_test_dataset = f['test']['X']

        for target_class_idx, class_name in enumerate(CLASSES):
            print(f"\n--- Klasa: {class_name} ---")

            valid_indices = np.where(y_test[:, target_class_idx] == 1)[0]
            if len(valid_indices) == 0:
                print(f"Brak próbek dla klasy {class_name} w zbiorze testowym")
                continue

            selected_idx = valid_indices[0]
            sample_x = x_test_dataset[selected_idx:selected_idx + 1]

            if crop:
                sample_x = sample_x[:, :, 50:950, :]

            print(f"Wybrano próbkę nr {selected_idx}. Kształt wejścia: {sample_x.shape}")

            fig, axes = plt.subplots(len(loaded_models), 1, figsize=(12, 5 * len(loaded_models)))
            if len(loaded_models) == 1:
                axes = [axes]

            for ax, (m_name, model) in zip(axes, loaded_models.items()):
                print(f"  > Analiza: {m_name}")

                preds = model.predict(sample_x, verbose=0)
                prob = preds[0][target_class_idx]

                heatmap = make_gradcam_heatmap(sample_x, model, pred_index=target_class_idx)

                bg_img = np.mean(sample_x[0], axis=-1)
                heatmap_resized = cv2.resize(heatmap, (bg_img.shape[1], bg_img.shape[0]))

                ax.imshow(bg_img, cmap='gray', aspect='auto')
                ax.imshow(heatmap_resized, cmap='jet', alpha=0.4, aspect='auto')
                ax.set_title(f"Model: {m_name.upper()} | Predykcja: {class_name} ({prob * 100:.1f}%)", fontsize=14,
                             fontweight='bold')
                ax.set_xlabel("Czas (próbki CWT)")
                ax.set_ylabel("Skala częstotliwości")

            plt.tight_layout()
            out_img = models_dir / f"gradcam_comparison_{class_name}_{iter_name}.png"
            plt.savefig(out_img, dpi=300)
            plt.close(fig)
            print(f"  > Zapisano wizualizację do: {out_img.name}")

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

    # ==========================================
    # WYKONANIE DLA ITERACJI 2
    # ==========================================
    generate_gradcam_for_iteration(
        h5_path=h5_path,
        models_dir=base_dir / 'models' / 'iter2',
        iter_name='iter2',
        crop=True,
        model_filenames=models_iter2
    )

    # ==========================================
    # WYKONANIE DLA ITERACJI 1
    # ==========================================
    """
    generate_gradcam_for_iteration(
        h5_path=h5_path,
        models_dir=base_dir / 'models',
        iter_name='iter1',
        crop=False,
        model_filenames=models_iter1
    )
    """

if __name__ == '__main__':
    main()