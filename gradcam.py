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


def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        elif hasattr(layer, 'output') and hasattr(layer.output, 'shape'):
            if len(layer.output.shape) == 4 and 'conv' in layer.name.lower():
                return layer.name
    raise ValueError("Nie znaleziono warstwy konwolucyjnej 2D w modelu.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Pomnożenie każdego kanału mapy cech przez "ważność" tego kanału
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU (odrzucenie ujemnych wpływów) i normalizacja
    heatmap = tf.maximum(heatmap, 0)
    max_heat = tf.math.reduce_max(heatmap)
    if max_heat > 0:
        heatmap = heatmap / max_heat

    return heatmap.numpy()

def main():
    model_names = ['baseline', 'densenet121']

    base_dir = Path(__file__).resolve().parent
    h5_path = base_dir / 'data' / 'processed' / 'cwt_scalograms_FULL.h5'
    models_dir = base_dir / 'models'

    if not h5_path.exists():
        print(f"Brak pliku z danymi: {h5_path}")
        return

    loaded_models = {}
    print("Inicjalizacja modeli w pamięci")
    for m_name in model_names:
        model_path = models_dir / f"best_{m_name}.keras"
        if not model_path.exists():
            print(f"Brak modelu: {model_path}. Pominięto model.")
            continue
        print(f"Ładowanie wag: {m_name}")
        model = tf.keras.models.load_model(model_path)
        conv_layer = get_last_conv_layer_name(model)
        loaded_models[m_name] = {'model': model, 'conv_layer': conv_layer}

    if not loaded_models:
        print("Nie załadowano żadnych modeli")
        return

    with h5py.File(h5_path, 'r') as f:
        y_test = f['test']['y'][:]

        for target_class_idx, class_name in enumerate(CLASSES):
            print("\n" + "=" * 50)
            print(f"Generowanie Grad-CAM dla klasy: {class_name}")
            print("=" * 50)

            valid_indices = np.where(y_test[:, target_class_idx] == 1)[0]
            if len(valid_indices) == 0:
                print(f"Brak próbek dla klasy {class_name} w zbiorze testowym")
                continue

            selected_idx = valid_indices[0]
            sample_x = f['test']['X'][selected_idx:selected_idx + 1]
            print(f"Wybrano próbkę nr {selected_idx}.")

            fig, axes = plt.subplots(len(loaded_models), 1, figsize=(12, 5 * len(loaded_models)))
            if len(loaded_models) == 1:
                axes = [axes]

            for ax, (m_name, m_data) in zip(axes, loaded_models.items()):
                model = m_data['model']
                conv_layer = m_data['conv_layer']

                print(f"-> Analiza modelem: {m_name} (warstwa: {conv_layer})")

                preds = model.predict(sample_x, verbose=0)
                prob = preds[0][target_class_idx]

                heatmap = make_gradcam_heatmap(sample_x, model, conv_layer, pred_index=target_class_idx)
                bg_img = np.mean(sample_x[0], axis=-1)
                heatmap_resized = cv2.resize(heatmap, (bg_img.shape[1], bg_img.shape[0]))

                ax.imshow(bg_img, cmap='gray', aspect='auto')
                ax.imshow(heatmap_resized, cmap='jet', alpha=0.4, aspect='auto')
                ax.set_title(f"Model: {m_name.upper()} | Predykcja: {class_name} ({prob * 100:.1f}%)", fontsize=14,
                             fontweight='bold')
                ax.set_xlabel("Czas (próbki CWT)")
                ax.set_ylabel("Skala częstotliwości")

            plt.tight_layout()
            out_img = models_dir / f"gradcam_comparison_{class_name}.png"
            plt.savefig(out_img, dpi=300)
            plt.close(fig)
            print(f"Zapisano wizualizację do: {out_img}")

if __name__ == '__main__':
    main()