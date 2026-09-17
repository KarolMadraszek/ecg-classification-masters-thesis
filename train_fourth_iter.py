import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Dostępne urządzenia fizyczne:", tf.config.list_physical_devices('GPU'))

import pickle
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve
from src.generator import H5MemorySafeGenerator

NUM_CLASSES = 5
INPUT_SHAPE = (39, 900, 12)
CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
H5_PATH = '/kaggle/input/datasets/karolmadraszek/ptbxl-cwt-scalograms/cwt_scalograms_FULL.h5'
OUT_DIR = Path('models/iter4')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def build_baseline(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs, name='baseline')

def build_transfer_model(model_name, input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(3, (1, 1), activation='relu', name='channel_compressor')(inputs)

    if model_name == 'densenet121':
        base_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_tensor=(input_shape[0], input_shape[1], 3))
    elif model_name == 'mobilenetv2':
        base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_tensor=(input_shape[0], input_shape[1], 3))
    else:
        raise ValueError("Nieznany model transferowy.")

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    x = base_model(x)

    out = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dropout(0.3)(out)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid')(out)
    return tf.keras.Model(inputs, outputs, name=model_name)


def calculate_optimal_thresholds(y_true, y_pred, method='f2'):
    """
    Dostępne metody: 'youden' (statystyka J - kryterium Youdena) lub 'f2' (maksymalizacja F2-Score).
    """
    thresholds_dict = {}

    for i, class_name in enumerate(CLASSES):
        if method == 'youden':
            fpr, tpr, thresholds = roc_curve(y_true[:, i], y_pred[:, i])
            # Statystyka Youdena J = TPR - FPR
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            best_thresh = np.clip(thresholds[best_idx], 0.0, 1.0)

        elif method == 'f2':
            precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])

            precision = precision[:-1]
            recall = recall[:-1]

            denominator = (4 * precision) + recall
            # F2 = 5 * (Precision * Recall) / (4 * Precision + Recall)
            f2_scores = np.divide(
                5 * (precision * recall),
                denominator,
                out=np.zeros_like(precision),
                where=denominator != 0
            )

            best_idx = np.argmax(f2_scores)
            best_thresh = thresholds[best_idx]
        else:
            raise ValueError(f"Nieznana metoda: {method}. Należy wybrać 'youden' lub 'f2'.")

        thresholds_dict[class_name] = float(best_thresh)
        print(f"[{class_name}] Optymalny próg ({method}): {best_thresh:.4f}")

    return thresholds_dict

def get_predictions_and_true_labels(model, val_gen):
    y_true_all = []
    y_pred_all = []

    print("Trwa predykcja na zbiorze walidacyjnym do wyznaczenia progów")
    for i in range(len(val_gen)):
        X_batch, y_batch = val_gen[i]
        preds = model.predict(X_batch, verbose=0)
        y_true_all.extend(y_batch)
        y_pred_all.extend(preds)

    return np.array(y_true_all), np.array(y_pred_all)

def main():
    focal_loss = tf.keras.losses.BinaryFocalCrossentropy(
        gamma=2.0,
        alpha=0.25,
        apply_class_balancing=False
    )

    batch_size = 64

    # models_to_train = ['baseline', 'mobilenetv2', 'densenet121']
    models_to_train = ['mobilenetv2', 'densenet121']

    train_gen = H5MemorySafeGenerator(
        H5_PATH,
        split='train',
        batch_size=batch_size,
        crop=True,
        crop_range=(50, 950),
        taper_edges=False,
        freq_masking=False
    )

    val_gen = H5MemorySafeGenerator(
        H5_PATH,
        split='val',
        batch_size=batch_size,
        crop=True,
        crop_range=(50, 950),
        taper_edges=False,
        freq_masking=False
    )

    for m_name in models_to_train:
        print(f"\n{'=' * 50}")
        print(f"Rozpoczynanie uczenia modelu: {m_name.upper()}")
        print(f"{'=' * 50}")

        if m_name == 'baseline':
            model = build_baseline(INPUT_SHAPE)
        else:
            model = build_transfer_model(m_name, INPUT_SHAPE)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=focal_loss,
            metrics=[tf.keras.metrics.AUC(multi_label=True, name='auc')]
        )

        checkpoint_path = OUT_DIR / f'{m_name}_best_iter4.keras'
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_auc',
                                               mode='max'),
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc', mode='max')
        ]

        print("Trwa uczenie modelu")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=15,
            callbacks=callbacks,
        )

        history_path = OUT_DIR / f'history_{m_name}_iter4.pkl'
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)

        y_true, y_pred = get_predictions_and_true_labels(model, val_gen)
        best_thresholds = calculate_optimal_thresholds(y_true, y_pred, method='f2')

        thresh_path = OUT_DIR / f'thresholds_{m_name}_iter4.json'
        with open(thresh_path, 'w') as f:
            json.dump(best_thresholds, f, indent=4)

        print(f"Zakończono przetwarzanie modelu: {m_name}")

if __name__ == '__main__':
    main()