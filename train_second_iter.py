import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import h5py
import numpy as np
import tensorflow as tf
from pathlib import Path

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
NUM_CLASSES = len(CLASSES)
TIME_STEPS = 900
FREQ_STEPS = 39
CHANNELS = 12

class H5MemorySafeGenerator(tf.keras.utils.Sequence):
    def __init__(self, h5_path, split='train', batch_size=32):
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

def build_baseline_cnn(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs, name="baseline")

def build_transfer_model(model_name, input_shape):
    """Architektury transferowe (wymagają redukcji z 12 do 3 kanałów RGB)."""
    inputs = tf.keras.Input(shape=input_shape)
    # Warstwa kompresująca 12 odprowadzeń do 3 warstw wizyjnych
    x = tf.keras.layers.Conv2D(3, (1, 1), activation='relu', name='channel_compressor')(inputs)

    if model_name == 'densenet121':
        base_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_tensor=x)
    elif model_name == 'mobilenetv2':
        base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_tensor=x)
    else:
        raise ValueError("Nieznany model transferowy.")

    # Odblokowanie tylko najwyższych warstw (fine-tuning)
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    out = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    out = tf.keras.layers.Dropout(0.3)(out)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid')(out)
    return tf.keras.Model(inputs, outputs, name=model_name)


def main():
    base_dir = Path(__file__).resolve().parent
    h5_path = base_dir / 'data' / 'processed' / 'cwt_scalograms_CROPPED.h5'

    # NOWY FOLDER NA ITERACJĘ 2
    out_dir = base_dir / 'models' / 'iter2'
    out_dir.mkdir(parents=True, exist_ok=True)

    if not h5_path.exists():
        print(f"Brak pliku danych: {h5_path}")
        return

    batch_size = 32
    epochs = 20  # Zmień w razie potrzeby

    train_gen = H5MemorySafeGenerator(h5_path, split='train', batch_size=batch_size)
    val_gen = H5MemorySafeGenerator(h5_path, split='val', batch_size=batch_size)

    models_to_train = ['baseline', 'densenet121', 'mobilenetv2']
    input_shape = (FREQ_STEPS, TIME_STEPS, CHANNELS)  # (39, 900, 12)

    for m_name in models_to_train:
        print("\n" + "=" * 50)
        print(f"ROZPOCZĘCIE TRENINGU ITERACJI 2: {m_name.upper()}")
        print("=" * 50)

        # Budowa modelu
        if m_name == 'baseline':
            model = build_baseline_cnn(input_shape)
        else:
            model = build_transfer_model(m_name, input_shape)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC(multi_label=True, name='auc')]
        )

        model_path = out_dir / f"best_{m_name}_iter2.keras"

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks
        )

        print(f"Zakończono trening {m_name.upper()}. Zapisano w {out_dir}")


if __name__ == '__main__':
    main()