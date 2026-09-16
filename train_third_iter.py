import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Dostępne urządzenia fizyczne:", tf.config.list_physical_devices('GPU'))

import argparse
import h5py
import numpy as np
import pickle
from src.generator import H5MemorySafeGenerator
from pathlib import Path

H5_PATH = '/kaggle/input/datasets/karolmadraszek/ptbxl-cwt-scalograms/cwt_scalograms_FULL.h5'
MODEL_DIR = Path('/kaggle/working/project/models')
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
NUM_CLASSES = len(CLASSES)
TIME_STEPS = 900
FREQ_STEPS = 39
CHANNELS = 12

def get_positive_weights(h5_path, split='train'):
    with h5py.File(h5_path, 'r') as f:
        y_train = f[split]['y'][:]

    total_samples = y_train.shape[0]
    pos_samples = np.sum(y_train, axis=0)
    neg_samples = total_samples - pos_samples

    pos_weights = neg_samples / pos_samples

    print("--- Rozkład klas w zbiorze treningowym ---")
    print(f"Ilość próbek pozytywnych na klasę: {pos_samples}")
    print(f"Wyliczone wagi (pos_weights): {pos_weights}")

    return tf.constant(pos_weights, dtype=tf.float32)

def weighted_binary_crossentropy(pos_weights):
    def custom_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        # Ochrona przed log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        bce = y_true * -tf.math.log(y_pred) * pos_weights + \
              (1 - y_true) * -tf.math.log(1 - y_pred)
        return tf.reduce_mean(bce, axis=-1)
    return custom_loss

def build_baseline_cnn(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(inputs)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def build_transfer_model(model_name):
    model_name = model_name.lower()
    input_shape = (FREQ_STEPS, TIME_STEPS, CHANNELS)

    if model_name == 'baseline':
        return build_baseline_cnn(input_shape)

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        3, (1, 1),
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name='channel_compression'
    )(inputs)

    x = tf.keras.layers.SpatialDropout2D(0.2, name='spatial_dropout')(x)

    base_input_shape = (input_shape[0], input_shape[1], 3)

    if model_name == 'densenet121':
        base_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_tensor=base_input_shape)
    elif model_name == 'mobilenetv2':
        base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_tensor=base_input_shape)
    else:
        raise ValueError(f"Nieobsługiwany model: {model_name}. Należy wybrać: baseline, densenet121 lub mobilenetv2.")

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['baseline', 'densenet121', 'mobilenetv2'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    print(f"--- ROZPOCZĘCIE TRENINGU: ITERACJA 3 ({args.model.upper()}) ---")

    pos_weights = get_positive_weights(H5_PATH)

    train_gen = H5MemorySafeGenerator(
        H5_PATH, split='train', batch_size=args.batch_size, shuffle=True,
        crop=True, crop_range=(50, 950),
        taper_edges=True, freq_masking=True
    )
    val_gen = H5MemorySafeGenerator(
        H5_PATH, split='val', batch_size=args.batch_size, shuffle=False,
        crop=True, crop_range=(50, 950),
        taper_edges=True, freq_masking=False
    )

    model = build_transfer_model(model_name=args.model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=weighted_binary_crossentropy(pos_weights),
        metrics=[
            tf.keras.metrics.AUC(multi_label=True, name='auc'),
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
        ]
    )

    checkpoint_path = os.path.join(MODEL_DIR, f'best_{args.model}_iter3.keras')

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )

    history_path = MODEL_DIR / f'history_{args.model}_iter3.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)

    print(f"Zakończono trening {args.model.upper()}. Zapisano w {MODEL_DIR}")