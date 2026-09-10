import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import pickle
import h5py
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from src.baseline_cnn import build_baseline_cnn
from src.transfer_models import build_transfer_model

class H5MemorySafeGenerator(tf.keras.utils.Sequence):
    def __init__(self, h5_path, split, batch_size=32):
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

def main():
    parser = argparse.ArgumentParser(description="Trening modeli na skalogramach CWT strumieniowo.")
    parser.add_argument('--model', type=str, required=True,
                        choices=['baseline', 'resnet50v2', 'efficientnetb0', 'mobilenetv2', 'densenet121'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / 'models'
    models_dir.mkdir(exist_ok=True)

    if args.data_path:
        h5_path = Path(args.data_path)
    else:
        kaggle_paths = list(Path('/kaggle/input').rglob('cwt_scalograms_FULL.h5'))
        h5_path = kaggle_paths[0] if kaggle_paths else base_dir / 'data' / 'processed' / 'cwt_scalograms_FULL.h5'

    if not h5_path.exists():
        raise FileNotFoundError(f"Brak pliku: {h5_path}")

    print(f"\nInicjalizacja generatorów strumieniowych z: {h5_path}")
    train_gen = H5MemorySafeGenerator(h5_path, split='train', batch_size=args.batch_size)
    val_gen = H5MemorySafeGenerator(h5_path, split='val', batch_size=args.batch_size)

    print(f"\nBudowanie struktury modelu: {args.model.upper()}")
    if args.model == 'baseline':
        model = build_baseline_cnn(input_shape=(39, 1000, 12), num_classes=5)
    else:
        model = build_transfer_model(model_name=args.model, input_shape=(39, 1000, 12), num_classes=5)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(multi_label=True, name='auc')]
    )

    model_save_path = models_dir / f"best_{args.model}.keras"
    callbacks = [
        ModelCheckpoint(filepath=model_save_path, monitor='val_auc', mode='max', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_auc', mode='max', patience=3, verbose=1, restore_best_weights=True)
    ]

    print("\nRozpoczęcie treningu strumieniowego")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )

    history_path = models_dir / f"history_{args.model}.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)

    print(f"\nTrening zakończony. Zapisano w: {model_save_path}")

if __name__ == '__main__':
    main()