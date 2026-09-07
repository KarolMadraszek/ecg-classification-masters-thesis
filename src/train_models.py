import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pickle
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.data_generator import CWTDataGenerator
from src.transfer_models import build_transfer_model

def train_all_models():
    base_dir = Path(__file__).resolve().parent.parent
    h5_path = base_dir / 'data' / 'processed' / 'cwt_scalograms_FULL.h5'
    models_dir = base_dir / 'models'
    models_dir.mkdir(exist_ok=True)

    print("Inicjalizacja generatorów danych")
    train_gen = CWTDataGenerator(h5_path, split='train', batch_size=32)
    val_gen = CWTDataGenerator(h5_path, split='val', batch_size=32)

    architectures = ['resnet50v2', 'efficientnetb0', 'mobilenetv2' ,'densenet121']

    for arch in architectures:
        print(f"\n{'=' * 50}")
        print(f"Rozpoczęcie treningu modelu: {arch.upper()}")
        print(f"{'=' * 50}")

        model = build_transfer_model(model_name=arch)

        model_save_path = models_dir / f"best_{arch}.keras"
        callbacks = [
            ModelCheckpoint(filepath=model_save_path, monitor='val_auc', mode='max', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_auc', mode='max', patience=3, verbose=1, restore_best_weights=True)
        ]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,
            callbacks=callbacks
        )

        history_path = models_dir / f"history_{arch}.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)

        print(f"\nZakończono trening {arch}. Model i historia zapisane.")


if __name__ == '__main__':
    train_all_models()