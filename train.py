import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import pickle
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from src.data_generator import CWTDataGenerator
from src.baseline_cnn import build_baseline_cnn
from src.transfer_models import build_transfer_model

def main():
    parser = argparse.ArgumentParser(description="Trening modeli na skalogramach CWT.")
    parser.add_argument('--model', type=str, required=True,
                        choices=['baseline', 'resnet50v2', 'efficientnetb0', 'mobilenetv2', 'densenet121'],
                        help="Proszę wybrać architekturę do uczenia")
    parser.add_argument('--epochs', type=int, default=15, help="Maksymalna liczba epok")
    parser.add_argument('--batch_size', type=int, default=32, help="Rozmiar paczki")
    parser.add_argument('--dev', action='store_true', help="Proszę użyć małego pliku _DEV.h5 do testu")
    parser.add_argument('--data_path', type=str, default=None, help="Opcjonalna ścieżka do pliku .h5 (np. na Kaggle)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / 'models'
    models_dir.mkdir(exist_ok=True)

    if args.data_path:
        h5_path = Path(args.data_path)
    elif args.dev:
        h5_path = base_dir / 'data' / 'processed' / 'cwt_scalograms_DEV.h5'
        print("\nTRYB DEV")
    else:
        kaggle_paths = list(Path('/kaggle/input').rglob('cwt_scalograms_FULL.h5'))
        if kaggle_paths:
            h5_path = kaggle_paths[0]
            print("\nAutomatycznie wykryto plik na Kaggle")
        else:
            h5_path = base_dir / 'data' / 'processed' / 'cwt_scalograms_FULL.h5'

    if not h5_path.exists():
        raise FileNotFoundError(f"Błąd: Nie znaleziono pliku z danymi {h5_path}")

    print(f"\nInicjalizacja danych z: {h5_path}")
    train_gen = CWTDataGenerator(h5_path, split='train', batch_size=args.batch_size)
    val_gen = CWTDataGenerator(h5_path, split='val', batch_size=args.batch_size)

    print(f"\nBudowanie struktury modelu: {args.model.upper()}")
    if args.model == 'baseline':
        model = build_baseline_cnn(input_shape=(39, 1000, 12), num_classes=5)
    else:
        model = build_transfer_model(model_name=args.model, input_shape=(39, 1000, 12), num_classes=5)

    model_save_path = models_dir / f"best_{args.model}.keras"
    callbacks = [
        ModelCheckpoint(filepath=model_save_path, monitor='val_auc', mode='max', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_auc', mode='max', patience=3, verbose=1, restore_best_weights=True)
    ]

    print("\nRozpoczęcie treningu")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        workers = 4,
        use_multiprocessing = True
    )

    history_path = models_dir / f"history_{args.model}.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)

    print(f"\nTrening pomyślnie zakończony. Najlepsze wagi zapisano jako:\n{model_save_path}")

if __name__ == '__main__':
    main()