import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import roc_auc_score, classification_report, multilabel_confusion_matrix

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

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
    parser = argparse.ArgumentParser(description="Ewaluacja modeli na zbiorze testowym.")
    parser.add_argument('--model', type=str, required=True,
                        choices=['baseline', 'resnet50v2', 'efficientnetb0', 'mobilenetv2', 'densenet121'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / 'models'

    if args.data_path:
        h5_path = Path(args.data_path)
    else:
        kaggle_paths = list(Path('/kaggle/input').rglob('cwt_scalograms_FULL.h5'))
        h5_path = kaggle_paths[0] if kaggle_paths else base_dir / 'data' / 'processed' / 'cwt_scalograms_FULL.h5'

    if not h5_path.exists():
        raise FileNotFoundError(f"Brak pliku danych: {h5_path}")

    model_path = models_dir / f"best_{args.model}.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Brak zapisanego modelu: {model_path}")

    print(f"Używany plik danych: {h5_path}")
    print(f"Ładowanie wag modelu z: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print("Przygotowywanie generatora zbioru testowego")
    test_gen = H5MemorySafeGenerator(h5_path, split='test', batch_size=args.batch_size)

    print("Wykonywanie predykcji na zbiorze testowym")
    y_pred_prob = model.predict(test_gen, verbose=1)

    with h5py.File(h5_path, 'r') as f:
        y_true = f['test']['y'][:]

    y_pred_prob = y_pred_prob[:len(y_true)]

    print("\n" + "="*50)
    print("Wyniki ROC-AUC")
    print("="*50)
    aucs = []
    for i, name in enumerate(CLASSES):
        auc = roc_auc_score(y_true[:, i], y_pred_prob[:, i])
        aucs.append(auc)
        print(f" Klasa {name.ljust(6)}: {auc:.4f}")
    print(f" Średnie macro ROC-AUC: {np.mean(aucs):.4f}")

    print("\n" + "="*50)
    print("Raport klasyfikacji (próg = 0.5)")
    print("="*50)
    y_pred_bin = (y_pred_prob > 0.5).astype(int)
    print(classification_report(y_true, y_pred_bin, target_names=CLASSES, zero_division=0))

    # Wykres macierzy pomyłek dla klas wieloetykietowych
    mcm = multilabel_confusion_matrix(y_true, y_pred_bin)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, matrix, name in zip(axes, mcm, CLASSES):
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title(f'Klasa: {name}', fontsize=12)
        ax.set_xlabel('Predykcja')
        ax.set_ylabel('Rzeczywistość')

    plt.tight_layout()
    out_img = models_dir / f"confusion_matrix_{args.model}.png"
    plt.savefig(out_img, dpi=300)
    print(f"\nMacierz pomyłek zapisano jako wykres: {out_img}")

if __name__ == '__main__':
    main()