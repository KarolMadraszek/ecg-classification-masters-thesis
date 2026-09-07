import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import roc_auc_score, classification_report, multilabel_confusion_matrix
from src.data_generator import CWTDataGenerator

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def main():
    parser = argparse.ArgumentParser(description="Ewaluacja wyuczonego modelu.")
    parser.add_argument('--model', type=str, required=True,
                        choices=['baseline', 'resnet50v2', 'efficientnetb0', 'mobilenetv2', 'densenet121'])
    parser.add_argument('--dev', action='store_true', help="Proszę użyć małego pliku _DEV.h5 do testu")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    if args.dev:
        h5_path = base_dir / 'data' / 'processed' / 'cwt_scalograms_DEV.h5'
        print("\nTRYB DEV")
    else:
        h5_path = base_dir / 'data' / 'processed' / 'cwt_scalograms_FULL.h5'

    model_path = base_dir / 'models' / f"best_{args.model}.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"Nie znaleziono wag modelu: {model_path}. Najpierw trzeba uruchomić train.py.")

    print(f"Ładowanie modelu: {args.model.upper()}")
    model = tf.keras.models.load_model(model_path)

    test_gen = CWTDataGenerator(h5_path, split='test', batch_size=32, shuffle=False)

    print("Generowanie predykcji na zbiorze testowym")
    y_pred_prob = model.predict(test_gen)

    with h5py.File(h5_path, 'r') as f:
        y_true = f['test']['y'][:]

    print("\n" + "=" * 40)
    print("Parametr ROC-AUC dla każdej klasy")
    print("=" * 40)
    auc_scores = []
    for i, class_name in enumerate(CLASSES):
        auc = roc_auc_score(y_true[:, i], y_pred_prob[:, i])
        auc_scores.append(auc)
        print(f"{class_name.ljust(5)}: {auc:.4f}")
    print(f"Średnia (Macro-AUC): {np.mean(auc_scores):.4f}")

    print("\n" + "=" * 40)
    print(" Raport klasyfikacji (próg = 0.5)")
    print("=" * 40)
    y_pred_binary = (y_pred_prob > 0.5).astype(int)
    print(classification_report(y_true, y_pred_binary, target_names=CLASSES, zero_division=0))

    mcm = multilabel_confusion_matrix(y_true, y_pred_binary)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, (ax, matrix, class_name) in enumerate(zip(axes, mcm, CLASSES)):
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title(f'Klasa: {class_name}')
        ax.set_xlabel('Predykcja')
        ax.set_ylabel('Prawda')

    plot_path = base_dir / 'models' / f"confusion_matrix_{args.model}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nZapisano wykresy macierzy pomyłek: {plot_path}")

if __name__ == '__main__':
    main()