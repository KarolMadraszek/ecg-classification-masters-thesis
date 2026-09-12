import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import h5py
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, recall_score, hamming_loss, multilabel_confusion_matrix
from src.generator import H5MemorySafeGenerator

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def main():
    base_dir = Path(__file__).resolve().parent
    h5_path = base_dir / 'data' / 'processed' / 'cwt_scalograms_FULL.h5'
    models_dir = base_dir / 'models'

    if not h5_path.exists():
        print(f"Brak pliku danych: {h5_path}")
        return

    with h5py.File(h5_path, 'r') as f:
        y_true = f['test']['y'][:]

    test_gen = H5MemorySafeGenerator(h5_path, split='test', batch_size=64, crop=True, crop_range=(50, 950)) 

    model_files = {
        'Baseline': models_dir / "best_baseline.keras",
        'DenseNet121': models_dir / "best_densenet121.keras",
        'MobileNetV2': models_dir / "best_mobilenetv2.keras",
        'ResNet50v2': models_dir / "best_resnet50v2.keras",
        'EfficientNetB0': models_dir / "best_efficientnetb0.keras"
    }

    results = []

    for name, path in model_files.items():
        if not path.exists():
            print(f"Pominięto model {name}: brak pliku wag w {path}")
            continue

        print(f"Ewaluacja modelu: {name}")
        model = tf.keras.models.load_model(path)
        y_pred_prob = model.predict(test_gen, verbose=0)[:len(y_true)]
        y_pred_bin = (y_pred_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_true, y_pred_prob, average='macro')
        macro_f1 = f1_score(y_true, y_pred_bin, average='macro', zero_division=0)
        macro_rec = recall_score(y_true, y_pred_bin, average='macro', zero_division=0)
        hl = hamming_loss(y_true, y_pred_bin)
        recalls_per_class = recall_score(y_true, y_pred_bin, average=None, zero_division=0)

        row = {
            'Model': name,
            'Macro_ROC_AUC': round(auc, 4),
            'Macro_F1': round(macro_f1, 4),
            'Macro_Recall': round(macro_rec, 4),
            'Hamming_Loss': round(hl, 4)
        }

        for idx, cls in enumerate(CLASSES):
            row[f'Recall_{cls}'] = round(recalls_per_class[idx], 4)

        results.append(row)

        mcm = multilabel_confusion_matrix(y_true, y_pred_bin)
        fig, axes = plt.subplots(1, 5, figsize=(22, 4))

        for ax, matrix, cls_name in zip(axes, mcm, CLASSES):
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_title(f'Klasa: {cls_name}', fontsize=12)
            ax.set_xlabel('Predykcja')
            ax.set_ylabel('Rzeczywistość')

        plt.tight_layout()
        out_img = models_dir / f"confusion_matrix_{name}.png"
        plt.savefig(out_img, dpi=300)
        plt.close(fig)
        print(f" -> Macierz pomyłek zapisano jako wykres: {out_img.name}")

    df_results = pd.DataFrame(results)
    out_csv = models_dir / "iter1_detailed_evaluation.csv"
    df_results.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print(f"Podsumowanie zapisano w: {out_csv}")
    print("=" * 80)
    print(df_results.to_string(index=False))

if __name__ == '__main__':
    main()