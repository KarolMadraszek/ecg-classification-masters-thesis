import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix,
                             roc_curve, recall_score, hamming_loss)
from src.generator import H5MemorySafeGenerator

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def load_thresholds(thresholds_path, default_val=0.5):
    if thresholds_path.exists():
        with open(thresholds_path, 'r') as f:
            data = json.load(f)
        return np.array([data.get(c, default_val) for c in CLASSES])
    return np.full(len(CLASSES), default_val)

def plot_multiclass_roc(y_true, y_pred, roc_aucs, model_name, save_path):
    plt.figure(figsize=(10, 8))
    for i, (c_name, auc) in enumerate(zip(CLASSES, roc_aucs)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'{c_name} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Krzywe ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrices(y_true, y_pred_bin, model_name, save_path):
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(25, 5))
    fig.suptitle(f'Macierze pomyłek - {model_name}', fontsize=16)

    for i, c_name in enumerate(CLASSES):
        cm = confusion_matrix(y_true[:, i], y_pred_bin[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i])
        axes[i].set_title(c_name)
        axes[i].set_xlabel('Predykcja')
        axes[i].set_ylabel('Prawda')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    base_dir = Path(__file__).resolve().parent
    h5_path = base_dir / 'data' / 'processed' / 'cwt_scalograms_FULL.h5'
    models_dir = base_dir / 'models' / 'iter4'
    out_dir = models_dir / 'evaluation_results'
    out_dir.mkdir(parents=True, exist_ok=True)

    if not h5_path.exists():
        print(f"Błąd: Brak pliku {h5_path}")
        return

    models_files = {
        'Baseline': 'baseline_best_iter4.keras',
        'DenseNet121': 'densenet121_best_iter4.keras',
        'MobileNetV2': 'mobilenetv2_best_iter4.keras'
    }

    test_gen = H5MemorySafeGenerator(str(h5_path), split='test', batch_size=64, crop=True, crop_range=(50, 950), shuffle=False)

    with h5py.File(h5_path, 'r') as f:
        y_true = f['test']['y'][:]

    if hasattr(test_gen, 'on_epoch_end'):
        test_gen.on_epoch_end()

    results_summary = []

    for m_name, filename in models_files.items():
        model_path = models_dir / filename
        if not model_path.exists():
            continue

        print(f"\nRozpoczęto ewaluację modelu: {m_name}")
        model = tf.keras.models.load_model(model_path, compile=False)
        y_pred = model.predict(test_gen, verbose=1)[:len(y_true)]

        thresh_vals = load_thresholds(models_dir / f'thresholds_{m_name.lower()}_iter4.json')
        y_pred_bin = (y_pred >= thresh_vals).astype(int)

        roc_aucs = [roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(len(CLASSES))]
        pr_aucs = [average_precision_score(y_true[:, i], y_pred[:, i]) for i in range(len(CLASSES))]
        report = classification_report(y_true, y_pred_bin, target_names=CLASSES, output_dict=True, zero_division=0)
        sensitivity_per_class = {f"Recall_{c}": report[c]['recall'] for c in CLASSES}

        macro_recall = recall_score(y_true, y_pred_bin, average='macro', zero_division=0)
        h_loss = hamming_loss(y_true, y_pred_bin)

        plot_multiclass_roc(y_true, y_pred, roc_aucs, m_name, out_dir / f'roc_curve_{m_name}_iter4.png')
        plot_confusion_matrices(y_true, y_pred_bin, m_name, out_dir / f'confusion_matrix_{m_name}_iter4.png')

        pd.DataFrame(report).transpose().to_csv(out_dir / f'classification_report_{m_name}_iter4.csv')

        result_entry = ({
            'Model': m_name,
            'Macro_ROC_AUC': round(float(np.mean(roc_aucs)), 4),
            'Macro_PR_AUC': round(float(np.mean(pr_aucs)), 4),
            'Macro_F1': round(report['macro avg']['f1-score'], 4),
            'Macro_Recall': round(macro_recall, 4),
            'Hamming_Loss': round(h_loss, 4)
        })

        for c_name in CLASSES:
            result_entry[f'Recall_{c_name}'] = round(report[c_name]['recall'], 4)

        results_summary.append(result_entry)

        del model
        tf.keras.backend.clear_session()

    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(out_dir / 'iter4_summary.csv', index=False)
        print("\n=== PODSUMOWANIE ITERACJI 4 ===")
        print(summary_df.to_markdown(index=False))

if __name__ == '__main__':
    main()