import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gc
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score
from src.generator import H5MemorySafeGenerator

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
MODELS = ['baseline', 'mobilenetv2', 'densenet121', 'resnet50v2', 'efficientnetb0']
N_BOOTSTRAP = 1000
RND_SEED = 42

def run_bootstrap(y_true, y_pred_prob, n_iterations=N_BOOTSTRAP, seed=RND_SEED):
    np.random.seed(seed)
    n_samples = len(y_true)

    auc_list = []
    f1_list = []

    y_pred_bin = (y_pred_prob > 0.5).astype(int)

    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_t_boot = y_true[indices]
        y_p_boot = y_pred_prob[indices]
        y_b_boot = y_pred_bin[indices]

        # Sprawdzenie czy w wylosowanej próbce występują obie klasy (0 i 1) dla każdego kanału
        valid_sample = True
        for col in range(y_true.shape[1]):
            if len(np.unique(y_t_boot[:, col])) < 2:
                valid_sample = False
                break
        if not valid_sample:
            continue

        try:
            auc = roc_auc_score(y_t_boot, y_p_boot, average='macro')
            f1 = f1_score(y_t_boot, y_b_boot, average='macro', zero_division=0)
            auc_list.append(auc)
            f1_list.append(f1)
        except ValueError:
            continue

    stats = {
        'auc_mean': np.mean(auc_list),
        'auc_std': np.std(auc_list),
        'auc_ci_lower': np.percentile(auc_list, 2.5),
        'auc_ci_upper': np.percentile(auc_list, 97.5),
        'f1_mean': np.mean(f1_list),
        'f1_std': np.std(f1_list),
        'f1_ci_lower': np.percentile(f1_list, 2.5),
        'f1_ci_upper': np.percentile(f1_list, 97.5),
    }
    return stats


def main():
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / 'models'

    kaggle_paths = list(Path('/kaggle/input').rglob('cwt_scalograms_FULL.h5'))
    h5_path = kaggle_paths[0] if kaggle_paths else base_dir / 'data' / 'processed' / 'cwt_scalograms_FULL.h5'

    print(f"Dane testowe: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        y_true = f['test']['y'][:]

    results = []

    for model_name in MODELS:
        model_file = models_dir / f"best_{model_name}.keras"
        if not model_file.exists():
            print(f"\nPominięto ewaliację modelu: Brak pliku wag dla: {model_name}")
            continue

        print(f"\n--- Ewaluacja: {model_name} ---")
        model = tf.keras.models.load_model(model_file)
        test_gen = H5MemorySafeGenerator(h5_path, split='test', batch_size=64)

        y_pred_prob = model.predict(test_gen, verbose=0)
        y_pred_prob = y_pred_prob[:len(y_true)]

        print(f"Obliczanie {N_BOOTSTRAP} iteracji bootstrapingu")
        stats = run_bootstrap(y_true, y_pred_prob)

        results.append({
            'Model': model_name,
            'ROC-AUC (Średnia ± STD)': f"{stats['auc_mean']:.4f} ± {stats['auc_std']:.4f}",
            'ROC-AUC 95% przedziału ufności': f"[{stats['auc_ci_lower']:.4f}, {stats['auc_ci_upper']:.4f}]",
            'F1-score (Średnia ± STD)': f"{stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}",
            'F1-score 95% przedziału ufności': f"[{stats['f1_ci_lower']:.4f}, {stats['f1_ci_upper']:.4f}]"
        })

        del model
        tf.keras.backend.clear_session()
        gc.collect()

    df = pd.DataFrame(results)
    out_csv = models_dir / "bootstrap_results.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("                      WYNIKI STATYSTYCZNE (BOOTSTRAP)")
    print("=" * 80)
    print(df.to_string(index=False))
    print(f"\nWyniki zapisano do pliku: {out_csv}")


if __name__ == '__main__':
    main()