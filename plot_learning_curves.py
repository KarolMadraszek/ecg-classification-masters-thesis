import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

def plot_history(history_path, output_dir, model_name, iter_name):
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    keys = history.keys()
    loss_key = next((k for k in keys if 'loss' in k and 'val' not in k), 'loss')
    val_loss_key = f"val_{loss_key}"
    auc_key = next((k for k in keys if 'auc' in k and 'val' not in k), 'auc')
    val_auc_key = f"val_{auc_key}"

    num_epochs = len(history[loss_key])
    epochs_range = range(1, num_epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Krzywe uczenia - {model_name.upper()} ({iter_name.upper()})', fontsize=16)

    if loss_key in history and val_loss_key in history:
        ax1.plot(epochs_range, history[loss_key], label='Trening', marker='o', color='blue')
        ax1.plot(epochs_range, history[val_loss_key], label='Walidacja', marker='o', color='orange')
        ax1.set_title('Funkcja straty')
        ax1.set_xlabel('Epoka')
        ax1.set_ylabel('Binarna entropia krzyżowa')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    if auc_key in history and val_auc_key in history:
        ax2.plot(epochs_range, history[auc_key], label='Trening (Train)', marker='o', color='green')
        ax2.plot(epochs_range, history[val_auc_key], label='Walidacja (Val)', marker='o', color='red')
        ax2.set_title('Metryka AUC-ROC')
        ax2.set_xlabel('Epoka')
        ax2.set_ylabel('Wartość AUC')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = output_dir / f"learning_curve_{model_name}_{iter_name}.png"
    plt.savefig(out_img_path, dpi=300)
    plt.close(fig)
    print(f"Zapisano wykres uczenia: {out_img_path.name}")

def main():
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / 'models'

    configurations = [
        {
            'iter_name': 'iter4',
            'folder': models_dir / 'iter4',
            'models': ['baseline', 'densenet121', 'mobilenetv2']
        }
    ]

    for config in configurations:
        iter_name = config['iter_name']
        folder = config['folder']

        print("\n" + "=" * 50)
        print(f"Generowanie wykresów dla: {iter_name.upper()}")
        print("=" * 50)

        for model_name in config['models']:
            pkl_path = folder / f"history_{model_name}_{iter_name}.pkl"

            if not pkl_path.exists():
                pkl_path = folder / f"history_{model_name}.pkl"

            if pkl_path.exists():
                plot_history(
                    history_path=pkl_path,
                    output_dir=folder,
                    model_name=model_name,
                    iter_name=iter_name
                )
            else:
                print(f"Brak pliku .pkl dla modelu {model_name} w iteracji {iter_name}. (Szukano: {pkl_path.name})")

if __name__ == '__main__':
    main()