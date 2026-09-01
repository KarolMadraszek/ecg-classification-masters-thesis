import pandas as pd
import numpy as np
import h5py
import wfdb
from pathlib import Path
from tqdm import tqdm

def build_signal_dataset(
    meta_dir: str = '../data/processed',
    raw_dir: str = '../data/raw',
    output_path: str = '../data/processed/signals_100hz.h5',
    dev_mode: bool = False
):
    base_meta = Path(meta_dir)
    base_raw = Path(raw_dir)
    out_file = Path(output_path)

    if out_file.exists():
        try:
            out_file.unlink()
            print(f"Usunięto istniejący plik: {out_file.name}")
        except PermissionError:
            print(
                f"Nie można usunąć {out_file.name} (uchwyt zablokowany). Tworzenie nowej struktury w trybie nadpisywania.")

    splits = {
        'train': base_meta / 'train_meta.csv',
        'val': base_meta / 'val_meta.csv',
        'test': base_meta / 'test_meta.csv'
    }
    target_classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

    with h5py.File(out_file, 'w') as h5f:
        for split_name, csv_path in splits.items():
            if not csv_path.exists():
                print(f"Brak pliku: {csv_path}. Uruchomić najpierw split_data.py!")
                continue

            print(f"\nPrzetwarzanie zbioru: {split_name.upper()}")
            df = pd.read_csv(csv_path, index_col='ecg_id')

            if dev_mode:
                df = df.head(100)
                print(f"Tryb deweloperski: skrócono zbiór do {len(df)} próbek.")

            filenames = df['filename_lr'].values  # Pobranie ścieżek do wersji 100 Hz
            labels = df[target_classes].values.astype(np.uint8)
            ecg_ids = df.index.values
            n_records = len(filenames)
            X = np.zeros((n_records, 1000, 12), dtype=np.float32)   # Alokacja macierzy dla 100Hz: (N, 1000 próbek, 12 odprowadzeń)

            for i, fname in enumerate(tqdm(filenames, desc=f"Wczytywanie WFDB ({split_name})")):
                record_path = str(base_raw / fname)
                signal, _ = wfdb.rdsamp(record_path)
                X[i] = signal

            # Zapis do pliku HDF5 z kompresją (zmniejsza rozmiar bez dużego obciążenia CPU) ---
            group = h5f.require_group(split_name)
            group.create_dataset('X', data=X, compression="gzip", compression_opts=4)
            group.create_dataset('y', data=labels)
            group.create_dataset('ecg_id', data=ecg_ids)

            print(f"Zapisano do HDF5: \nX={X.shape} \ny={labels.shape}")


if __name__ == "__main__":
    build_signal_dataset(
        output_path='../data/processed/signals_100hz_DEV.h5',
        dev_mode=True
    )

    # Opcja 2: Docelowe, pełne przetwarzanie (Odkomentuj, gdy kod będzie gotowy)
    # build_signal_dataset(
    #    output_path='data/processed/signals_100hz_FULL.h5',
    #    dev_mode=False
    # )


