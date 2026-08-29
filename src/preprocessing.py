import ast                  # Abstract Syntax Tree
import h5py
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm       # Paski postępu
from pathlib import Path

def convert_ptbxl_to_hdf5(raw_dir_path: str, processed_dir_path: str):
    raw_dir = Path(raw_dir_path)
    processed_dir = Path(processed_dir_path)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("1. Parsowanie metadanych i kodów SCP")
    df = pd.read_csv(raw_dir / "ptbxl_database.csv", index_col="ecg_id")
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

    agg_df = pd.read_csv(raw_dir / "scp_statements.csv", index_col=0)
    diag_agg_df = agg_df[agg_df['diagnostic'] == 1]
    scp_to_superclass = diag_agg_df['diagnostic_class'].dropna().to_dict()

    def get_superclasses(scp_dict):
        classes = set()
        for key in scp_dict.keys():
            if key in scp_to_superclass:
                val = scp_to_superclass[key]
                if isinstance(val, str):
                    classes.add(val)
        return list(classes)

    df['diagnostic_superclasses'] = df['scp_codes'].apply(get_superclasses)

    target_classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    for c in target_classes:
        df[c] = df['diagnostic_superclasses'].apply(lambda x: 1 if c in x else 0)

    metadata_out = processed_dir / "ptbxl_metadata_prepared.csv"
    df.to_csv(metadata_out)
    print(f"Zapisano metadane do: {metadata_out}")

    print("\n2. Wczytywanie surowych sygnałów 100 Hz do tablicy NumPy")
    filenames = df['filename_lr'].values
    num_records = len(filenames)

    # Alokacja pamięci dla: 21799 pacjentów x 1000 próbek (10s) x 12 kanałów
    signals = np.zeros((num_records, 1000, 12), dtype=np.float32)

    for i, fname in enumerate(tqdm(filenames, desc="Wczytywanie WFDB")):
        record_path = str(raw_dir / fname)
        data, meta = wfdb.rdsamp(record_path)
        signals[i] = data

    print("\n3. Zapisywanie do zoptymalizowanego kontenera HDF5")
    h5_path = processed_dir / "raw_signals_100hz.h5"

    with h5py.File(h5_path, "w") as h5f:
        # Kompresja gzip zmniejsza rozmiar pliku o połowę kosztem minimalnego narzutu CPU
        h5f.create_dataset("signals", data=signals, compression="gzip", compression_opts=4)
        h5f.create_dataset("labels", data=df[target_classes].values.astype(np.uint8))
        h5f.create_dataset("ecg_id", data=df.index.values)
        h5f.create_dataset("strat_fold", data=df['strat_fold'].values)

    print("Dane umieszczone w pliku HDF5.")

if __name__ == "__main__":
    convert_ptbxl_to_hdf5(raw_dir_path="../data/raw", processed_dir_path="../data/processed")