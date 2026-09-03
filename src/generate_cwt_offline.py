import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
from src.baseline_cnn import convert_patient_to_tensor

def process_and_save_cwt(input_h5_path, output_h5_path):
    print(f"Otwieranie pliku wejściowego: {input_h5_path}")

    with h5py.File(input_h5_path, 'r') as f_in, h5py.File(output_h5_path, 'w') as f_out:
        for split in ['train', 'val', 'test']:
            if split not in f_in:
                print(f"Pomijam '{split}' - brak grupy w pliku.")
                continue
            print(f"\nPrzetwarzanie grupy: {split}")
            group_in = f_in[split]
            group_out = f_out.create_group(split)

            num_samples = group_in['X'].shape[0]

            ds_x_out = group_out.create_dataset(
                'X',
                shape=(num_samples, 39, 1000, 12),
                dtype=np.float16,
                chunks=(1, 39, 1000, 12)
            )

            print(f"Kopiowanie etykiet (y) i ID (ecg_id) dla {split}...")
            group_out.create_dataset('y', data=group_in['y'][:])
            group_out.create_dataset('ecg_id', data=group_in['ecg_id'][:])

            print(f"Generowanie skalogramów dla {num_samples} pacjentów...")
            for i in tqdm(range(num_samples), desc=f"CWT {split}"):
                raw_signal = group_in['X'][i]
                cwt_tensor = convert_patient_to_tensor(raw_signal)
                ds_x_out[i] = cwt_tensor.astype(np.float16)

    print(f"\nGotowy plik ze skalogramami zapisano jako:\n{output_h5_path}")


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent

    input_file = base_dir / 'data' / 'processed' / 'signals_100hz_DEV.h5'
    output_file = base_dir / 'data' / 'processed' / 'cwt_scalograms_DEV.h5'

    if not input_file.exists():
        print(f"Błąd: Nie znaleziono pliku wejściowego {input_file}")
        sys.exit(1)

    process_and_save_cwt(input_file, output_file)