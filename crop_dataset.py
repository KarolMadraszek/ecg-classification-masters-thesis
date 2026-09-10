import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

def main():
    base_dir = Path(__file__).resolve().parent
    input_h5 = base_dir / 'data' / 'processed' / 'cwt_scalograms_FULL.h5'
    output_h5 = base_dir / 'data' / 'processed' / 'cwt_scalograms_CROPPED.h5'

    if not input_h5.exists():
        print(f"Brak pliku wejściowego: {input_h5}")
        return

    crop_left = 50
    crop_right = 50
    batch_size = 256

    print("Rozpoczęcie cięcia skalogramów")
    print(f"Plik źródłowy: {input_h5.name}")
    print(f"Plik docelowy: {output_h5.name}\n")

    with h5py.File(input_h5, 'r') as f_in, h5py.File(output_h5, 'w') as f_out:
        for split in ['train', 'val', 'test']:
            if split not in f_in:
                continue

            n_samples = len(f_in[split]['y'])
            orig_shape = f_in[split]['X'].shape

            new_time_steps = orig_shape[2] - crop_left - crop_right
            new_shape = (orig_shape[0], orig_shape[1], new_time_steps, orig_shape[3])

            g_out = f_out.create_group(split)
            X_out = g_out.create_dataset('X', shape=new_shape, dtype=np.float32)
            y_out = g_out.create_dataset('y', shape=f_in[split]['y'].shape, dtype=np.float32)

            for start_idx in tqdm(range(0, n_samples, batch_size), desc=f"Przetwarzanie {split.upper():<5}"):
                end_idx = min(start_idx + batch_size, n_samples)

                X_batch = f_in[split]['X'][start_idx:end_idx]
                X_batch_cropped = X_batch[:, :, crop_left:-crop_right, :]

                X_out[start_idx:end_idx] = X_batch_cropped
                y_out[start_idx:end_idx] = f_in[split]['y'][start_idx:end_idx]

    print("\n" + "=" * 50)
    print(f"Wygenerowano przycięty zbiór danych. Nowa długość osi czasu: {new_time_steps}")
    print("=" * 50)

if __name__ == '__main__':
    main()