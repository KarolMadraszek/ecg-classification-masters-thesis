from pathlib import Path
import h5py
import numpy as np

def load_stratified_subset(
    hdf5_path, split: str = 'train', sample_fraction: float = 1.0, random_seed: int = 42
):
    safe_path = str(hdf5_path)
    safe_split = str(split)

    with h5py.File(safe_path, 'r') as h5_file:
        group = h5_file[safe_split]
        total_samples = group['X'].shape[0]

        if sample_fraction >= 1.0:
            return group['X'][:], group['y'][:], group['ecg_id'][:]

        rng = np.random.default_rng(random_seed)
        sample_size = int(total_samples * sample_fraction)

        selected_indices = np.sort(
            rng.choice(total_samples, size=sample_size, replace=False)
        ).tolist()

        sampled_x = group['X'][selected_indices, :, :]
        sampled_y = group['y'][selected_indices, :]
        sampled_ids = group['ecg_id'][selected_indices]

        return sampled_x, sampled_y, sampled_ids

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    hdf5_dev_path = base_dir / 'data' / 'processed' / 'signals_100hz_DEV.h5'

    x_sub, y_sub, ids_sub = load_stratified_subset(
        hdf5_path=hdf5_dev_path,
        split='train',
        sample_fraction=0.1
    )
    print(f'Pobrano podzbiór deweloperski X: {x_sub.shape}, y: {y_sub.shape}')