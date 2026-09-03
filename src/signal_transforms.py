import pywt
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

def standardize_signal(ecg_lead):
    mean_val = np.mean(ecg_lead)
    std_val = np.std(ecg_lead)

    if std_val > 0:
        return (ecg_lead - mean_val) / std_val
    return ecg_lead

def generate_log_cwt_normalized(ecg_lead, fs=100.0):
    """
    Standaryzacja 1D -> CWT -> Kompresja Log -> Normalizacja Min-Max 2D.
    """
    ecg_lead_scaled = standardize_signal(ecg_lead)

    scales = np.arange(1, 40)
    dt = 1.0 / fs
    cwt_matrix, freqs = pywt.cwt(ecg_lead_scaled, scales, 'cmor1.5-1.0', sampling_period=dt)

    cwt_power = np.abs(cwt_matrix)
    cwt_log = np.log(cwt_power + 1e-7)

    min_val = np.min(cwt_log)
    max_val = np.max(cwt_log)

    if max_val > min_val:
        cwt_normalized = (cwt_log - min_val) / (max_val - min_val)
    else:
        cwt_normalized = cwt_log - min_val

    return cwt_normalized, freqs


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    hdf5_path = base_dir / 'data' / 'processed' / 'signals_100hz_DEV.h5'

    with h5py.File(hdf5_path, 'r') as f:
        available_keys = list(f.keys())
        print(f"Dostępne klucze główne HDF5: {available_keys}")

        if not available_keys:
            raise ValueError("Plik HDF5 jest pusty!")

        first_key = available_keys[0]
        item = f[first_key]

        if isinstance(item, h5py.Group):
            inner_keys = list(item.keys())
            print(f"Klucz '{first_key}' to grupa. Zawiera podklucze: {inner_keys[:5]}")
            first_inner_key = inner_keys[0]
            dataset = item[first_inner_key]
        else:
            print(f"Klucz '{first_key}' to bezpośredni Dataset.")
            dataset = item

        print(f"Ostatecznie pobrany zbiór danych ma kształt: {dataset.shape}")

        if len(dataset.shape) == 3:
            # Struktura: (N_pacjentów, 1000, 12)
            patient_signal = dataset[0]
        else:
            # Struktura: (1000, 12) - pacjenci zapisani pojedynczo
            patient_signal = dataset[:]

    lead_I = patient_signal[:, 0]
    cwt_image, freqs = generate_log_cwt_normalized(lead_I, fs=100.0)

    print(f"Kształt obrazu: {cwt_image.shape}, Min: {np.min(cwt_image):.2f}, Max: {np.max(cwt_image):.2f}")
    plt.figure(figsize=(10, 4))
    plt.imshow(cwt_image, aspect='auto', cmap='jet', origin='lower')
    plt.title(f'Znormalizowany skalogram CWT (z elementu: {first_key})')
    plt.ylabel('Skale')
    plt.xlabel('Próbki (100 Hz)')
    plt.show()