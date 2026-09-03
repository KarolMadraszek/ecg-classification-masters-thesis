import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Blokuje komunikaty INFO i WARNING z C++
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from src.signal_transforms import generate_log_cwt_normalized

def build_baseline_cnn(input_shape=(40, 1000, 12), num_classes=5):
    inputs = layers.Input(shape=input_shape)

    # Blok konwolucyjny
    x = layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)

    # 2. Blok konwolucyjny
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)

    # 3. Blok konwolucyjny
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)

    # Kompresja przestrzenna (redukuje liczbę parametrów i zapobiega przeuczeniu)
    x = layers.GlobalAveragePooling2D()(x)

    # Warstwy w pełni połączone (klasyfikator)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Regularyzacja - wyłączamy 50% neuronów

    # Warstwa wyjściowa
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Baseline_CWT_CNN")

    # Kompilacja modelu
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(multi_label=True, name='auc')
        ]
    )

    return model


def convert_patient_to_tensor(patient_signal, fs=100.0):
    channels = patient_signal.shape[1]  # Powinno być 12
    cwt_channels = []

    for ch in range(channels):
        lead = patient_signal[:, ch]
        cwt_img, _ = generate_log_cwt_normalized(lead, fs=fs)
        cwt_channels.append(cwt_img)

    tensor_3d = np.stack(cwt_channels, axis=-1)
    return tensor_3d


if __name__ == '__main__':
    model = build_baseline_cnn(input_shape=(39, 1000, 12))
    # Dla scales = np.arange(1, 40) jest 39 skal

    model.summary()

    print("\nTest przepływu danych")
    # "Fałszywey" pacjenta (losowy szum) symulującego surowe wejście (1000, 12)
    dummy_patient = np.random.randn(1000, 12)

    cwt_tensor = convert_patient_to_tensor(dummy_patient)
    print(f"Kształt po CWT: {cwt_tensor.shape} (oczekiwane: 39, 1000, 12)")

    # 4. Wymiar 'batch' (wymagany przez sieć neuronową) -> (1, 39, 1000, 12)
    batch_tensor = np.expand_dims(cwt_tensor, axis=0)

    predictions = model.predict(batch_tensor, verbose=0)
    print(f"Wymiary predykcji: {predictions.shape} (oczekiwane: 1, 5)")
    print(f"Prawdopodobieństwa dla 5 klas: {np.round(predictions[0], 3)}")