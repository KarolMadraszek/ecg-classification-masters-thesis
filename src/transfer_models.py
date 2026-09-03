import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Blokuje komunikaty INFO i WARNING z C++
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2, EfficientNetB0, MobileNetV2, DenseNet121
import numpy as np

def build_transfer_model(model_name='resnet50v2', input_shape=(39, 1000, 12), num_classes=5):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(3, kernel_size=(1, 1), padding='same', activation='relu', name='rgb_adapter')(inputs)
    base_input_shape = (input_shape[0], input_shape[1], 3)

    if model_name.lower() == 'resnet50v2':
        base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=base_input_shape)
    elif model_name.lower() == 'efficientnetb0':
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=base_input_shape)
    elif model_name.lower() == 'mobilenetv2':
        base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=base_input_shape)
    elif model_name.lower() == 'densenet121':
        base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=base_input_shape)
    else:
        raise ValueError("Nieobsługiwany model. Wybór: resnet50v2, efficientnetb0, mobilenetv2, densenet121")

    base_model.trainable = False
    x = base_model(x)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid', name='classifier')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"Transfer_{model_name}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(multi_label=True, name='auc')
        ]
    )

    return model


if __name__ == '__main__':
    print("Testowanie architektur uczenia transferowego")
    dummy_input = np.random.randn(1, 39, 1000, 12).astype(np.float32)

    for arch in ['resnet50v2', 'efficientnetb0', 'mobilenetv2', 'densenet121']:
        print(f"\nBudowanie modelu: {arch.upper()}...")
        model = build_transfer_model(model_name=arch)
        preds = model.predict(dummy_input, verbose=0)

        print(f"Liczba parametrów: {model.count_params():,} ")
        print(f"{arch.upper()} działa poprawnie.")