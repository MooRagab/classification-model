from typing import Tuple

import tensorflow as tf


def build_model(
    num_classes: int,
    image_size: Tuple[int, int] = (224, 224),
    dropout_rate: float = 0.3,
    trainable_base: bool = False,
) -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = trainable_base

    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer")
    return model


def unfreeze_top_layers(model: tf.keras.Model, fine_tune_at: int = 100) -> tf.keras.Model:
    base_model = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)), None)
    if base_model is None:
        raise ValueError("Could not find base model for fine-tuning.")

    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    return model

