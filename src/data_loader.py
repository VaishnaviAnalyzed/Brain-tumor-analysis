"""Data pipeline: loading, 0-1 normalisation, 224x224 resize and augmentation."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config


def _load(directory, shuffle, batch_size=config.BATCH_SIZE):
    return keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASS_NAMES,
        image_size=config.IMG_SIZE,   # resize to a consistent shape
        batch_size=batch_size,
        shuffle=shuffle,
        seed=config.SEED,
    )


def build_augmenter():
    """Rotation, horizontal/vertical flips, zoom, brightness and shifts (train only)."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomBrightness(0.2, value_range=(0.0, 1.0)),
        ],
        name="augmentation",
    )


def get_datasets(scale_01=True):
    """Return (train, valid, test) tf.data datasets.

    scale_01=True  -> pixels normalised to 0-1 (custom CNN).
    scale_01=False -> pixels kept in 0-255; each pretrained model applies its own
                      preprocess_input inside the model graph (see models.py).
    """
    train, valid, test = (
        _load(config.TRAIN_DIR, True),
        _load(config.VALID_DIR, False),
        _load(config.TEST_DIR, False),
    )
    aug = build_augmenter()
    div = 255.0 if scale_01 else 1.0   # custom CNN: 0-1; pretrained: keep 0-255

    def prep_train(x, y):
        x = tf.cast(x, tf.float32) / 255.0          # augment in 0-1 space
        x = aug(x, training=True)
        return x * (255.0 / div) if div == 1.0 else x, y

    def prep_eval(x, y):
        return tf.cast(x, tf.float32) / div, y

    train = train.map(prep_train, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    valid = valid.map(prep_eval, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test = test.map(prep_eval, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return train, valid, test


def class_weights():
    """Balanced class weights computed from the training folder counts."""
    counts = [len(list((config.TRAIN_DIR / c).glob("*"))) for c in config.CLASS_NAMES]
    total = sum(counts)
    return {i: total / (len(counts) * n) for i, n in enumerate(counts)}
