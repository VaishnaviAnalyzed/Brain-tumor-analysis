"""Model builders: custom CNN and transfer-learning backbones."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications as apps

import config

IMG_SHAPE = (*config.IMG_SIZE, 3)


def build_custom_cnn():
    """CNN from scratch (5 conv blocks, BatchNorm + Dropout). Expects inputs scaled to 0-1.

    Early blocks use a single conv (cheap at high resolution); deeper blocks use two.
    """
    inp = keras.Input(IMG_SHAPE)
    x = inp
    for filters, n_convs, drop in ((16, 1, 0), (32, 1, 0), (64, 2, 0), (128, 2, 0.1), (256, 2, 0.2)):
        for _ in range(n_convs):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            # momentum 0.9: moving statistics track quickly (stable train/eval behaviour)
            x = layers.BatchNormalization(momentum=0.9)(x)
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        if drop:  # dropout only in the deep blocks avoids BatchNorm variance shift early on
            x = layers.SpatialDropout2D(drop)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inp, out, name="custom_cnn")


@keras.utils.register_keras_serializable(package="brain_mri")
class CaffePreprocess(layers.Layer):
    """ResNet50 preprocessing (RGB->BGR, subtract ImageNet mean) as a serialisable layer,
    so the saved .h5 loads without Lambda/pickled functions."""

    def call(self, x):
        x = x[..., ::-1]
        return x - tf.constant([103.939, 116.779, 123.68], dtype=x.dtype)


def _preprocess_layer(name):
    if name == "resnet50":
        return CaffePreprocess(name="preprocess")
    if name in ("mobilenetv2", "inceptionv3"):
        return layers.Rescaling(1 / 127.5, offset=-1.0, name="preprocess")  # -> [-1, 1]
    return layers.Identity(name="preprocess")  # EfficientNet rescales internally


# name -> backbone constructor
BACKBONES = {
    "resnet50": apps.ResNet50,
    "mobilenetv2": apps.MobileNetV2,
    "inceptionv3": apps.InceptionV3,
    "efficientnetb0": apps.EfficientNetB0,
}


def build_transfer_model(name, weights="imagenet"):
    """Pretrained backbone + new dense head. Expects raw 0-255 pixels.

    The model-specific preprocessing is a layer inside the graph, so the saved .h5 accepts
    0-255 images and needs no model-specific handling at inference time.
    """
    ctor = BACKBONES[name]
    base = ctor(include_top=False, weights=weights, input_shape=IMG_SHAPE)
    base.trainable = False  # stage 1: frozen

    inp = keras.Input(IMG_SHAPE)
    x = _preprocess_layer(name)(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inp, out, name=name), base


def unfreeze_top(model, base, n=config.UNFREEZE_LAST_N):
    """Stage 2: unfreeze the last n layers (BatchNorm stays frozen for stability)."""
    base.trainable = True
    for layer in base.layers[:-n]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return model


def input_mode(name):
    """'01' for the custom CNN, 'raw' (0-255) for pretrained models."""
    return "01" if name == "custom_cnn" else "raw"
