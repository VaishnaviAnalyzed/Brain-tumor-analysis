"""Backend inference service used by the Streamlit app (and usable from scripts)."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

import config
from src.models import CaffePreprocess, input_mode



class CompatDense(keras.layers.Dense):
    """Dense that ignores `quantization_config`, written by newer Keras versions (e.g. Colab).
    Lets models saved with a newer Keras load on an older local Keras."""

    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)


CUSTOM_OBJECTS = {"CaffePreprocess": CaffePreprocess, "Dense": CompatDense}


def available_models():
    """Names of every trained model found in models/."""
    return sorted(p.stem for p in config.MODELS_DIR.glob("*.h5"))


@lru_cache(maxsize=4)
def load_model(name):
    return keras.models.load_model(config.MODELS_DIR / f"{name}.h5", custom_objects=CUSTOM_OBJECTS, compile=False)


def preprocess(image, name):
    """PIL image -> (1, 224, 224, 3) float32 batch in the range the model expects."""
    img = image.convert("RGB").resize(config.IMG_SIZE)
    arr = np.asarray(img, dtype="float32")
    if input_mode(name) == "01":
        arr /= 255.0
    return arr[None]


def predict(image, name):
    """Return dict(label, confidence, probabilities{class: p})."""
    probs = load_model(name).predict(preprocess(image, name), verbose=0)[0]
    top = int(np.argmax(probs))
    return {
        "label": config.CLASS_NAMES[top],
        "confidence": float(probs[top]),
        "probabilities": {c: float(p) for c, p in zip(config.CLASS_NAMES, probs)},
    }


if __name__ == "__main__":
    import sys
    print(predict(Image.open(sys.argv[2]), sys.argv[1]))