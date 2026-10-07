"""Train the custom CNN and/or transfer-learning models.

Usage:
    python -m src.train --model custom_cnn
    python -m src.train --model mobilenetv2 resnet50 inceptionv3 efficientnetb0
    python -m src.train --model all
"""
import argparse
import json
import time

from tensorflow import keras

import config
from src import data_loader, models


def _callbacks(path, patience=6):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(path, monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7),
    ]


def _compile(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def _merge(h1, h2=None):
    out = {k: list(v) for k, v in h1.history.items()}
    if h2:
        for k, v in h2.history.items():
            out[k] += list(v)
    return out


def train_one(name, epochs=None):
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = str(config.MODELS_DIR / f"{name}.h5")
    train, valid, _ = data_loader.get_datasets(scale_01=(name == "custom_cnn"))
    cw = data_loader.class_weights()
    t0 = time.time()

    if name == "custom_cnn":
        model = models.build_custom_cnn()
        _compile(model, config.LR_CNN)
        hist = _merge(model.fit(train, validation_data=valid, epochs=epochs or config.EPOCHS_CNN,
                                class_weight=cw, callbacks=_callbacks(path, 10)))
    else:
        model, base = models.build_transfer_model(name)
        _compile(model, config.LR_HEAD)
        h1 = model.fit(train, validation_data=valid, epochs=epochs or config.EPOCHS_HEAD,
                       class_weight=cw, callbacks=_callbacks(path))
        # stage 2: fine-tune top layers with a small learning rate
        model = keras.models.load_model(path, custom_objects={"CaffePreprocess": models.CaffePreprocess})
        base = next(l for l in model.layers if isinstance(l, keras.Model))
        models.unfreeze_top(model, base)
        _compile(model, config.LR_FINE)
        h2 = model.fit(train, validation_data=valid, epochs=config.EPOCHS_FINE,
                       class_weight=cw, callbacks=_callbacks(path))
        model.save(path)  # EarlyStopping restored the best fine-tuning weights
        hist = _merge(h1, h2)

    hist["train_minutes"] = round((time.time() - t0) / 60, 2)
    (config.METRICS_DIR / f"{name}_history.json").write_text(json.dumps(hist))
    print(f"[{name}] saved -> {path}  ({hist['train_minutes']} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["custom_cnn"])
    ap.add_argument("--epochs", type=int, default=None, help="override epochs of the first stage")
    args = ap.parse_args()
    names = ["custom_cnn", *models.BACKBONES] if "all" in args.model else args.model
    for n in names:
        train_one(n, args.epochs)


if __name__ == "__main__":
    main()
