"""Evaluate saved models on the test set: accuracy, precision/recall/F1, confusion matrix,
training-history plots. Writes outputs/metrics/<model>_metrics.json and plots.

Usage: python -m src.evaluate            (all models found in models/)
       python -m src.evaluate --model custom_cnn
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import config
from src import data_loader, predict


def plot_history(name):
    f = config.METRICS_DIR / f"{name}_history.json"
    if not f.exists():
        return
    h = json.loads(f.read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, m in zip(ax, ("accuracy", "loss")):
        a.plot(h[m], label="train")
        a.plot(h[f"val_{m}"], label="validation")
        a.set_title(f"{name} - {m}")
        a.set_xlabel("epoch")
        a.legend()
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / f"{name}_history.png", dpi=140)
    plt.close(fig)


def evaluate(name):
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _, _, test = data_loader.get_datasets(scale_01=(name == "custom_cnn"))
    model = predict.load_model(name)
    y_true = np.concatenate([np.argmax(y, 1) for _, y in test])
    t0 = time.time()
    probs = model.predict(test, verbose=0)
    ms_per_img = (time.time() - t0) / len(y_true) * 1000
    y_pred = probs.argmax(1)

    rep = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    out = {
        "model": name,
        "accuracy": rep["accuracy"],
        "precision_macro": rep["macro avg"]["precision"],
        "recall_macro": rep["macro avg"]["recall"],
        "f1_macro": rep["macro avg"]["f1-score"],
        "per_class": {c: rep[c] for c in config.CLASS_NAMES},
        "confusion_matrix": cm.tolist(),
        "params": int(model.count_params()),
        "size_mb": round((config.MODELS_DIR / f"{name}.h5").stat().st_size / 1e6, 1),
        "ms_per_image": round(ms_per_img, 2),
    }
    (config.METRICS_DIR / f"{name}_metrics.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{name} - confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / f"{name}_confusion.png", dpi=140)
    plt.close(fig)
    plot_history(name)
    print(f"[{name}] test acc={out['accuracy']:.4f}  macro-F1={out['f1_macro']:.4f}")
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=None)
    args = ap.parse_args()
    for n in args.model or predict.available_models():
        evaluate(n)


if __name__ == "__main__":
    main()
