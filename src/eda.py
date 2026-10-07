"""Step 1 - understand the dataset: class counts, imbalance, resolution, sample grid."""
import json
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config

SPLITS = {"train": config.TRAIN_DIR, "valid": config.VALID_DIR, "test": config.TEST_DIR}


def main():
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    counts = {s: {c: len(list((d / c).glob("*"))) for c in config.CLASS_NAMES} for s, d in SPLITS.items()}
    sizes = Counter()
    for d in SPLITS.values():
        for f in d.glob("*/*"):
            sizes[Image.open(f).size] += 1
    train_counts = list(counts["train"].values())
    summary = {
        "counts": counts,
        "image_sizes": {f"{w}x{h}": n for (w, h), n in sizes.items()},
        "imbalance_ratio_train_max_min": round(max(train_counts) / min(train_counts), 2),
    }
    (config.METRICS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # class distribution per split
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x, w = np.arange(config.NUM_CLASSES), 0.27
    for i, s in enumerate(SPLITS):
        ax.bar(x + (i - 1) * w, list(counts[s].values()), w, label=s)
    ax.set_xticks(x, config.CLASS_NAMES)
    ax.set_ylabel("images")
    ax.set_title("Class distribution per split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "class_distribution.png", dpi=150)

    # sample grid
    random.seed(config.SEED)
    fig, axes = plt.subplots(config.NUM_CLASSES, 5, figsize=(12, 10))
    for r, c in enumerate(config.CLASS_NAMES):
        files = random.sample(list((config.TRAIN_DIR / c).glob("*")), 5)
        for k, f in enumerate(files):
            axes[r, k].imshow(Image.open(f).convert("RGB"))
            axes[r, k].axis("off")
            if k == 0:
                axes[r, k].set_title(c, loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "sample_images.png", dpi=120)


if __name__ == "__main__":
    main()
