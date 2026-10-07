"""Compare all evaluated models and pick the best one for deployment."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config


def main():
    rows = [json.loads(f.read_text()) for f in sorted(config.METRICS_DIR.glob("*_metrics.json"))]
    if not rows:
        raise SystemExit("No metrics found - run `python -m src.evaluate` first.")
    df = pd.DataFrame(rows)[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro",
                             "params", "size_mb", "ms_per_image"]]
    df = df.sort_values(["f1_macro", "accuracy"], ascending=False).reset_index(drop=True)
    df.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    df.set_index("model")[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].plot.bar(ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison on the test set")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "model_comparison.png", dpi=150)
    print(f"\nBest model for deployment (macro-F1, then accuracy): {df.loc[0, 'model']}")


if __name__ == "__main__":
    main()
