"""Streamlit front-end for Brain Tumor MRI classification.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from PIL import Image

import config
from src import predict

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
      .result-card {padding: 1.2rem 1.4rem; border-radius: 14px; border: 1px solid rgba(128,128,128,.3);}
      .result-label {font-size: 2rem; font-weight: 700; margin: 0;}
      .result-sub {opacity: .7; margin: 0 0 .4rem 0; font-size: .85rem; text-transform: uppercase; letter-spacing: .06em;}
    </style>
    """,
    unsafe_allow_html=True,
)

DISPLAY = {"glioma": "Glioma", "meningioma": "Meningioma", "no_tumor": "No tumor", "pituitary": "Pituitary tumor"}
INFO = {
    "glioma": "Tumor arising from glial cells in the brain or spinal cord. Often infiltrative; typically needs prompt specialist evaluation.",
    "meningioma": "Tumor of the meninges (membranes around the brain). Frequently slow-growing and often benign.",
    "no_tumor": "No tumor pattern detected by the model in this scan.",
    "pituitary": "Tumor of the pituitary gland at the base of the brain; can affect hormone levels and vision.",
}
LOW_CONFIDENCE = 0.70

# ---------------------------------------------------------------- sidebar
models_found = predict.available_models()
with st.sidebar:
    st.title("🧠 MRI Classifier")
    st.caption("Brain tumor type classification: glioma · meningioma · pituitary · no tumor")
    if models_found:
        # open on the best model from the comparison table (falls back to the first model)
        csv_path = config.METRICS_DIR / "model_comparison.csv"
        best = pd.read_csv(csv_path).iloc[0]["model"] if csv_path.exists() else None
        default = models_found.index(best) if best in models_found else 0
        model_name = st.selectbox("Model", models_found, index=default)
    else:
        model_name = None
    st.divider()
   
    

tab_predict, tab_perf, tab_data = st.tabs(["🔍 Predict", "📊 Model performance", "🗂 Dataset"])

# ---------------------------------------------------------------- predict tab
with tab_predict:
    if not models_found:
        st.error("No trained models found in `models/`. Run `python -m src.train --model custom_cnn` first.")
    else:
        files = st.file_uploader(
            "Upload one or more brain MRI images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
        )
        if not files:
            st.info("Upload an MRI slice (JPG/PNG) to get a prediction.")
        for f in files:
            img = Image.open(f)
            with st.spinner(f"Analysing {f.name} ..."):
                res = predict.predict(img, model_name)
            left, right = st.columns([1, 1.3], gap="large")
            with left:
                st.image(img, caption=f.name, width="stretch")
            with right:
                label = res["label"]
                st.markdown(
                    f"<div class='result-card'><p class='result-sub'>Predicted class · {model_name}</p>"
                    f"<p class='result-label'>{DISPLAY[label]}</p></div>",
                    unsafe_allow_html=True,
                )
                st.metric("Confidence", f"{res['confidence'] * 100:.1f}%")
                st.progress(res["confidence"])
                if res["confidence"] < LOW_CONFIDENCE:
                    st.warning("Low confidence - treat this result with extra caution and seek expert review.")
                st.caption(INFO[label])
                probs = pd.Series({DISPLAY[k]: v for k, v in res["probabilities"].items()}, name="probability")
                st.bar_chart(probs, horizontal=True)
            st.divider()

# ---------------------------------------------------------------- performance tab
with tab_perf:
    csv = config.METRICS_DIR / "model_comparison.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        st.subheader("Model comparison (test set)")
        st.dataframe(df.style.format({c: "{:.4f}" for c in df.columns if c.endswith(("accuracy", "macro"))}),
                     width="stretch", hide_index=True)
        st.success(f"Best model by macro-F1: **{df.iloc[0]['model']}**")
        plot = config.PLOTS_DIR / "model_comparison.png"
        if plot.exists():
            st.image(str(plot))
        pick = st.selectbox("Inspect a model", df["model"], key="perf_pick")
        c1, c2 = st.columns(2)
        for col, suffix in ((c1, "confusion"), (c2, "history")):
            p = config.PLOTS_DIR / f"{pick}_{suffix}.png"
            if p.exists():
                col.image(str(p), width="stretch")
    else:
        st.info("Run `python -m src.evaluate` and `python -m src.compare` to populate this tab.")

# ---------------------------------------------------------------- dataset tab
with tab_data:
    for name in ("class_distribution", "sample_images"):
        p = config.PLOTS_DIR / f"{name}.png"
        if p.exists():
            st.image(str(p), width="stretch")
    if not (config.PLOTS_DIR / "class_distribution.png").exists():
        st.info("Run `python -m src.eda` to generate dataset plots.")