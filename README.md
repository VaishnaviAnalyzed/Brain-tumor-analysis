# 🧠 Brain Tumor MRI Image Classification

Deep-learning pipeline that classifies brain MRI slices into **glioma, meningioma, pituitary tumor, or no tumor**, with a custom CNN, four transfer-learning backbones, and a Streamlit app for real-time prediction.

> ⚠️ Educational/research prototype. Not a medical device.

## Project status (read this first)

| Item | Status |
|---|---|
| Data loading, preprocessing, augmentation, EDA | ✅ done and run |
| Custom CNN (from scratch) | ✅ trained **(CPU-only, short run; see results)** |
| Transfer learning (ResNet50, MobileNetV2, InceptionV3, EfficientNetB0) | ⚠️ **code written, not yet trained or tested.** ImageNet weights could not be downloaded in the build environment. Run `bash run_pipeline.sh` with internet access |
| Evaluation + comparison scripts | ✅ run for the custom CNN only |
| Streamlit app | ✅ loads and predicts with the custom CNN; other models appear automatically once their `.h5` files exist |

## Dataset

Brain Tumor MRI multi-class dataset (provided as `train/`, `valid/`, `test/` folders under `data/`).

| Split | glioma | meningioma | no_tumor | pituitary | total |
|---|---|---|---|---|---|
| train | 564 | 358 | 335 | 438 | 1695 |
| valid | 161 | 124 | 99 | 118 | 502 |
| test | 80 | 63 | 49 | 54 | 246 |

All images are 640×640 RGB. Class imbalance is mild (max/min = 1.68 in train); it is handled with balanced class weights.

**Data-leakage caveat.** File names are Roboflow-augmented copies (`Tr-gl_0538_jpg.rf.<hash>`). 97 validation and 57 test images share a source ID with a training image, and the ones checked were near-duplicates. The splits were kept as provided, so **test metrics are likely optimistic**. Re-splitting by source ID is recommended for any serious evaluation.

## Structure

```
config.py              paths, classes, hyper-parameters
src/data_loader.py     load, resize to 224x224, normalise, augment
src/models.py          custom CNN + transfer-learning builders
src/train.py           training (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
src/evaluate.py        accuracy/precision/recall/F1, confusion matrix, history plots
src/compare.py         model comparison table + best-model selection
src/predict.py         inference backend used by the app
src/eda.py             dataset exploration plots
app/streamlit_app.py   web front-end
models/                saved .h5 models
outputs/               plots and metrics
run_pipeline.sh        EDA -> train -> evaluate -> compare
```

## Setup and usage

```bash
pip install -r requirements.txt
# place the dataset so that data/train, data/valid, data/test exist

python -m src.eda
python -m src.train --model custom_cnn
python -m src.train --model mobilenetv2 resnet50 inceptionv3 efficientnetb0   # needs internet
python -m src.evaluate
python -m src.compare
streamlit run app/streamlit_app.py
```
(`bash run_pipeline.sh` runs everything. A GPU or Google Colab is strongly advised for the pretrained models.)

## Method

- **Preprocessing:** resize to 224×224; the custom CNN uses 0–1 pixels, and pretrained models apply their own preprocessing as a layer inside the saved model, so every `.h5` accepts raw 0–255 input.
- **Augmentation (train only):** horizontal/vertical flips, rotation, zoom, shifts, brightness.
- **Custom CNN:** 5 conv blocks (16→256 filters) with BatchNorm, spatial dropout in the deep blocks, global average pooling, Dense(256)+BatchNorm+Dropout(0.5), softmax. 1.24M parameters.
- **Transfer learning:** frozen ImageNet backbone + new dense head, then fine-tuning of the top 30 layers (BatchNorm frozen) at a low learning rate.
- **Training:** Adam, categorical cross-entropy, class weights, EarlyStopping / ModelCheckpoint on validation loss, ReduceLROnPlateau.

## Results

Test set (246 images), custom CNN only so far:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Params | Size |
|---|---|---|---|---|---|---|
| custom_cnn | 0.703 | 0.737 | 0.734 | 0.699 | 1.24M | 15 MB |

Per-class: glioma recall 0.55 (precision 1.00), meningioma recall 0.52, no_tumor recall 0.92, pituitary recall 0.94. The model misses many gliomas and meningiomas, which is a serious weakness for a medical task.

**Honest notes on this run:**
- Trained on a single CPU for a short time; validation accuracy was unstable (30–70%) and training was interrupted at epoch 15 of 25. The saved model is the best checkpoint (epoch 11, validation loss 0.78). The training history was reconstructed from the log.
- Treat 70% as a low baseline, not a final result. Longer training, a lower or scheduled learning rate, and especially the pretrained backbones are expected to improve it, but **that has not been verified**.
- The comparison between the custom CNN and pretrained models (deliverable 4) is therefore **incomplete** until the transfer-learning models are trained. `python -m src.compare` produces the table and picks the best model automatically; the app reads it.

## Deliverables checklist (from the project brief)

1. Trained models (.h5): custom CNN ✅, pretrained models ⏳ (run training)
2. Streamlit app ✅
3. Training/evaluation/deployment scripts ✅
4. Model comparison: script ✅, full results ⏳
5. Public GitHub repo with README: README ✅ (push the repo yourself; `data/` is git-ignored)
6. Clean, modular, commented code ✅
