"""Central configuration shared by training, evaluation and the Streamlit app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_DIR, VALID_DIR, TEST_DIR = DATA_DIR / "train", DATA_DIR / "valid", DATA_DIR / "test"
MODELS_DIR = ROOT / "models"
PLOTS_DIR = ROOT / "outputs" / "plots"
METRICS_DIR = ROOT / "outputs" / "metrics"

# Alphabetical order == Keras directory order, so labels stay consistent everywhere.
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Training schedule
EPOCHS_CNN = 40
EPOCHS_HEAD = 15      # transfer learning: frozen-base stage
EPOCHS_FINE = 15      # transfer learning: fine-tuning stage
LR_CNN = 5e-4
LR_HEAD = 1e-3
LR_FINE = 1e-5
UNFREEZE_LAST_N = 30  # layers unfrozen during fine-tuning
