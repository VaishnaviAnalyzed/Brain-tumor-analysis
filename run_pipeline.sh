#!/usr/bin/env bash
# End-to-end pipeline: EDA -> train all models -> evaluate -> compare.
# Usage: bash run_pipeline.sh            (all 5 models; needs internet for ImageNet weights)
#        bash run_pipeline.sh custom_cnn (subset)
set -euo pipefail
MODELS=${@:-all}
python -m src.eda
python -m src.train --model $MODELS
python -m src.evaluate
python -m src.compare
echo "Done. Launch the app with: streamlit run app/streamlit_app.py"
