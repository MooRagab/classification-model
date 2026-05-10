# TensorFlow MobileNetV2 Image Classification Project

Complete image classification pipeline using TensorFlow + MobileNetV2 transfer learning, Kaggle dataset integration, and a Streamlit web app.

## Features
- Pretrained `MobileNetV2` on ImageNet
- Transfer learning with frozen base layers (phase 1)
- Fine-tuning top layers (phase 2)
- Data preprocessing + augmentation
- Training/validation plots for accuracy and loss
- Save/load trained models
- Predict uploaded images with top-3 classes + confidence
- Streamlit app for interactive inference

## Requirements Coverage (What is implemented + where)
This section maps each requested requirement to the exact implementation location.

1. Use a pretrained MobileNetV2 model trained on ImageNet
- Implemented in: `src/model.py` (`MobileNetV2(..., weights="imagenet", include_top=False)`)
- Explanation: We reuse ImageNet-pretrained convolutional features instead of training from scratch.

2. Freeze base layers
- Implemented in: `src/model.py` (`base_model.trainable = trainable_base`, default `False`)
- Used in training flow: `src/train.py` phase 1
- Explanation: Base CNN layers are frozen first to train only the custom head.

3. Add custom dense layers
- Implemented in: `src/model.py`
- Layers added: `GlobalAveragePooling2D -> Dense(256, relu) -> Dropout -> Dense(num_classes, softmax)`
- Explanation: These layers adapt pretrained features to your dataset classes.

4. Train on a custom dataset with multiple classes
- Implemented in: `src/data.py` + `src/train.py`
- Dataset rule: class-per-folder structure (`train/<class_name>/images...`)
- Explanation: Class names are discovered automatically from folder names, enabling multi-class training.

5. Apply image preprocessing and data augmentation
- Implemented in: `src/data.py`
- Preprocessing: `mobilenet_v2.preprocess_input`
- Augmentation: `RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`
- Explanation: Preprocessing matches MobileNetV2 input expectations; augmentation improves generalization.

6. Show training and validation accuracy/loss graphs
- Implemented in: `src/evaluate.py` (`plot_history`)
- Output file: `artifacts/plots/training_curves.png`
- Explanation: Reads training history and saves side-by-side accuracy/loss charts for train vs val.

7. Save and load the trained model
- Save implemented in: `src/train.py`
- Outputs: `artifacts/model/best_model.keras`, `artifacts/model/last_model.keras`, `artifacts/model/class_names.json`
- Load implemented in: `src/predict.py` (`load_model`) and `app.py`
- Explanation: Best model is checkpointed by validation accuracy; app/CLI load it for inference.

8. Create prediction function for uploaded images
- Implemented in: `src/predict.py` (`preprocess_image`, `predict_top_k`)
- Explanation: Uploaded image is resized/preprocessed, passed to model, then top-k classes are returned.

9. Build a Streamlit web app for image upload and prediction
- Implemented in: `app.py`
- Features: upload image, optional sample-image button, prediction display, confidence visualization.
- Explanation: Interactive UI wraps model inference for end users.

10. Return top 3 predictions with confidence scores
- Implemented in: `app.py` and `src/predict.py` (`top_k=3`)
- Explanation: Predictions are sorted by probability and top 3 labels are shown with percentages.

## Project Structure
```text
.
├── app.py
├── Procfile
├── requirements.txt
├── .streamlit/config.toml
├── scripts/
│   └── download_dataset.py
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
└── artifacts/
    ├── model/
    └── plots/
```

## 1) Install Dependencies
```bash
pip install -r requirements.txt
```

## 2) Configure Kaggle API
1. Create a Kaggle API token from Kaggle account settings.
2. Place `kaggle.json` in:
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`

## 3) Download and Prepare Dataset
Default dataset: `puneet6060/intel-image-classification`
```bash
python scripts/download_dataset.py --force
```

Prepared structure:
- `data/raw/train/<class_name>/...`
- `data/raw/test/<class_name>/...` (if available)

## 4) Train Model
```bash
python -m src.train --data_dir data/raw --epochs 20 --batch_size 32
```

Main outputs:
- `artifacts/model/best_model.keras`
- `artifacts/model/last_model.keras`
- `artifacts/model/class_names.json`
- `artifacts/model/history.json`

## 5) Evaluate + Plot Curves
```bash
python -m src.evaluate --data_dir data/raw --model_path artifacts/model/best_model.keras
```

Plot output:
- `artifacts/plots/training_curves.png`

## 6) CLI Prediction (Top-3)
```bash
python -m src.predict --model_path artifacts/model/best_model.keras --class_names_path artifacts/model/class_names.json --image path/to/image.jpg --top_k 3
```

## 7) Run Streamlit App
```bash
streamlit run app.py
```

Then upload an image and view top 3 predictions with confidence scores.

## Streamlit UI Notes
- Theme toggle (Light/Dark) is available from sidebar.
- `Try Sample Image` loads one image automatically from:
  - `data/raw/test/*/*` (first available), else
  - `data/raw/train/*/*`.
- Top-3 results are shown as:
  - ranked cards (label + confidence),
  - progress bars,
  - bar chart (`Confidence Chart`).

## Useful Training Args
- `--initial_lr` (phase 1 learning rate)
- `--fine_tune_lr` (phase 2 learning rate)
- `--fine_tune_at` (which base layers remain frozen)
- `--phase1_ratio` (fraction of epochs for frozen training)

## Notes
- Input size is fixed at `224x224` with `mobilenet_v2.preprocess_input`.
- Works with any class-per-folder dataset, not only Intel.
- If you ever see only one class prediction (e.g., `seg_train`), dataset structure is likely nested incorrectly.
  Ensure:
  - `data/raw/train/buildings/...`
  - `data/raw/train/forest/...`
  - `data/raw/train/glacier/...`
  - `data/raw/train/mountain/...`
  - `data/raw/train/sea/...`
  - `data/raw/train/street/...`
