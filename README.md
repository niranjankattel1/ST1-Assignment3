# Stream Macroinvertebrate Image Classification

## Overview
This project analyzes and classifies stream macroinvertebrate images using a machine learning pipeline. It downloads the raw dataset, indexes image files, generates exploratory data analysis (EDA) outputs, trains a baseline classifier, and can predict the class of a single image.

## Key Features
- Downloads the `kennethtm/stream-macroinvertebrates` dataset automatically
- Builds an indexed dataset from raw image folders
- Generates EDA outputs:
  - class distribution plot
  - image size distribution plots
  - sample image grid
- Trains a baseline `RandomForestClassifier`
- Saves the trained model and evaluation reports
- Supports console-driven interaction via `src/console_app.py`

## Requirements
Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Required packages include:
- `opencv-python`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`
- `kagglehub`

## Project Structure

- `src/`
  - `main.py` - default non-interactive pipeline runner
  - `console_app.py` - interactive menu-driven application
  - `config.py` - path and image configuration constants
  - `services/`
    - `workflow.py` - orchestrates dataset loading, EDA, training, and prediction
    - `dataset_indexer.py` - scans raw folders and builds the image index
    - `eda_service.py` - creates and saves EDA visualizations
    - `image_preprocessor.py` - resizes, normalizes, and flattens images
    - `classifier.py` - trains, evaluates, and saves the model
    - `kaggle_hub.py` - downloads dataset via KaggleHub
  - `models/` - model-related classes and helpers
- `data/raw/` - raw dataset images organized by class
- `data/processed/` - generated indexed CSV data
- `output/`
  - `EDA/` - saved EDA charts
  - `models/` - trained model artifact(s)
  - `classification_report.txt` - classification metrics report

## Usage

### Run the full pipeline
This will download the dataset, index images, generate EDA outputs, train the model, and save results.

```bash
python -m src.main
```

### Run the interactive console app

```bash
python -m src.console_app
```

From the console menu you can:
1. Show dataset summary
2. Generate EDA outputs
3. Train the baseline classifier
4. Predict a single image
5. Exit

### Predict a single image
After training, use the console app option `4` or call `WorkflowService.predict_image()` with an image file path.

## Output

- `output/EDA/`
  - `class_distribution.png`
  - `image_size_distribution.png`
  - `sample_grid.png`
- `output/models/macro_classifier.joblib`
- `output/classification_report.txt`
- `output/confusion_matrix.png` (if enabled by training)

## Notes
- Image preprocessing uses grayscale resizing to `128x128` pixels.
- The dataset is expected to be structured by class name in `data/raw/`.
- The model training uses a stratified train/test split for consistent evaluation.

## Troubleshooting
- If the dataset does not download, confirm `kagglehub` is installed and network access is available.
- If images fail to read, verify supported extensions are present: `.jpg`, `.jpeg`, `.png`, `.bmp`.
- If running from a different working directory, use the full path to `src/main.py` and `src/console_app.py`.
