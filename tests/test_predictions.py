"""Predict classes for test images in the test_data directory."""

from pathlib import Path
import joblib
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, MODEL_OUTPUT_DIR, SUPPORTED_EXTENSIONS
from src.services.image_preprocessor import ImagePreprocessor


def predict_test_images():
    """Load trained model and predict classes for all images in test_data directory."""
    
    test_data_dir = DATA_DIR / "test_data"
    model_path = MODEL_OUTPUT_DIR / "macro_classifier.joblib"
    
    # Validate paths exist
    if not test_data_dir.exists():
        print(f"Error: test_data directory not found at {test_data_dir}")
        return
    
    if not model_path.exists():
        print(f"Error: trained model not found at {model_path}")
        return
    
    # Load trained model
    model = joblib.load(model_path)
    preprocessor = ImagePreprocessor()
    
    # Collect all image files in test_data
    image_files = [
        f for f in test_data_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    
    if not image_files:
        print(f"No images found in {test_data_dir}")
        return
    
    print(f"Found {len(image_files)} images in test_data directory.\n")
    print("-" * 100)
    print(f"{'Image File':<50} | {'Predicted Class':<20} | {'Confidence':<10}")
    print("-" * 100)
    
    predictions_results = []
    
    # Predict class for each image
    for image_path in sorted(image_files):
        try:
            features = preprocessor.transform(str(image_path))
            predicted_class = model.predict(features.reshape(1, -1))[0]
            probability = model.predict_proba(features.reshape(1, -1))[0]
            confidence = float(probability.max()) * 100
            relative_path = image_path.relative_to(test_data_dir)
            print(
                f"{str(relative_path):<50} | {predicted_class:<20} | {confidence:>7.2f}%"
            )
            predictions_results.append({
                "file": str(relative_path),
                "predicted_class": predicted_class,
                "confidence": round(confidence, 2),
            })
        except Exception as e:
            print(f"Error predicting {image_path}: {e}")
    
    print("-" * 80)
    print(f"\nTotal predictions: {len(predictions_results)}")
    
    return predictions_results


if __name__ == "__main__":
    predict_test_images()
