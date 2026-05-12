from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
class Classifier:
    """Train, evaluate, and persist the baseline classification model."""
    def __init__(self, preprocessor, model_output_dir: Path) -> None:
        self.preprocessor = preprocessor
        self.model_output_dir = model_output_dir
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )


    def prepare_features(self, dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Convert indexed file paths into model features and labels."""

        features = []
        labels = []
        for _, row in dataframe.iterrows():
            features.append(self.preprocessor.transform(row["file_path"]))
            labels.append(row["label"])
        return np.array(features), np.array(labels)

    def train(self, dataframe: pd.DataFrame) -> dict[str, object]:
        """Fit the model and return evaluation outputs."""

        X, y = self.prepare_features(dataframe)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        results = {
            "accuracy": accuracy_score(y_test, predictions),
            "report": classification_report(y_test, predictions, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, predictions),
        }
        return results

    def save_model(self, file_name="macro_classifier.joblib") -> Path:
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.model_output_dir / file_name
        joblib.dump(self.model, output_path)
        return output_path

    def load_model(self, file_name="macro_classifier.joblib") -> None:
        # Load a saved trained model

        model_path = self.model_output_dir / file_name

        self.model = joblib.load(model_path)

    def predict_single(self, image_path: str):

        image = self.preprocessor.transform(image_path)

        image = np.array([image])

        probabilities = self.model.predict_proba(image)

        confidence = np.max(probabilities)

        prediction = self.model.predict(image)

        return prediction[0], confidence