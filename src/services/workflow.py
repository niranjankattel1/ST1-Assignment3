import joblib
import pandas as pd
from pandas import DataFrame

from src.config import EDA_OUTPUT_DIR, MODEL_OUTPUT_DIR, RAW_DATA_DIR
from src.services.classifier import Classifier
from src.services.dataset_indexer import DatasetIndexer
from src.services.eda_service import EDAService
from src.services.image_preprocessor import ImagePreprocessor
from .kaggle_hub import KaggleHub


class WorkflowService:
    """Coordinate the shared workflow used by batch, GUI, and console entry points."""

    def __init__(self) -> None:
        EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.indexer = DatasetIndexer()
        self.preprocessor = ImagePreprocessor()
        self.classifier = Classifier(self.preprocessor, MODEL_OUTPUT_DIR)
        self.dataframe: pd.DataFrame | None = None

        kaggle_hub = KaggleHub(RAW_DATA_DIR)
        kaggle_hub.get_raw_data()
        kaggle_hub.organize_data()


    def load_dataframe(self) -> DataFrame:
        """Load and cache the indexed dataset."""

        if self.dataframe is None:
            self.dataframe = self.indexer.build_dataframe()
            self.indexer.save_indexed_data(self.dataframe)
        return self.dataframe

    def show_summary(self) -> dict[str, float]:
        """Build and print dataset summary statistics."""

        dataframe = self.load_dataframe()
        eda = EDAService(dataframe, EDA_OUTPUT_DIR)
        summary = eda.build_summary()
        print(summary)
        return summary

    def get_available_classes(self) -> list[str]:
        """Return available class folder names for EDA selection."""

        class_root = RAW_DATA_DIR / "stream_macroinvertebrates"
        if class_root.exists() and class_root.is_dir():
            classes = sorted(
                [path.name for path in class_root.iterdir() if path.is_dir()]
            )
            if classes:
                return classes

        dataframe = self.load_dataframe()
        return sorted(dataframe["label"].dropna().unique())

    def generate_eda(self, class_name: str | None = None) -> None:
        """Create and save the main EDA outputs, optionally for one class."""

        if class_name:
            self.generate_eda_for_class(class_name)
            return

        dataframe = self.load_dataframe()
        eda = EDAService(dataframe, EDA_OUTPUT_DIR)
        eda.save_class_distribution()
        eda.save_image_size_distribution()
        eda.save_sample_grid()

    def generate_eda_for_class(self, class_name: str) -> None:
        """Create and save EDA outputs for a single class."""

        dataframe = self.load_dataframe()
        if class_name not in dataframe["label"].unique():
            raise ValueError(f"Class '{class_name}' was not found in the dataset.")

        class_df = dataframe[dataframe["label"] == class_name]
        if class_df.empty:
            raise ValueError(f"No images were found for class '{class_name}'.")

        output_dir = EDA_OUTPUT_DIR / class_name
        eda = EDAService(class_df, output_dir)
        eda.save_class_distribution()
        eda.save_image_size_distribution()
        eda.save_sample_grid(output_dir / "sample_grid.png")
        print(f"Saved EDA outputs for class '{class_name}' to {output_dir}")

    def train_model(self) -> dict[str, object]:
        """Train the baseline model and save it to disk."""

        dataframe = self.load_dataframe()
        results = self.classifier.train(dataframe, True, True)
        self.classifier.save_model()
        return results

    def predict_image(self, file_path: str) -> dict[str, object]:
        """Predict the class of one input image and return the prediction with confidence."""

        model_path = MODEL_OUTPUT_DIR / "macro_classifier.joblib"
        if model_path.exists():
            self.classifier.model = joblib.load(model_path)
        features = self.preprocessor.transform(file_path).reshape(1, -1)
        prediction = str(self.classifier.model.predict(features)[0])
        probability = self.classifier.model.predict_proba(features)[0]
        confidence = float(probability.max()) * 100

        return {
            "predicted_class": prediction,
            "confidence": round(confidence, 2),
        }

    def run_full_pipeline(self) -> None:
        """Run the default Stage 1 and Stage 2 workflow."""

        self.show_summary()
        self.generate_eda()
        results = self.train_model()
        print(f"Training accuracy: {results['accuracy']:.4f}")
        print(results["report"])