from pathlib import Path
import cv2
import pandas as pd
from src.config import RAW_DATA_DIR, SUPPORTED_EXTENSIONS, PROCESSED_DATA_DIR
from src.models.image_record import ImageRecord


class DatasetIndexer:
    """Scan the dataset folder and build a tabular image index.

    Accepts an optional `data_dir` so callers can index custom raw data
    directories instead of the default `RAW_DATA_DIR` from config.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else RAW_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir = PROCESSED_DATA_DIR
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.supported_extensions = SUPPORTED_EXTENSIONS

    def build_dataframe(self) -> pd.DataFrame:
        """Return one row per image with file path, label, and dimensions."""

        records : list[ImageRecord] = []
        for file_path in self.data_dir.rglob("*"):

            ##Skip the non-supported files
            if file_path.suffix.lower() not in self.supported_extensions:
                continue

            image = cv2.imread(str(file_path))
            if image is None:
                continue

            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1
            label = file_path.parent.name

            records.append(ImageRecord(file_path, label, width, height, channels))
        return pd.DataFrame(records)

    def save_indexed_data(self, df,filename: str = 'processed_data.csv') -> Path:
        """Save the processed dataframe to the processed data directory."""

        output_path = self.processed_data_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Indexed data saved to {output_path}")
        return output_path
