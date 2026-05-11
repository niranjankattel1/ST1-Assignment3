from pathlib import Path
import cv2
import pandas as pd
import os

from matplotlib import pyplot as plt

from config import RAW_DATA_DIR, SUPPORTED_EXTENSIONS, PROCESSED_DATA_DIR, OUTPUTS_DIR
from models.image import ImageRecord


class DatasetIndexer:
    """Scan the dataset folder and build a tabular image index."""

    def __init__(self) -> None:
        self.data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR
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

    def validate_dataframe(self, df) -> pd.DataFrame:
        """Validate and clean the indexed dataframe."""

        print("Validating dataframe...")
        # Remove rows with missing file paths or labels
        df = df.dropna(subset=['file_path', 'label'])

        # Check if files exist
        df['file_exists'] = df['file_path'].apply(lambda x: os.path.exists(x))
        df = df[df['file_exists']].drop(columns=['file_exists'])

        # Ensure dimensions are reasonable (e.g., width/height > 0)
        df = df[(df['width'] > 0) & (df['height'] > 0)]

        return df.reset_index(drop=True)

    def save_indexed_data(self, df,filename: str = 'processed_data.csv') -> Path:
        """Save the processed dataframe to the processed data directory."""

        output_path = self.processed_data_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Indexed data saved to {output_path}")
        return output_path

    def save_sample_grid(
            self,
            dataframe: pd.DataFrame,
            output_path: Path = OUTPUTS_DIR / "sample_grid.png",
            sample_count: int = 9,
    ) -> None:
        """Save a grid of sample images for quick visual inspection."""

        if output_path is None:
            output_path = self.processed_data_dir / "sample_grid.png"

        sample_df = dataframe.sample(min(sample_count, len(dataframe)),random_state=42)
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        for ax, (_, row) in zip(axes.flat, sample_df.iterrows()):
            image = cv2.imread(row["file_path"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            ax.set_title(row["label"])
            ax.axis("off")
        for ax in axes.flat[len(sample_df):]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
