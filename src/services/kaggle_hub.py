from pathlib import Path
from src.config import SUPPORTED_EXTENSIONS
import random
import shutil
import kagglehub

class KaggleHub:
    """Fetch raw data from Kaggle and organize them into test and train data"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def get_raw_data(self):
        path = kagglehub.dataset_download("kennethtm/stream-macroinvertebrates", output_dir=str(self.output_dir))
        print("Dataset files available at", path)

    def organize_data(self):
        raw_root = self.output_dir / "stream_macroinvertebrates"
        if not raw_root.exists() or not raw_root.is_dir():
            raise FileNotFoundError(
                f"Expected raw data folder not found: {raw_root}. "
                "Run get_raw_data() first or verify the dataset location."
            )

        expected_dirs = [
            "Gammarus sp",
            "Simuliidae sp",
            "Asellus sp",
            "Sericostomatidae sp",
            "Oligochaeta sp",
        ]

        selected_dirs = []
        for folder_name in expected_dirs:
            folder_path = raw_root / folder_name
            if not folder_path.exists() or not folder_path.is_dir():
                raise FileNotFoundError(
                    f"Expected folder not found: {folder_path}. "
                    "Verify the raw dataset structure under stream_macroinvertebrates."
                )
            selected_dirs.append(folder_path)

        test_data_root = self.output_dir.parent / "test_data"
        test_data_root.mkdir(parents=True, exist_ok=True)

        for species_dir in selected_dirs:
            image_files = [file for file in species_dir.iterdir()
                           if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS]
            if len(image_files) < 5:
                raise ValueError(
                    f"Expected at least 5 images in {species_dir}, found {len(image_files)}."
                )

            selected_images = random.sample(image_files, 5)

            for image_path in selected_images:
                destination = test_data_root / image_path.name
                if destination.exists():
                    destination = test_data_root / f"{species_dir.name}_{image_path.name}"
                counter = 1
                while destination.exists():
                    destination = test_data_root / f"{species_dir.name}_{counter}_{image_path.name}"
                    counter += 1
                shutil.move(str(image_path), str(destination))

        print(f"Moved 5 images from each of {len(selected_dirs)} folders into {test_data_root}.")