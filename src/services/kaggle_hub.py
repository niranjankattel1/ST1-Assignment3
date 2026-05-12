from pathlib import Path
import kagglehub

class KaggleHub:
    """KaggleHub class."""
    def __init__(self, output_dir:Path) -> None:
        self.output_dir = output_dir


    def get_raw_data(self):
        path = kagglehub.dataset_download("kennethtm/stream-macroinvertebrates", output_dir=str(self.output_dir))
        print("Dataset files available at", path)