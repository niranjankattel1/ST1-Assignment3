from pathlib import Path
import cv2
import pandas as pd
from config import RAW_DATA_DIR, SUPPORTED_EXTENSIONS

class Data_indexer:
    def __init__(self, data_dir: Path = RAW_DATA_DIR):
        self.data_dir = data_dir

    def build_dataframe(self):
        records = []

        for file in self.data_dir.rglob("*"):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:

                Label = file_path.parent.name
                width_x_height = image.shape[:2]

                append.records[{
                    path: RAW_DATA_DIR.name,
                    label: Label,
                    
                }]

        return pd.build_dataframe(records)

