import kagglehub
from config import RAW_DATA_DIR

def download():
    path = kagglehub.dataset_download("kennethtm/stream-macroinvertebrates", output_dir=RAW_DATA_DIR)
    print("Path to dataset files:", path)