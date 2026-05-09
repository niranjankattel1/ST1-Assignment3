# Basic test for DataLoader
import pytest
import pandas as pd
from src.services.dataLoader import DataLoader

def test_data_loader():
    loader = DataLoader()
    # Mock data
    df = pd.DataFrame({
        'file_path': ['path1.jpg', 'path2.jpg'],
        'label': ['species1', 'species2'],
        'width': [128, 128],
        'height': [128, 128],
        'channels': [3, 3]
    })
    loader.df = df
    assert len(loader.get_labels()) == 2