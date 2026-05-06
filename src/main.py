from utils import kaggleHub
from services import datasetIndexer

kaggleHub.download()
datasetIndexer.DatasetIndexer().build_dataframe()
