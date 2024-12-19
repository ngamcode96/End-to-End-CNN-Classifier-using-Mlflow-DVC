import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig

# 6. components
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Fetch data from url

        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            root_dir = self.config.root_dir
            os.makedirs(root_dir, exist_ok=True)

            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")
            file_id = dataset_url.split('/')[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id, zip_download_dir)
            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    def extract_zip_file(self):
        """
            Extract zip file into the data directory
            Function return None
        """
        unzip_dir = self.config.unzip_dir
        os.makedirs(unzip_dir, exist_ok=True)

        logger.info(f"Extracting all files from the zip file: {self.config.local_data_file}")
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
        
        logger.info(f"Extracted files successfully from the zip file: {self.config.local_data_file}")

    
        