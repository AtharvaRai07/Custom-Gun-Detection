import os
import sys
import kagglehub
import shutil
from src.logger import logging
from src.exception import CustomException
from config.paths import *
import zipfile

class DataIngestion:
    def __init__(self, dataset_name:str, target_dir:str):
        self.dataset_name = dataset_name
        self.target_dir = target_dir

    def create_raw_dir(self):
        try:
            os.makedirs(self.target_dir, exist_ok=True)
            logging.info(f"Created the {self.target_dir} directory.")
        except Exception as e:
            logging.error("Error while creating directory..")
            raise CustomException(e, sys)

    def extract_images_and_labels(self, path:str):
        try:
            if path.endswith('.zip'):
                logging.info("Extracting zip file")
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(self.target_dir)

            images_folder = os.path.join(path, "Images")
            labels_folder = os.path.join(path, "Labels")

            if os.path.exists(images_folder):
                shutil.move(images_folder, IMAGES_DIR)
                logging.info("Images moved successfully..")
            else:
                logging.info("Images folder doesn't exist..")

            if os.path.exists(labels_folder):
                shutil.move(labels_folder, LABELS_DIR)
                logging.info("Labels moved successfully..")

            logging.info("Images and Labels extracted and moved successfully..")

        except Exception as e:
            logging.error("Error while extracting .")
            raise CustomException(e, sys)

    def download_dataset(self):
        try:
            path = kagglehub.dataset_download(self.dataset_name)
            logging.info(f"Downloaded the data from {path}")

            self.extract_images_and_labels(path)

        except Exception as e:
            logging.error("Error while downloading dataset.")
            raise CustomException(e, sys)


    def run(self):
        try:
            logging.info("Starting the data ingestion process.")

            self.create_raw_dir()
            self.download_dataset()

            logging.info("Data ingestion process completed successfully.")

        except Exception as e:
            logging.error("Error while data ingestion process.")


if __name__ == "__main__":
    data_ingestion = DataIngestion(dataset_name=DATASET_NAME, target_dir=TARGET_DIR)
    data_ingestion.run()
