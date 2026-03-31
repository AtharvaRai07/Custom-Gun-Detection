import os
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torch import optim

from src.data_processing import DataProcessing
from src.model_artitechture import ModelArchitecture
from config.paths_config import *
from src.logger import logging
from src.exception import CustomException



class ModelTraining:
    def __init__(self, model_class, dataset_path,num_classes:int = 2, learning_rate:float = 0.001, num_epochs:int = 10, device:str = "cpu"):
        self.model_class = model_class
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device

        os.makedirs(MODEL_DIR, exist_ok=True)

        try:
             logging.info("Starting ModelTraining...")
             logging.info("Initializing ModelTraining class...")

             self.model = self.model_class(num_classes=self.num_classes, device=self.device)
             self.model.to(self.device)

             logging.info("Model initialized and moved to device successfully.")

             self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

             logging.info("Optimizer initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            raise CustomException(e, sys)

    def collate_fn(self,batch):
        return tuple(zip(*batch))

    def split_dataset(self):
        try:
            logging.info("Loading dataset...")

            dataset = DataProcessing(root=self.dataset_path, device=self.device)
            dataset = torch.utils.data.Subset(dataset, range(5))

            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size

            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=self.collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=self.collate_fn)

            logging.info("Dataset loaded and split successfully.")

            return train_loader, val_loader

        except Exception as e:
            logging.error(f"Error splitting dataset: {e}")
            raise CustomException(e, sys)

    def train(self):
        try:
            train_loader, val_loader = self.split_dataset()

            logging.info("Loaded data loaders, starting training loop...")

            for epoch in range(self.num_epochs):
                self.model.train()

            for i,(images,targets) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    losses = self.model(images,targets)

                    if isinstance(losses,dict):
                        total_loss=0
                        for key,value in losses.items():
                            if isinstance(value , torch.Tensor):
                                total_loss+=value
                        if total_loss == 0:
                            logging.warning(f"Total loss is zero at epoch {epoch}, batch {i}. Check individual losses: {losses}")

                    else:
                        total_loss = losses[0]

                    total_loss.backward()
                    self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    losses = self.model(images, targets)

                    if isinstance(losses, dict):
                        val_loss = sum(loss for loss in losses.values() if isinstance(loss, torch.Tensor))
                    elif isinstance(losses, torch.Tensor):
                        val_loss = losses
                    elif isinstance(losses, list) and len(losses) > 0:
                        val_loss = losses[0]
                    else:
                        val_loss = losses

                    logging.info(f"Validation Loss: {val_loss}")
                    logging.info(f"Validation Loss Type: {type(val_loss)}")

            torch.save(self.model.state_dict(), MODEL_SAVE_PATH)

            logging.info(f"Model saved successfully at {MODEL_SAVE_PATH}")
            logging.info("Model training completed successfully.")

        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_trainer = ModelTraining(model_class=ModelArchitecture, num_classes=2, learning_rate=0.001, num_epochs=1, device=device, dataset_path=TARGET_DIR)
    model_trainer.train()
