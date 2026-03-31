import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from src.logger import logging
from src.exception import CustomException


class ModelArchitecture(nn.Module):
    def __init__(self, num_classes:int = 2, device:str = "cpu"):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.optimizer = None
        self.model = self.create_model().to(self.device)

    def create_model(self):
        try:
            logging.info("Creating model...")

            model = fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            logging.info("Model created successfully.")
            return model

        except Exception as e:
            logging.error(f"Error creating model {e}")
            raise CustomException(e, sys)

    def compile(self, learning_rate:float = 0.001):
        try:
            logging.info("Compiling model...")

            self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

            logging.info("Model compiled successfully.")
        except Exception as e:
            logging.error(f"Error compiling model: {e}")
            raise CustomException(e, sys)

    def forward(self, images, targets=None):
        return self.model(images, targets)



