import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from src.logger import logging
from src.exception import CustomException
from config.paths import *

class DataProcessing(Dataset):
    def __init__(self, root, device:str = "cpu"):
        self.image_path = os.path.join(root, 'Images')
        self.labels_path = os.path.join(root, 'Labels')

        self.img_name = sorted(os.listdir(self.image_path))
        self.label_name = sorted(os.listdir(self.labels_path))
        self.device = device

    def __getitem__(self, idx):
        try:
            logging.info(f"Loading Data for index {idx}")

            img_path = os.path.join(self.image_path, self.img_name[idx])

            logging.info(f"Reading image from path: {img_path}")

            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            img_res = img_rgb / 255.0
            img_res = torch.as_tensor(img_res).permute(2, 0, 1)

            logging.info("Loading labels...")

            label_name = self.img_name[idx].rsplit('.',1)[0] + '.txt'
            label_path = os.path.join(self.labels_path, label_name)

            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found : {label_path}")

            with open(label_path, 'r') as label_file:
                l_count = int(label_file.readline())
                box = [list(map(int, label_file.readline().split())) for _ in range(l_count)]

            area = [(b[2] - b[0]) * (b[3] - b[1]) for b in box]
            labels = [1] * len(box)

            target = {
                'boxes':    torch.as_tensor(box, dtype=torch.float32),
                'area':     torch.as_tensor(area, dtype=torch.float32),
                'image_id': torch.as_tensor([idx]),
                'labels':   torch.as_tensor(labels, dtype=torch.int64),
            }

            img_res = img_res.to(self.device)
            target = {k: v.to(self.device) for k, v in target.items()}

            logging.info(f"Labels processed successfully for index {idx}")

            return img_res, target

        except Exception as e:
            logging.error(f"Error loading data for index {idx}: {e}")
            raise CustomException(e)

    def __len__(self):
        return len(self.img_name)

if __name__ == "__main__":
    root_path = TARGET_DIR
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_processor = DataProcessing(root=root_path, device=device)

    image, target = data_processor[0]
    print("Image shape:", image.shape)
    print("Target:", target)
    print("Bounding boxes:", target['boxes'])
