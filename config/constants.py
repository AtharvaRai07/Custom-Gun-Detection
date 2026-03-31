import torch

MODEL_CLASSES = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

