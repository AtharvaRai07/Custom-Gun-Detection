from pathlib import Path

######################################## DATA INGESTION PATHS ########################################

DATASET_NAME = "issaisasank/guns-object-detection"
TARGET_DIR = Path("artifacts/raw")
IMAGES_DIR = Path("artifacts/raw/Images")
LABELS_DIR = Path("artifacts/raw/Labels")

######################################## MODEL TRAINING PATHS ########################################
MODEL_DIR = Path("artifacts/model")
MODEL_SAVE_PATH = MODEL_DIR / "guns_detector.pth"

######################################## LOGGING PATHS
LOG_DIR = Path("tensorboard_logs")
