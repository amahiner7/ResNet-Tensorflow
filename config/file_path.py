import os
from datetime import datetime

BASE_DATASET_DIR = "./datasets"
UTKFace_DIR = os.path.join(BASE_DATASET_DIR, "UTKFace")
IMAGE_DATA_FILE_PATH = os.path.join(UTKFace_DIR, "Images")
FACE_DATASET_FILE_NAME = "face_dataset.csv"
LOG_ROOT_DIR = "./log"
TRAINING_LOG_DIR = os.path.join(LOG_ROOT_DIR, "training")
DATETIME_DIR = datetime.now().strftime("%Y%m%d-%H%M%S")
TENSORBOARD_LOG_DIR = os.path.join(TRAINING_LOG_DIR, DATETIME_DIR)
TENSORBOARD_LEARNING_RATE_LOG_DIR = os.path.join(TENSORBOARD_LOG_DIR, "learning_rate")
MODEL_FILE_BASE_DIR = "./model_files"
MODEL_FILE_DIR = os.path.join(MODEL_FILE_BASE_DIR, DATETIME_DIR)
MODEL_FILE_PATH = os.path.join(MODEL_FILE_DIR, "ResNet50_UTKFace_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5")


def make_directory():
    if not os.path.exists(BASE_DATASET_DIR):
        os.mkdir(BASE_DATASET_DIR)

    if not os.path.exists(UTKFace_DIR):
        os.mkdir(UTKFace_DIR)

    if not os.path.exists(IMAGE_DATA_FILE_PATH):
        os.mkdir(IMAGE_DATA_FILE_PATH)

    if not os.path.exists(LOG_ROOT_DIR):
        os.mkdir(LOG_ROOT_DIR)

    if not os.path.exists(TRAINING_LOG_DIR):
        os.mkdir(TRAINING_LOG_DIR)

    if not os.path.exists(TENSORBOARD_LOG_DIR):
        os.mkdir(TENSORBOARD_LOG_DIR)

    if not os.path.exists(TENSORBOARD_LEARNING_RATE_LOG_DIR):
        os.mkdir(TENSORBOARD_LEARNING_RATE_LOG_DIR)

    if not os.path.exists(MODEL_FILE_BASE_DIR):
        os.mkdir(MODEL_FILE_BASE_DIR)

    if not os.path.exists(MODEL_FILE_DIR):
        os.mkdir(MODEL_FILE_DIR)
