import os
ROOT_FOLDER_PATH = os.getenv("ROOT_FOLDER_PATH","")

SAMPLE_DATASET_PATH = os.path.join(ROOT_FOLDER_PATH,"training", "data", "sample_data1.csv")
PROCESSED_DATASET_PATH = os.path.join(ROOT_FOLDER_PATH,"training", "data", "cleaned_data1.csv")

MODEL_DIR_PATH = os.getenv("MODEL_DIR_PATH","")
# HYPER PARAMETERS

EPOCHS = 6
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
EPSILON = 1e-8

