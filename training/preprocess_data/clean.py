import pandas as pd
from training.config.config import SAMPLE_DATASET_PATH, PROCESSED_DATASET_PATH
from training.preprocess_data.utils.cleaner import remove_rows_na, remove_empty_rows
# remove rows with no labels
# dataset = dataset

dataset = pd.read_csv(SAMPLE_DATASET_PATH)
cleaned_dataset = remove_rows_na(dataset)
cleaned_dataset = remove_empty_rows(dataset, "text")
cleaned_dataset.to_csv(PROCESSED_DATASET_PATH)