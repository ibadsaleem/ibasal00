import numpy as np
import pandas as pd
import re
import string
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

from config.config import LEARNING_RATE, EPOCHS , EPSILON, BATCH_SIZE, PROCESSED_DATASET_PATH, MODEL_DIR_PATH
from utils.model import create_model
from utils.dataset_creator import DatasetCreator

from utils.model import predict

from utils.model import GPT2_collator

def pre_process(dataset):
    # Remove hashtag
    dataset['clean_tweet'] = dataset['text']
    return dataset





dataset = pd.read_csv(PROCESSED_DATASET_PATH).iloc[:10]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(predict(dataset , device))

