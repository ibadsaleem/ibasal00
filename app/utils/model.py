import pandas as pd

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

from app.config.config import   PROCESSED_DATASET_PATH, MODEL_DIR_PATH
from app.utils.dataset_creator  import DatasetCreator


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GPT2_collator(object):
    def __init__(self, tokenizer, max_seq_len=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        return

    def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]
        labels = [sequence['label'] for sequence in sequences]

        # Passing encoded columns

        inputs = self.tokenizer(text=texts,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=self.max_seq_len)
        inputs.update({'labels': torch.tensor(labels)})
        return inputs

def _predict(model,dataloader, device):
    model.eval()
    predictions_labels = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = {k: v.type(torch.long).to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            _, logits = outputs[:2]
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    return predictions_labels
def predict( dataset, device=device):
    model_config, tokenizer, model = create_model()
    gpt2_collator = GPT2_collator(tokenizer=tokenizer, max_seq_len=None)
    test_data = DatasetCreator(dataset, train=False)
    test_dataloader = DataLoader(test_data, shuffle=True, collate_fn=gpt2_collator)
    result = _predict(model , test_dataloader , device)
    return tokenizer.decode(result)

def create_model():
    print('Loading gpt-2 model')
    print("dirpath", MODEL_DIR_PATH)
    numb_of_labels = 6
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=MODEL_DIR_PATH, num_labels=numb_of_labels)

    print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_DIR_PATH)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    print('Loading model...')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=MODEL_DIR_PATH,
                                                          config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)

    return model_config, tokenizer, model