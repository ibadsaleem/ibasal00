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

dataset = pd.read_csv(PROCESSED_DATASET_PATH)

# Passing encoded columns
dataset["labels"] = dataset["labels"].astype('category')

dataset['labels'] = dataset['labels'].cat.codes


dataset_copy = dataset.copy()



dataset = dataset_copy.sample(frac=0.80, random_state=0)
val_dataset = dataset_copy.drop(dataset.index)

max_len = None # Max lenght of the text for input
batch_size = BATCH_SIZE
epochs = EPOCHS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset creator for Pytorch
class DatasetCreator(Dataset):
    def __init__(self, processed_data, train):
        self.data = processed_data
        self.train = train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        line = self.data.iloc[index]
        if self.train:
            return {'text': line['text'], 'label': line['labels']}
        else:
            return {'text': line['clean_tweet'], 'label': 0}

# Class to tokenize and process the text for input to the dataloader    
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

# Function to remove hashtags....etc. Intended to improve performance   
def pre_process(dataset):
    # Remove hashtag
    dataset['clean_tweet'] = dataset['text']
    return dataset

# Function for training
def train(dataloader, optimizer, scheduler, device):
    global model
    model.train()
    predictions_labels = []
    true_labels = []
    total_loss = 0
    
    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss, logits = outputs[:2]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    avg_epoch_loss = total_loss / len(dataloader)
    return predictions_labels, true_labels, avg_epoch_loss

# Function for validation 
def validate(dataloader, device):
    global model
    model.eval()
    predictions_labels = []
    true_labels = []
    total_loss = 0
    
    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            total_loss += loss.item()
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    avg_epoch_loss = total_loss / len(dataloader)
    return predictions_labels, true_labels, avg_epoch_loss

def predict(dataloader, device):
    global model
    model.eval()
    predictions_labels = []
    
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            _, logits = outputs[:2]
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    return predictions_labels   

print('Loading gpt-2 model')

numb_of_labels = dataset["labels"].nunique()
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='gpt2', num_labels=numb_of_labels)

print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt2')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

print('Loading model...')
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path='gpt2', config=model_config)
model.resize_token_embeddings(len(tokenizer)) 
model.config.pad_token_id = model.config.eos_token_id
model.to(device)


gpt2_collator = GPT2_collator(tokenizer=tokenizer, max_seq_len=max_len)

# Prepare training data
processed_data = pre_process(dataset)
train_data = DatasetCreator(processed_data, train=True)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=gpt2_collator)
#
# Prepare validation data
val_processed = pre_process(val_dataset)
val_data = DatasetCreator(val_processed, train=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=gpt2_collator)

gpt2_collator = GPT2_collator(tokenizer=tokenizer, max_seq_len=max_len)

# Prepare training data
processed_data = pre_process(dataset)
train_data = DatasetCreator(processed_data, train=True)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=gpt2_collator)
#
# # Prepare validation data
val_processed = pre_process(val_dataset)
val_data = DatasetCreator(val_processed, train=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=gpt2_collator)
#
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPOCHS, weight_decay=0.01)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
loss = []
accuracy = []
val_loss_list = []
val_accuracy_list = []
#
for epoch in tqdm(range(epochs)):

    train_labels, true_labels, train_loss = train(train_dataloader, optimizer, scheduler, device)
    train_acc = accuracy_score(true_labels, train_labels)
    print('epoch: %.2f train accuracy %.2f' % (epoch, train_acc))
    loss.append(train_loss)
    accuracy.append(train_acc)

    val_labels, val_true_labels, val_loss = validate(val_dataloader, device)
    val_acc= accuracy_score(val_true_labels, val_labels)
    print('epoch: %.2f validation accuracy %.2f' % (epoch, val_acc))
    val_loss_list.append(val_loss)
    val_accuracy_list.append(val_acc)
# save model
model.save_pretrained(MODEL_DIR_PATH)
# save tokenizer
tokenizer.save_pretrained(MODEL_DIR_PATH)