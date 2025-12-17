from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from datasets import Dataset

from review import Review

import torch

def tokenize(reviews: list[Review], model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = []
    attention_masks = []

    for review in tqdm(reviews, desc="Tokenising texts"):
        tokens = tokenizer(
            review.review,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors='pt'
        )
        input_ids.append(tokens['input_ids'].squeeze(0).tolist())
        attention_masks.append(tokens['attention_mask'].squeeze(0).tolist())

    return {"input_ids": input_ids, "attention_mask": attention_masks}

    
def inference(inputs, model_name: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Ensure inputs are in tensor format
    input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0)
    attention_mask = torch.tensor(inputs["attention_mask"]).unsqueeze(0)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    predictions = torch.argmax(logits, dim=-1)

    return predictions