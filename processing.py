from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch

from review import Review

def tokenize(reviews: list[Review], model_name: str):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = dict()

    for review in tqdm(reviews, desc="Tokenising texts"):
        tokens = tokenizer(review.review, return_tensors='pt')
        if tokens['input_ids'].size(1) <= 512:
            inputs[review] = tokens

    return inputs

    
def inference(inputs, model_name: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    outputs = model(**inputs)
    logits = outputs.logits

    predictions = torch.argmax(logits, dim=-1)

    return predictions