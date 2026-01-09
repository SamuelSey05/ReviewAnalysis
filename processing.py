from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

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
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Ensure inputs are in tensor format
    input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0)
    attention_mask = torch.tensor(inputs["attention_mask"]).unsqueeze(0)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    predictions = torch.argmax(logits, dim=-1)

    return predictions

def wordwise_sentiment_analysis(review: Review):
    analyser = SentimentIntensityAnalyzer()

    wordwise_sentiment_scores = [analyser.polarity_scores(word)["compound"] for word in review.review.split()]

    polarities = [1 if score >= 0.1 else (-1 if score <= -0.1 else 0) for score in wordwise_sentiment_scores]

    return max(0, min(5, 10 * np.mean(polarities) + 1)) * 2.5  # Scale from -1 to 1 into 0 to 5 (with amplification) but kept within bounds