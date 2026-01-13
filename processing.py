from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

from review import Review

import torch

def tokenize(reviews: list[Review], model_name: str) -> dict[str, list[int]]:
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

def get_word_embeddings(inputs: dict[str, list[int]], model_name: str, device: torch.device, batch_size: int = 32) -> torch.Tensor:
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    embeddings = []

    print("Generating word embeddings...")

    with torch.no_grad():
        for i in tqdm(range(0, len(inputs["input_ids"]), batch_size), desc="Generating embeddings"):
            batch_input_ids = torch.tensor(inputs["input_ids"][i:i+batch_size]).to(device)
            batch_attention_masks = torch.tensor(inputs["attention_mask"][i:i+batch_size]).to(device)
            embeddings.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_masks).last_hidden_state.cpu())

    return torch.cat(embeddings, dim=0)
    
    
def sentiment_inference(embeddings, model_name: str, device: torch.device, batch_size: int = 64):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Running inference"):
            batch_embeddings = embeddings[i:i+batch_size].to(device)
            # These stpes replicate the process done when classifying with DistilBERT from embedding to prediction
            x = model.pre_classifier(batch_embeddings[:, 0, :])
            x = torch.relu(x)
            x = model.dropout(x)
            logits = model.classifier(x)
            predictions.append(torch.argmax(logits, dim=-1))

    return torch.cat(predictions, dim=0)

def wordwise_sentiment_analysis(review: Review):
    analyser = SentimentIntensityAnalyzer()

    wordwise_sentiment_scores = [analyser.polarity_scores(word)["compound"] for word in review.review.split()]

    polarities = [1 if score >= 0.1 else (-1 if score <= -0.1 else 0) for score in wordwise_sentiment_scores]

    return max(0, min(5, 10 * np.mean(polarities) + 1)) * 2.5  # Scale from -1 to 1 into 0 to 5 (with amplification) but kept within bounds