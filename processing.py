import logging
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from review import Review

logger = logging.getLogger(__name__)


def tokenize(reviews: list[Review], model_name: str) -> dict[str, list[int]]:
    """Convert from words to tokens

    Args:
        reviews (list[Review]): list of reviews to convert into tokens
        model_name (str): name of model to use for tokenization

    Returns:
        dict[str, list[int]]: Dictionary containing input IDs and attention masks for the tokenized reviews.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = []
    attention_masks = []

    for review in tqdm(reviews, desc="Tokenising texts"):
        # Tokenize the review text with padding and truncation
        tokens = tokenizer(
            review.review,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors='pt'
        )

        # Add the input IDs and attention masks to the respective lists
        input_ids.append(tokens['input_ids'].squeeze(0).tolist())
        attention_masks.append(tokens['attention_mask'].squeeze(0).tolist())

    return {"input_ids": input_ids, "attention_mask": attention_masks}

def get_word_embeddings(inputs: dict[str, list[int]], model_name: str, device: torch.device, batch_size: int = 32) -> torch.Tensor:
    """Generate word embeddings for the given inputs using the specified model.

    Args:
        inputs (dict[str, list[int]]): List of tokenized inputs containing 'input_ids' and 'attention_mask'.
        model_name (str): Name or path of the pre-trained model to use for generating embeddings.
        device (torch.device): Device to run the model on (e.g., CPU, GPU).
        batch_size (int, optional): Batch size for processing inputs. Defaults to 32.

    Returns:
        torch.Tensor: Tensor containing the generated word embeddings.
    """
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    embeddings = []

    logger.info("Generating word embeddings...")

    with torch.no_grad():
        # Process inputs in batches
        for i in tqdm(range(0, len(inputs["input_ids"]), batch_size), desc="Generating embeddings"):
            batch_input_ids = torch.tensor(inputs["input_ids"][i:i+batch_size]).to(device)
            batch_attention_masks = torch.tensor(inputs["attention_mask"][i:i+batch_size]).to(device)
            embeddings.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_masks).last_hidden_state.cpu())

    # Concatenate across the batch dimension
    return torch.cat(embeddings, dim=0)
    
    
def sentiment_inference(embeddings: torch.Tensor, model: PreTrainedModel, device: torch.device, batch_size: int = 64) -> torch.Tensor:
    """Do inference to get sentiment predictions from given embeddings

    Args:
        embeddings (torch.Tensor): Tensor containing the word embeddings for the reviews
        model (PreTrainedModel): Pre-trained model to use for sentiment analysis
        device (torch.device): Device to carry out torch computations on
        batch_size (int, optional): Batch size. Defaults to 64

    Returns:
        torch.Tensor: Tensor of predicted sentiment classes (0: Negative, 1: Neutral, 2: Positive)
    """
    predictions = []

    pre_classifier = cast(nn.Module, model.pre_classifier)
    dropout = cast(nn.Module, model.dropout)
    classifier = cast(nn.Module, model.classifier)

    with torch.no_grad():
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Running inference"):
            batch_embeddings: torch.Tensor = embeddings[i:i+batch_size].to(device)
            # These steps replicate the process done when classifying with DistilBERT from embedding to prediction
            cls_embeddings = batch_embeddings[:, 0, :]
            x = pre_classifier(cls_embeddings)
            x = torch.relu(x)
            x = dropout(x)
            logits = classifier(x)

            predictions.append(torch.argmax(logits, dim=-1))

    return torch.cat(predictions, dim=0)

def wordwise_sentiment_analysis(review: Review):
    analyser = SentimentIntensityAnalyzer()

    wordwise_sentiment_scores = [analyser.polarity_scores(word)["compound"] for word in review.review.split()]

    polarities = [1 if score >= 0.1 else (-1 if score <= -0.1 else 0) for score in wordwise_sentiment_scores]

    return max(0, min(5, 10 * np.mean(polarities) + 1)) * 2.5  # Scale from -1 to 1 into 0 to 5 (with amplification) but kept within bounds