import logging
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import DEVICE
from training_review import TrainingReview

logger = logging.getLogger(__name__)


def tokenize(reviews: list[str], model_name: str) -> dict[str, torch.Tensor]:
    """Convert from words to tokens

    Args:
        reviews (list[Review]): list of reviews to convert into tokens
        model_name (str): name of model to use for tokenization

    Returns:
        dict[str, torch.Tensor]: Dictionary containing input IDs and attention masks for the tokenized reviews.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer(
        reviews,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    }

def get_word_embeddings(inputs: dict[str, torch.Tensor], model_name: str, batch_size: int = 32) -> torch.Tensor:
    """Generate word embeddings for the given inputs using the specified model.

    Args:
        inputs (dict[str, torch.Tensor]): Tokenized inputs containing 'input_ids' and 'attention_mask'.
        model_name (str): Name or path of the pre-trained model to use for generating embeddings.
        batch_size (int, optional): Batch size for processing inputs. Defaults to 32.

    Returns:
        torch.Tensor: Tensor containing the generated word embeddings.
    """
    model = AutoModel.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    embeddings = []

    logger.info("Generating word embeddings...")

    with torch.no_grad():
        # Process inputs in batches
        total = inputs["input_ids"].size(0)
        for i in tqdm(range(0, total, batch_size), desc="Generating embeddings"):
            batch_input_ids = inputs["input_ids"][i:i+batch_size].to(DEVICE)
            batch_attention_masks = inputs["attention_mask"][i:i+batch_size].to(DEVICE)
            embeddings.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_masks).last_hidden_state.cpu())

    # Concatenate across the batch dimension
    return torch.cat(embeddings, dim=0)
    
    
def sentiment_inference(embeddings: torch.Tensor, model: PreTrainedModel, batch_size: int = 64) -> torch.Tensor:
    """Do inference to get sentiment predictions from given embeddings

    Args:
        embeddings (torch.Tensor): Tensor containing the word embeddings for the reviews
        model (PreTrainedModel): Pre-trained model to use for sentiment analysis
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
            batch_embeddings: torch.Tensor = embeddings[i:i+batch_size].to(DEVICE)
            # These steps replicate the process done when classifying with DistilBERT from embedding to prediction
            cls_embeddings = batch_embeddings[:, 0, :]
            x = pre_classifier(cls_embeddings)
            x = torch.relu(x)
            x = dropout(x)
            logits = classifier(x)

            predictions.append(torch.argmax(logits, dim=-1))

    return torch.cat(predictions, dim=0)

def wordwise_sentiment_analysis(review: TrainingReview):
    analyser = SentimentIntensityAnalyzer()

    wordwise_sentiment_scores = [analyser.polarity_scores(word)["compound"] for word in review.review.split()]

    polarities = [1 if score >= 0.1 else (-1 if score <= -0.1 else 0) for score in wordwise_sentiment_scores]

    return max(0, min(5, 10 * np.mean(polarities) + 1)) * 2.5  # Scale up from -1 to 1