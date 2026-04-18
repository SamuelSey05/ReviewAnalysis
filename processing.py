import logging
from functools import lru_cache
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import DEVICE
from training_review import TrainingReview

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load and cache a tokenizer for repeated reuse."""

    return AutoTokenizer.from_pretrained(model_name)


@lru_cache(maxsize=None)
def _get_encoder(model_name: str) -> nn.Module:
    """Load and cache an encoder model for repeated reuse."""

    return AutoModel.from_pretrained(model_name)


def tokenize(reviews: list[str], model_name: str) -> dict[str, torch.Tensor]:
    """Convert from words to tokens

    Args:
        reviews (list[Review]): list of reviews to convert into tokens
        model_name (str): name of model to use for tokenization

    Returns:
        dict[str, torch.Tensor]: Dictionary containing input IDs and attention masks for the tokenized reviews.
    """

    tokenizer = _get_tokenizer(model_name)
    tokens = tokenizer(
        reviews,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    return {
        "input_ids": cast(torch.Tensor, tokens["input_ids"]),
        "attention_mask": cast(torch.Tensor, tokens["attention_mask"]),
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

    model = _get_encoder(model_name)
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

def wordwise_sentiment_analysis(review: TrainingReview):
    """Analyze sentiment at word-level using VADER sentiment analyzer and aggregate to review-level rating.

    Args:
        review (TrainingReview): Training review object containing review text.

    Returns:
        float: Aggregated sentiment score scaled to range [0, 10].
    """

    analyser = SentimentIntensityAnalyzer()

    wordwise_sentiment_scores = [analyser.polarity_scores(word)["compound"] for word in review.review.split()]

    polarities = [1 if score >= 0.1 else (-1 if score <= -0.1 else 0) for score in wordwise_sentiment_scores]

    return max(0, min(5, 10 * np.mean(polarities) + 1)) * 2.5  # Scale up from -1 to 1