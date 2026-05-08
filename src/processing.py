from collections import Counter
import csv
import re
from datasets import Dataset
import json
import logging
from functools import lru_cache
from typing import cast
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import DEVICE, NEUTRAL_SENTIMENT_THRESHOLD, POSITIVE_SENTIMENT_THRESHOLD
from src.data_models import TrainingReview, TrainingOpinion

logger = logging.getLogger(__name__)

def load_csv_rows(file_path: str) -> list[dict[str, str]]:
    """Loads CSV rows into a list of dictionaries.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        list[dict[str, str]]: CSV rows keyed by column name.
    """

    with open(file_path, 'r') as file:
        return list(csv.DictReader(file))

def load_dataset_csv(file_path: str) -> tuple[dict[str, TrainingReview], list[TrainingOpinion]]:
    """Load a csv file and return its contents as a list of dictionaries.
    
    Args:
        file_path (str): Path to the csv file.
        
    Returns:
        tuple[dict[str, TrainingReview], list[TrainingOpinion]]: A tuple containing:
            - A dictionary mapping review IDs to TrainingReview objects.
            - A list of TrainingOpinion objects for rows marked as opinions.
    """
    
    logger.info(f"Loading dataset from {file_path}...")

    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        reviews = {}
        opinions = []
        for row in csv_reader:
            review = TrainingReview.from_dict(row)
            if review.review_id not in reviews:
                reviews[review.review_id] = review
            if review.is_opinion:
                # Opinions are only stored for reviews that contain opinions
                opinions.append(TrainingOpinion.from_review_and_dict(review, row))

    return reviews, opinions

def load_aspect_labels(path: str = "./resources/aspect_labels.json") -> list[str]:
    """Load aspect category labels from a JSON file.

    Args:
        path (str): Path to the JSON file containing aspect labels. Defaults to "./resources/aspect_labels.json".

    Returns:
        list[str]: List of aspect category labels.
    """
    
    labels_path = Path(path)
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)

@lru_cache(maxsize=None)
def _get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load and cache a tokenizer for repeated reuse."""

    return AutoTokenizer.from_pretrained(model_name)


@lru_cache(maxsize=None)
def _get_encoder(model_name: str) -> torch.nn.Module:
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

def extract_keyword_tokens(text: str) -> list[str]:
    """Extract normalized word-like tokens from text.

    Args:
        text (str): Input text.

    Returns:
        list[str]: Lower-cased alphanumeric tokens including apostrophes.
    """

    return re.findall(r"[a-z0-9']+", text.lower())

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
    """Analyze sentiment at word-level using VADER sentiment analyser and aggregate to review-level rating.

    Args:
        review (TrainingReview): Training review object containing review text.

    Returns:
        float: Aggregated sentiment score scaled to range [0, 10].
    """

    analyser = SentimentIntensityAnalyzer()

    wordwise_sentiment_scores = [analyser.polarity_scores(word)["compound"] for word in review.review.split()]

    polarities = [1 if score >= 0.1 else (-1 if score <= -0.1 else 0) for score in wordwise_sentiment_scores]

    return max(0, min(5, 10 * np.mean(polarities) + 1)) * 2.5  # Scale up from -1 to 1

def map_rating_to_sentiment(rating: float) -> int:
    """maps from rating to sentiment class

    Args:
        rating (float): The rating value to be mapped. In range [1, 10].

    Returns:
        int: The corresponding sentiment class (0: Negative, 1: Neutral, 2: Positive).
    """

    if rating >= POSITIVE_SENTIMENT_THRESHOLD:
        return 2  # Positive
    elif rating >= NEUTRAL_SENTIMENT_THRESHOLD:
        return 1  # Neutral
    else:
        return 0  # Negative
    
def prepare_aspect_dataset(
        opinions: list, 
        review_ids: list[str], 
        review_inputs: dict[str, torch.Tensor], 
        true_sentiments: list[int]
        ) -> tuple[Dataset, list[str], torch.Tensor, dict[str, int]]:
    """Prepare a dataset for aspect classification from opinion-level aspect annotations.

    Args:
        opinions (list): List of TrainingOpinion objects with category and review attributes.
        review_ids (list[str]): List of review IDs.
        review_inputs (dict[str, torch.Tensor]): Dictionary containing tokenized review inputs (input_ids and attention_mask).
        true_sentiments (list[int]): List of true sentiment labels for reviews.

    Returns:
        tuple[Dataset, list[str], torch.Tensor, dict[str, int]]: A tuple containing:
            - Dataset: Hugging Face Dataset with tokenized sentences and labels
            - list[str]: Sorted list of unique aspects
            - torch.Tensor: Class weights for handling aspect imbalance
            - dict[str, int]: Mapping from review IDs to their indices
    """

    # Sort aspects to have consistent indexing
    aspects = sorted(set([opinion.category for opinion in opinions]))
    aspect_to_idx = {aspect: idx for idx, aspect in enumerate(aspects)}

    review_id_to_idx = {review_id: idx for idx, review_id in enumerate(review_ids)}

    # Calculate aspect weights to handle class imbalance
    aspects_counter = Counter([opinion.category for opinion in opinions])
    total_opinions = len(opinions)
    aspect_weights = torch.tensor([total_opinions / (len(aspects) * aspects_counter[aspect]) for aspect in aspects], dtype=torch.float).to(DEVICE)
    
    # Make dataset on opinion by opinion basis
    tokenised_opinion_dataset = Dataset.from_dict({
        "input_ids": [review_inputs["input_ids"][review_id_to_idx[opinion.review.review_id]] for opinion in opinions],
        "attention_mask": [review_inputs["attention_mask"][review_id_to_idx[opinion.review.review_id]] for opinion in opinions],
        "aspect": [aspect_to_idx[opinion.category] for opinion in opinions],
        "sentiment": [true_sentiments[review_id_to_idx[opinion.review.review_id]] for opinion in opinions]
        })
    
    tokenised_opinion_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "aspect", "sentiment"],
    )
    
    return tokenised_opinion_dataset, aspects, aspect_weights, review_id_to_idx

def pool_embeddings(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the embeddings using mean pooling, taking into account the attention mask.

    Args:
        embeddings (torch.Tensor): The input embeddings of shape (batch_size, seq_length, hidden_size).
        attention_mask (torch.Tensor): The attention mask of shape (batch_size, seq_length).

    Returns:
        torch.Tensor: The pooled embeddings of shape (batch_size, hidden_size).
    """

    # Apply attention mask to the embeddings
    masked = embeddings * attention_mask.unsqueeze(-1)
    lengths = attention_mask.sum(dim=1).clamp(min=1)

    # Mean pooling of the masked embeddings
    mean_pooled = masked.sum(dim=1) / lengths.unsqueeze(-1)
    return mean_pooled
