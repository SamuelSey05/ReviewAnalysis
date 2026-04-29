import logging
import torch
from datasets import Dataset
from tqdm import tqdm

from src.config import DEVICE

logger = logging.getLogger(__name__)

def weighted_aspect_sentiment_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor], aspect_criterion: torch.nn.Module, sentiment_criterion: torch.nn.Module,  device: torch.device, aspect_weight: float = 0.8) -> torch.Tensor:
    """Calculate loss for both aspect and sentiment heads and combine with provided weight

    Args:
        model (torch.nn.Module): Model to calculate loss for
        batch (dict[str, torch.Tensor]): Batch to calculate loss on
        aspect_criterion (torch.nn.Module): Loss function to use for aspect classification
        sentiment_criterion (torch.nn.Module): Loss function to use for sentiment classification
        device (torch.device): Device being used
        aspect_weight (float, optional): Weighting towards aspect classification loss. Defaults to 0.8.

    Returns:
        torch.Tensor: _description_
    """

    input_ids = batch["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = batch["attention_mask"].to(device=device, dtype=torch.long)
    aspects = batch["aspect"].to(device=device, dtype=torch.long)
    sentiments = batch["sentiment"].to(device=device, dtype=torch.long)

    aspect_logits, sentiment_logits = model.forward(input_ids, attention_mask)

    # Apply loss functions and backpropagate
    aspect_loss = aspect_criterion(aspect_logits, aspects)
    sentiment_loss = sentiment_criterion(sentiment_logits, sentiments)

    return (aspect_loss * aspect_weight) + (sentiment_loss * (1 - aspect_weight)) # Weighted sum of aspect and sentiment losses


def train_aspect_sentiment_extractor(
    model: torch.nn.Module,
    dataset: Dataset, 
    aspect_criterion: torch.nn.Module, 
    sentiment_criterion: torch.nn.Module, 
    num_epochs: int = 3,
    device: torch.device = DEVICE,
    ) -> None:
    """Train the AspectSentimentExtractor model on the given dataset using combined aspect and sentiment losses.

    Args:
        model (torch.nn.Module): Model to train
        dataset (Dataset): Dataset to train the model on
        aspect_criterion (torch.nn.Module): Loss function to use for aspect classification
        sentiment_criterion (torch.nn.Module): Loss function to use for sentiment classification
        num_epochs (int, optional): Number of epochs to train for. Defaults to 3.
        device (torch.device, optional): Device to train on. Defaults to DEVICE.
    """

    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Set model to training mode
    model.train()

    # Use DataLoader for batching and shuffling
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    total_loss = 0.0
    for _ in tqdm(range(num_epochs), desc="Training Aspect Sentiment Extractor"):
        total_loss = 0.0
        for batch in tqdm(data_loader, desc="Batches", leave=False):
            # Reset gradients
            optimiser.zero_grad()

            loss = weighted_aspect_sentiment_loss(
                model=model,
                batch=batch,
                device=device,
                aspect_criterion=aspect_criterion,
                sentiment_criterion=sentiment_criterion,
            )

            loss.backward()
            optimiser.step()

            total_loss += loss.item()
    
    logger.info("Training completed. Final loss: %.4f", total_loss)