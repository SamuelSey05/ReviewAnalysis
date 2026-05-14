import logging
from sklearn.metrics import accuracy_score
import torch
from datasets import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.config import BATCH_SIZE, DEVICE
from src.model_architecture import AspectSentimentExtractor


logger = logging.getLogger(__name__)

def plot_accuracies(aspect_accuracies, sentiment_accuracies, output_path=None):
    """Plot aspect and sentiment accuracies on the same graph.

    Args:
        aspect_accuracies (list[float]): List of aspect accuracy values (per epoch).
        sentiment_accuracies (list[float]): List of sentiment accuracy values (per epoch).
        output_path (str, optional): If provided, saves the plot to this path. Otherwise, displays it.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(aspect_accuracies, label="Aspect Accuracy")
    plt.plot(sentiment_accuracies, label="Sentiment Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Aspect and Sentiment Accuracy per Epoch")
    plt.legend()
    plt.grid(True)
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_losses(losses, output_path=None):
    """Plot a graph of training losses.

    Args:
        losses (list[float]): List of loss values (per batch or per step).
        output_path (str, optional): If provided, saves the plot to this path. Otherwise, displays it.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

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
    model: AspectSentimentExtractor,
    dataset: Dataset, 
    aspect_criterion: torch.nn.Module, 
    sentiment_criterion: torch.nn.Module, 
    num_epochs: int = 3,
    opinions: list = list(),
    device: torch.device = DEVICE,
    plot_progress: bool = False
    ) -> None:
    """Train the AspectSentimentExtractor model on the given dataset using combined aspect and sentiment losses.

    Args:
        model (torch.nn.Module): Model to train
        dataset (Dataset): Dataset to train the model on
        aspect_criterion (torch.nn.Module): Loss function to use for aspect classification
        sentiment_criterion (torch.nn.Module): Loss function to use for sentiment classification
        num_epochs (int, optional): Number of epochs to train for. Defaults to 3.
        opinions (list, optional): List of TrainingOpinion objects corresponding to the dataset, used for accuracy calculation. Defaults to list().
        device (torch.device, optional): Device to train on. Defaults to DEVICE.
        plot_progress (bool, optional): Whether to plot loss and accuracy curves after training. Defaults to False.
    """

    if plot_progress and not opinions:
        raise ValueError("Opinions must be provided to plot accuracies during training.")

    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Set model to training mode
    model.train()

    # Use DataLoader for batching and shuffling
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    total_loss = 0.0
    losses = []
    aspect_accuracies = []
    sentiment_accuracies = []
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


        losses.append(total_loss)

        if plot_progress:
            aspect_predictions, sentiment_predictions = model.aspect_sentiment_inference(list(map(lambda x: x.review.review, opinions)), batch_size=BATCH_SIZE)

            aspect_accuracies.append(accuracy_score(dataset['aspect'], aspect_predictions))
            sentiment_accuracies.append(accuracy_score(dataset['sentiment'], sentiment_predictions))

    
    logger.info("Training completed. Final loss: %.4f", total_loss)

    if plot_progress:
        plot_losses(losses, output_path="./results/model_eval/aspect_sentiment_training_loss_5_epochs.png")
        plot_accuracies(aspect_accuracies, sentiment_accuracies, output_path="./results/model_eval/aspect_sentiment_training_accuracies_5_epochs.png")
    