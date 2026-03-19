import torch
from tqdm import tqdm

from datasets import Dataset


def train_aspect_sentiment_extractor(
    model: torch.nn.Module,
        dataset: Dataset, 
        aspect_criterion: torch.nn.Module, 
        sentiment_criterion: torch.nn.Module, 
        device: torch.device, 
        num_epochs: int = 3
    ) -> None:
    """Train the AspectSentimentExtractor model on the given dataset and embeddings

    Args:
        model (torch.nn.Module): Model to train
        dataset (Dataset): Dataset to train the model on
        embeddings (torch.Tensor): Embeddings to use for training
        sentence_indices (list[int]): List of indices mapping each sentence to its review embedding in the embeddings tensor
        aspect_criterion (torch.nn.Module): Loss function to use for aspect classification
        sentiment_criterion (torch.nn.Module): Loss function to use for sentiment classification
        device (torch.device): Device to carry out torch computations on
        num_epochs (int, optional): Number of epochs to train for. Defaults to 3.
    """

    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Set model to training mode
    model.train()
    
    tensor_dataset = torch.utils.data.TensorDataset(
        torch.tensor(dataset["input_ids"]),
        torch.tensor(dataset["attention_mask"]),
        torch.tensor(dataset["aspect"]),
        torch.tensor(dataset["sentiment"])
    )

    # Use DataLoader for batching and shuffling
    data_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    total_loss = 0.0
    for _ in tqdm(range(num_epochs), desc="Training Aspect Sentiment Extractor"):
        total_loss = 0.0
        for batch in tqdm(data_loader, desc="Batches", leave=False):
            input_ids, attention_mask, aspects, sentiments = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            aspects, sentiments = aspects.to(device), sentiments.to(device)

            # Reset gradients
            optimiser.zero_grad()
            aspect_logits, sentiment_logits = model.forward(input_ids, attention_mask)

            # Apply loss functions and backpropagate
            aspect_loss = aspect_criterion(aspect_logits, aspects)
            sentiment_loss = sentiment_criterion(sentiment_logits, sentiments)
            loss = (aspect_loss * 0.8) + (sentiment_loss * 0.2)  # Weighted sum of aspect and sentiment losses

            loss.backward()
            optimiser.step()

            total_loss += loss.item()
    
    print(f"Training completed. Final loss: {total_loss:.4f}")