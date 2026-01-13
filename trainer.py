import torch
from tqdm import tqdm

from aspect_based import AspectExtractor
from datasets import Dataset


def train_aspect_extractor(model: AspectExtractor, dataset: Dataset, embeddings: torch.Tensor, sentence_indices: list[int], optimiser: torch.optim.Optimizer, aspect_criterion: torch.nn.Module, sentiment_criterion: torch.nn.Module, device: torch.device, num_epochs: int = 3) -> None:
    model.train()
    
    tensor_dataset = torch.utils.data.TensorDataset(
        torch.tensor(sentence_indices, dtype=torch.long),
        torch.tensor(dataset["attention_mask"]),
        torch.tensor(dataset["aspect"]),
        torch.tensor(dataset["sentiment"])
    )

    data_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=32, shuffle=True)

    total_loss = 0.0
    for epoch in tqdm(range(num_epochs), desc="Training Aspect Extractor"):
        total_loss = 0.0
        for batch in tqdm(data_loader, desc="Batches", leave=False):
            indices, attention_mask, aspects, sentiments = batch
            # Look up embeddings on-the-fly
            batch_embeddings = embeddings[indices].to(device)
            attention_mask = attention_mask.to(device)
            aspects, sentiments = aspects.to(device), sentiments.to(device)

            optimiser.zero_grad()
            aspect_logits, sentiment_logits = model.forward(batch_embeddings, attention_mask)

            aspect_loss = aspect_criterion(aspect_logits, aspects)
            sentiment_loss = sentiment_criterion(sentiment_logits, sentiments)
            loss = aspect_loss + sentiment_loss

            loss.backward()
            optimiser.step()

            total_loss += loss.item()
    
    print(f"Training completed. Final loss: {total_loss:.4f}")