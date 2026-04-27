import torch
from datasets import Dataset

from tests.helpers import use_deterministic_dataloader
from src.trainer import train_aspect_sentiment_extractor, weighted_aspect_sentiment_loss

class DummyExtractor(torch.nn.Module):
    """
    A dummy aspect sentiment extractor model for testing the training loop.

    """

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32, 8)
        self.aspect_head = torch.nn.Linear(8, 3)
        self.sentiment_head = torch.nn.Linear(8, 3)
    
    def forward(self, input_ids, attention_mask):
        embeddings = self.embed(input_ids)
        masked = embeddings * attention_mask.unsqueeze(-1)
        lengths = attention_mask.sum(dim=1).clamp(min=1)
        pooled = masked.sum(dim=1) / lengths.unsqueeze(-1)
        return self.aspect_head(pooled), self.sentiment_head(pooled)

def test_train_aspect_sentiment_extractor_updates_weights_and_improves_loss(monkeypatch):
    torch.manual_seed(0)
    use_deterministic_dataloader(monkeypatch)

    model = DummyExtractor()
    device = next(model.parameters()).device

    dataset = Dataset.from_dict({
        "input_ids": torch.tensor([
            [1, 1, 1],
            [1, 1, 0],
            [2, 2, 2],
            [2, 2, 0],
            [3, 3, 3],
            [3, 3, 0],
        ], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([
            [1, 1, 1],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 0],
        ], dtype=torch.long, device=device),
        "aspect": torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long).to(device),
        "sentiment": torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long).to(device),
    }).with_format("torch")
    aspect_loss = torch.nn.CrossEntropyLoss()
    sentiment_loss = torch.nn.CrossEntropyLoss()

    # Device for unit test is CPU as we have smaller tensors and want to avoid overhead of GPU transfer 
    batch = dataset.with_format("torch")[:]

    batch = {k: v.to(device) for k, v in batch.items()}

    before = model.aspect_head.weight.detach().clone()
    before_loss = weighted_aspect_sentiment_loss(model, batch, aspect_loss, sentiment_loss, device)

    train_aspect_sentiment_extractor(
        model=model,
        dataset=dataset,
        aspect_criterion=aspect_loss,
        sentiment_criterion=sentiment_loss,
        num_epochs=10,
        device=device,
    )  

    after = model.aspect_head.weight.detach().clone()
    after_loss = weighted_aspect_sentiment_loss(model, batch, aspect_loss, sentiment_loss, device)

    assert not torch.equal(before, after)
    assert after_loss.item() < before_loss.item()