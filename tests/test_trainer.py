import torch

from trainer import train_aspect_sentiment_extractor

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


def weighted_dataset_loss(model, dataset, aspect_criterion, sentiment_criterion):
    input_ids = torch.tensor(dataset["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(dataset["attention_mask"], dtype=torch.long)
    aspects = torch.tensor(dataset["aspect"], dtype=torch.long)
    sentiments = torch.tensor(dataset["sentiment"], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        aspect_logits, sentiment_logits = model(input_ids, attention_mask)
        aspect_loss = aspect_criterion(aspect_logits, aspects)
        sentiment_loss = sentiment_criterion(sentiment_logits, sentiments)
        return (aspect_loss * 0.8) + (sentiment_loss * 0.2)


def test_train_aspect_sentiment_extractor_updates_weights_and_improves_loss(monkeypatch):
    torch.manual_seed(0)
    dataloader = torch.utils.data.DataLoader

    def dataloader_no_workers(*args, **kwargs):
        """Monkeypatch DataLoader to use num_workers=0 and pin_memory=False for testing."""
        kwargs["num_workers"] = 0
        kwargs["pin_memory"] = False
        kwargs["shuffle"] = False
        return dataloader(*args, **kwargs)

    monkeypatch.setattr(torch.utils.data, "DataLoader", dataloader_no_workers)

    model = DummyExtractor()
    dataset = {
        "input_ids": [
            [1, 1, 1],
            [1, 1, 0],
            [2, 2, 2],
            [2, 2, 0],
            [3, 3, 3],
            [3, 3, 0],
        ],
        "attention_mask": [
            [1, 1, 1],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 0],
        ],
        "aspect": [0, 0, 1, 1, 2, 2],
        "sentiment": [0, 0, 1, 1, 2, 2],
    }
    aspect_loss = torch.nn.CrossEntropyLoss()
    sentiment_loss = torch.nn.CrossEntropyLoss()

    before = model.aspect_head.weight.detach().clone()
    before_loss = weighted_dataset_loss(model, dataset, aspect_loss, sentiment_loss)

    train_aspect_sentiment_extractor(
        model=model,
        dataset=dataset,
        aspect_criterion=aspect_loss,
        sentiment_criterion=sentiment_loss,
        device=torch.device("cpu"),
        num_epochs=10,
    )  

    after = model.aspect_head.weight.detach().clone()
    after_loss = weighted_dataset_loss(model, dataset, aspect_loss, sentiment_loss)

    assert not torch.equal(before, after)
    assert after_loss.item() < before_loss.item()