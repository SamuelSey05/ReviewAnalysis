import torch


def use_deterministic_dataloader(monkeypatch) -> None:
    """Patch torch DataLoader to deterministic single-process settings for tests."""
    original_dataloader = torch.utils.data.DataLoader

    def dataloader_no_workers(*args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["pin_memory"] = False
        kwargs["shuffle"] = False
        return original_dataloader(*args, **kwargs)

    monkeypatch.setattr(torch.utils.data, "DataLoader", dataloader_no_workers)


def weighted_dataset_loss(model, dataset, aspect_criterion, sentiment_criterion):
    device = next(model.parameters()).device
    batch = dataset.with_format("torch")[:]
    input_ids = batch["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = batch["attention_mask"].to(device=device, dtype=torch.long)
    aspects = batch["aspect"].to(device=device, dtype=torch.long)
    sentiments = batch["sentiment"].to(device=device, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        aspect_logits, sentiment_logits = model(input_ids, attention_mask)
        aspect_loss = aspect_criterion(aspect_logits, aspects)
        sentiment_loss = sentiment_criterion(sentiment_logits, sentiments)
        return (aspect_loss * 0.8) + (sentiment_loss * 0.2)
