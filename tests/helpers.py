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
