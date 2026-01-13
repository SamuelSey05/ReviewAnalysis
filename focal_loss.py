import torch


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)

        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=logits.shape[-1])

        true_class_probs = (probs * targets_one_hot).sum(dim=-1)

        focal_weight = (1 - true_class_probs) ** self.gamma

        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')

        loss = focal_weight * ce_loss

        loss = self.alpha[targets] * loss

        return loss.mean()
        
