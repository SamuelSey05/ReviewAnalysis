import torch


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float) -> None:
        """init procedure for FocalLoss, which works well for imbalanced datasets.

        Args:
            alpha (torch.Tensor): Weighting factor for each class.
            gamma (float): Focusing parameter to reduce the relative loss for well-classified examples.
        """

        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass to get loss of logits.

        Args:
            logits (torch.Tensor): logits tensor of shape (batch_size, num_classes).
            targets (torch.Tensor): targets tensor of shape (batch_size,).

        Returns:
            torch.Tensor: computed focal loss.
        """

        probs = torch.softmax(logits, dim=-1)

        # One-hot encode the targets for gathering true class probabilities
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=logits.shape[-1])

        true_class_probs = (probs * targets_one_hot).sum(dim=-1)

        # Calculate the focal weight for each example to increase focus on rarer classes
        focal_weight = (1 - true_class_probs) ** self.gamma

        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')

        # Apply example-by-example focal weights to cross-entropy loss
        loss = focal_weight * ce_loss

        # Apply class weights to the loss
        loss = self.alpha[targets] * loss

        return loss.mean()
        
