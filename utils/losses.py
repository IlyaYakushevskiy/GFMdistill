import torch
from torch.nn import functional as F


class WeightedCrossEntropy(torch.nn.Module):
    def __init__(self, ignore_index: int, distribution: list[float]) -> None:
        super(WeightedCrossEntropy, self).__init__()
        # Initialize the weights based on the given distribution
        self.weights = [1 / w for w in distribution]

        # Convert weights to a tensor and move to CUDA
        loss_weights = torch.Tensor(self.weights).to("cuda")
        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=loss_weights
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute the weighted cross-entropy loss
        return self.loss(logits, target)


class DICELoss(torch.nn.Module):
    def __init__(self, ignore_index: int) -> None:
        super(DICELoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        num_classes = logits.shape[1]

        # Convert logits to probabilities using softmax or sigmoid
        if num_classes == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)

        # Create a mask to ignore the specified index
        mask = target != self.ignore_index
        target = target.clone()
        target[~mask] = 0

        # Convert target to one-hot encoding if necessary
        if num_classes == 1:
            target = target.unsqueeze(1)
        else:
            target = F.one_hot(target, num_classes=num_classes)
            target = target.permute(0, 3, 1, 2)

        # Apply the mask to the target
        target = target.float() * mask.unsqueeze(1).float()
        intersection = torch.sum(probs * target, dim=(2, 3))
        union = torch.sum(probs + target, dim=(2, 3))

        # Compute the Dice score
        dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
        valid_dice = dice_score[mask.any(dim=1).any(dim=1)]
        dice_loss = 1 - valid_dice.mean()  # Dice loss is 1 minus the Dice score

        return dice_loss

class DistillFeaturesLoss(torch.nn.Module):
    def __init__(self, ignore_index: int) -> None:
        super(DistillFeaturesLoss, self).__init__()
        self.ignore_index = ignore_index

class DistillLogitLoss(torch.nn.Module):

    """L_total = α * L_hard + (1 - α) * L_soft"""
    
    def __init__(self, hard_loss_fn: nn.Module, alpha: float, temperature: float):
        """Args:
        hard_loss_fn (nn.Module): The loss function for the hard labels (e.g., WeightedCrossEntropy).
        alpha (float): The weighting factor for the hard loss. Must be between 0 and 1.
        temperature (float): The temperature for softening the logits. Higher values create a softer
                                probability distribution."""

        super(DistillLogitLoss, self).__init__()

        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be between 0 and 1.")
            
        self.hard_loss_fn = hard_loss_fn
        self.soft_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_logits: torch.Tensor, target: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Calculates the total distillation loss.

        Args:
            student_logits (torch.Tensor): The output logits from the student model.
            target (torch.Tensor): The ground truth labels.
            teacher_logits (torch.Tensor): The output logits from the teacher model.

        Returns:
            torch.Tensor: The final combined loss.
        """
        loss_hard = self.hard_loss_fn(student_logits, target)

        
        soft_student_logits = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher_targets = F.softmax(teacher_logits / self.temperature, dim=1)

        # KLDivLoss expects log-probabilities for the input and probabilities for the target
        # The gradients from the soft loss are scaled by 1/T^2, so we multiply by T^2 to
        # ensure the relative contribution of the hard and soft loss is controlled by alpha.
        loss_soft = self.soft_loss_fn(soft_student_logits, soft_teacher_targets) * (self.temperature ** 2)

        loss_total = self.alpha * loss_hard + (1 - self.alpha) * loss_soft
        
        return loss_total
         