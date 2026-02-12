"""
Loss functions for medical image segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation

    Dice coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice loss = 1 - Dice coefficient
    """

    def __init__(self, smooth=1.0, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Logits of shape (N, C, D, H, W)
            targets: Ground truth of shape (N, D, H, W)
        """
        # Apply softmax to get probabilities
        probs = F.softmax(predictions, dim=1)

        # One-hot encode targets
        num_classes = predictions.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)

        # Permute to (N, C, D, H, W)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()

        # Compute Dice for each class
        dice_scores = []
        for c in range(num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue

            pred_c = probs[:, c]
            target_c = targets_one_hot[:, c]

            # Flatten
            pred_c = pred_c.contiguous().view(-1)
            target_c = target_c.contiguous().view(-1)

            # Dice coefficient
            intersection = (pred_c * target_c).sum()
            dice = (2.0 * intersection + self.smooth) / (
                pred_c.sum() + target_c.sum() + self.smooth
            )

            dice_scores.append(dice)

        # Average Dice across classes
        dice_loss = 1 - torch.stack(dice_scores).mean()

        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined Dice + Cross Entropy Loss

    This combination works well for medical segmentation:
    - Cross-entropy helps with class separation
    - Dice loss handles class imbalance
    """

    def __init__(self, weight_ce=0.5, weight_dice=0.5, class_weights=None):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

        # Cross-entropy loss with optional class weights
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss(ignore_index=0)  # Ignore background for Dice

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Logits of shape (N, C, D, H, W)
            targets: Ground truth of shape (N, D, H, W)
        """
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)

        combined = self.weight_ce * ce + self.weight_dice * dice

        return combined, ce, dice
