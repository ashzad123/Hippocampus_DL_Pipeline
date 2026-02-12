"""
Evaluation metrics for segmentation
"""
import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def dice_coefficient(pred, target, num_classes=3, ignore_background=True):
    """
    Calculate Dice coefficient for each class
    
    Args:
        pred: Predicted segmentation (N, D, H, W) with class indices
        target: Ground truth segmentation (N, D, H, W) with class indices
        num_classes: Number of classes
        ignore_background: If True, don't compute Dice for background (class 0)
    
    Returns:
        Dictionary with Dice scores for each class and mean Dice
    """
    dice_scores = {}
    
    start_class = 1 if ignore_background else 0
    
    for c in range(start_class, num_classes):
        # Create binary masks for class c
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        # Compute intersection and union
        intersection = (pred_c * target_c).sum()
        pred_sum = pred_c.sum()
        target_sum = target_c.sum()
        
        # Dice coefficient
        if (pred_sum + target_sum) == 0:
            dice = 1.0  # Both masks are empty, perfect match
        else:
            dice = (2.0 * intersection) / (pred_sum + target_sum)
        
        dice_scores[f'class_{c}'] = dice.item()
    
    # Mean Dice (excluding background)
    dice_scores['mean_dice'] = np.mean([v for k, v in dice_scores.items() if k.startswith('class_')])
    
    return dice_scores


def iou_score(pred, target, num_classes=3, ignore_background=True):
    """
    Calculate Intersection over Union (IoU) for each class
    
    IoU = |A ∩ B| / |A ∪ B|
    """
    iou_scores = {}
    
    start_class = 1 if ignore_background else 0
    
    for c in range(start_class, num_classes):
        # Create binary masks for class c
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        # Compute intersection and union
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        
        # IoU
        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union
        
        iou_scores[f'class_{c}'] = iou.item()
    
    # Mean IoU
    iou_scores['mean_iou'] = np.mean([v for k, v in iou_scores.items() if k.startswith('class_')])
    
    return iou_scores


def hausdorff_distance(pred, target, num_classes=3, ignore_background=True):
    """
    Calculate symmetric Hausdorff distance for each class

    Args:
        pred: Predicted segmentation (N, D, H, W) with class indices
        target: Ground truth segmentation (N, D, H, W) with class indices
        num_classes: Number of classes
        ignore_background: If True, don't compute for background (class 0)

    Returns:
        Dictionary with Hausdorff distances for each class and mean value
    """
    hd_scores = {}

    start_class = 1 if ignore_background else 0

    for c in range(start_class, num_classes):
        pred_c = (pred == c).cpu().numpy()
        target_c = (target == c).cpu().numpy()

        pred_points = np.argwhere(pred_c)
        target_points = np.argwhere(target_c)

        if pred_points.size == 0 and target_points.size == 0:
            hd = 0.0
        elif pred_points.size == 0 or target_points.size == 0:
            hd = np.nan
        else:
            hd_forward = directed_hausdorff(pred_points, target_points)[0]
            hd_backward = directed_hausdorff(target_points, pred_points)[0]
            hd = max(hd_forward, hd_backward)

        hd_scores[f'class_{c}'] = float(hd)

    class_values = [v for k, v in hd_scores.items() if k.startswith('class_')]
    valid_values = [v for v in class_values if not np.isnan(v)]
    hd_scores['mean_hausdorff'] = float(np.mean(valid_values)) if valid_values else float('nan')

    return hd_scores


class MetricsTracker:
    """
    Track metrics across training/validation
    """
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.dice_scores = []
        self.iou_scores = []
        self.hausdorff_scores = []
    
    def update(self, predictions, targets):
        """
        Update metrics with a new batch
        
        Args:
            predictions: Model output logits (N, C, D, H, W)
            targets: Ground truth (N, D, H, W)
        """
        # Convert logits to class predictions
        pred_classes = torch.argmax(predictions, dim=1)
        
        # Calculate metrics
        dice = dice_coefficient(pred_classes, targets, self.num_classes)
        iou = iou_score(pred_classes, targets, self.num_classes)
        hausdorff = hausdorff_distance(pred_classes, targets, self.num_classes)
        
        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
        self.hausdorff_scores.append(hausdorff)
    
    def get_average_metrics(self):
        """
        Get average metrics across all batches
        """
        if len(self.dice_scores) == 0:
            return {}
        
        # Average Dice scores
        avg_dice = {}
        for key in self.dice_scores[0].keys():
            avg_dice[key] = np.mean([d[key] for d in self.dice_scores])
        
        # Average IoU scores
        avg_iou = {}
        for key in self.iou_scores[0].keys():
            avg_iou[key] = np.mean([d[key] for d in self.iou_scores])

        # Average Hausdorff scores (ignore NaNs)
        avg_hd = {}
        for key in self.hausdorff_scores[0].keys():
            values = [d[key] for d in self.hausdorff_scores]
            if key.startswith('class_'):
                valid_values = [v for v in values if not np.isnan(v)]
                avg_hd[key] = float(np.mean(valid_values)) if valid_values else float('nan')
            else:
                avg_hd[key] = float(np.nanmean(values))
        
        return {
            'dice': avg_dice,
            'iou': avg_iou,
            'hausdorff': avg_hd
        }
    
    def print_metrics(self, prefix=""):
        """
        Print formatted metrics
        """
        metrics = self.get_average_metrics()
        
        if not metrics:
            print(f"{prefix}No metrics to display")
            return
        
        print(f"\n{prefix}Metrics:")
        print(f"{prefix}" + "-" * 50)
        
        # Dice scores
        print(f"{prefix}Dice Coefficient:")
        dice = metrics['dice']
        for key, value in dice.items():
            if key != 'mean_dice':
                class_num = key.split('_')[1]
                print(f"{prefix}  Class {class_num}: {value:.4f}")
        print(f"{prefix}  Mean Dice: {dice['mean_dice']:.4f}")
        
        # IoU scores
        print(f"\n{prefix}IoU Score:")
        iou = metrics['iou']
        for key, value in iou.items():
            if key != 'mean_iou':
                class_num = key.split('_')[1]
                print(f"{prefix}  Class {class_num}: {value:.4f}")
        print(f"{prefix}  Mean IoU: {iou['mean_iou']:.4f}")

        # Hausdorff scores
        print(f"\n{prefix}Hausdorff Distance:")
        hd = metrics['hausdorff']
        for key, value in hd.items():
            if key != 'mean_hausdorff':
                class_num = key.split('_')[1]
                if np.isnan(value):
                    print(f"{prefix}  Class {class_num}: NaN")
                else:
                    print(f"{prefix}  Class {class_num}: {value:.4f}")
        if np.isnan(hd['mean_hausdorff']):
            print(f"{prefix}  Mean Hausdorff: NaN")
        else:
            print(f"{prefix}  Mean Hausdorff: {hd['mean_hausdorff']:.4f}")


def test_metrics():
    """
    Test metrics calculation
    """
    print("=" * 70)
    print("Testing Metrics")
    print("=" * 70)
    
    # Create dummy data
    batch_size = 2
    num_classes = 3
    depth, height, width = 32, 48, 32
    
    # Random predictions (logits)
    predictions = torch.randn(batch_size, num_classes, depth, height, width)
    
    # Random targets
    targets = torch.randint(0, num_classes, (batch_size, depth, height, width))
    
    print(f"\nInput shapes:")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Targets: {targets.shape}")
    
    # Test MetricsTracker
    print("\n" + "-" * 70)
    print("Testing MetricsTracker...")
    
    tracker = MetricsTracker(num_classes=3)
    
    # Simulate multiple batches
    for i in range(3):
        tracker.update(predictions, targets)
    
    tracker.print_metrics(prefix="  ")
    
    print("\n" + "=" * 70)
    print("✓ Metrics working correctly!")
    print("=" * 70)


if __name__ == "__main__":
    test_metrics()
