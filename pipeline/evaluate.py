"""
Evaluation script for trained model
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
    

# Add project root to path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# from src.dataset import get_train_val_dataloaders
from src.dataset import get_train_val_dataloaders
from src.model import UNet3D
from src.vnet import VNet3D
from src.metrics import MetricsTracker, dice_coefficient


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Create model
    model_type = config.get('model_type', 'unet').lower()
    if model_type == 'vnet':
        model = VNet3D(
            in_channels=config['in_channels'],
            num_classes=config['num_classes'],
            base_channels=config['base_channels']
        ).to(device)
    else:
        model = UNet3D(
            in_channels=config['in_channels'],
            num_classes=config['num_classes'],
            base_channels=config['base_channels']
        ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
    print(f"Best validation Dice: {checkpoint['best_dice']:.4f}")
    
    return model, config


def evaluate_model(model, val_loader, device, num_classes=3):
    """Evaluate model on validation set"""
    metrics_tracker = MetricsTracker(num_classes=num_classes)
    
    print("\nEvaluating model...")
    print("="*70)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            # Move to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Update metrics
            metrics_tracker.update(outputs.cpu(), labels.cpu())
            
            # Store predictions
            pred_classes = torch.argmax(outputs, dim=1)
            all_predictions.append(pred_classes.cpu())
            all_labels.append(labels.cpu())
    
    # Print metrics
    metrics_tracker.print_metrics()
    
    # Get average metrics
    avg_metrics = metrics_tracker.get_average_metrics()
    
    return avg_metrics, all_predictions, all_labels


def visualize_predictions(model, val_loader, device, output_dir, num_samples=5):
    """
    Visualize predictions on validation samples
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    print("="*70)
    
    model.eval()
    
    samples_visualized = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            if samples_visualized >= num_samples:
                break
            
            # Move to device
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Move to CPU for visualization
            images = images.cpu().numpy()
            labels = labels.cpu().numpy()
            predictions = predictions.cpu().numpy()
            
            # Visualize each sample in batch
            batch_size = images.shape[0]
            for i in range(batch_size):
                if samples_visualized >= num_samples:
                    break
                
                visualize_sample(
                    images[i, 0],  # Remove channel dimension
                    labels[i],
                    predictions[i],
                    sample_idx=samples_visualized,
                    output_dir=output_dir
                )
                
                samples_visualized += 1
    
    print(f"✓ Saved {samples_visualized} visualizations to {output_dir}")


def visualize_sample(image, label, prediction, sample_idx, output_dir):
    """
    Create visualization for a single sample
    """
    # Get middle slices
    d, h, w = image.shape
    mid_d = d // 2
    mid_h = h // 2
    mid_w = w // 2
    
    # Calculate Dice for this sample
    dice = dice_coefficient(
        torch.from_numpy(prediction).unsqueeze(0),
        torch.from_numpy(label).unsqueeze(0),
        num_classes=3
    )
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Sample {sample_idx + 1} - Mean Dice: {dice["mean_dice"]:.4f}', 
                 fontsize=16, fontweight='bold')
    
    views = [
        (mid_d, slice(None), slice(None), 'Axial'),
        (slice(None), mid_h, slice(None), 'Coronal'),
        (slice(None), slice(None), mid_w, 'Sagittal')
    ]
    
    for row, (d_slice, h_slice, w_slice, view_name) in enumerate(views):
        # Extract slice
        img_slice = image[d_slice, h_slice, w_slice]
        label_slice = label[d_slice, h_slice, w_slice]
        pred_slice = prediction[d_slice, h_slice, w_slice]
        
        # Image
        axes[row, 0].imshow(img_slice, cmap='gray')
        axes[row, 0].set_title(f'{view_name} - Image')
        axes[row, 0].axis('off')
        
        # Ground truth
        axes[row, 1].imshow(img_slice, cmap='gray')
        axes[row, 1].imshow(label_slice, cmap='jet', alpha=0.5, vmin=0, vmax=2)
        axes[row, 1].set_title(f'{view_name} - Ground Truth')
        axes[row, 1].axis('off')
        
        # Prediction
        axes[row, 2].imshow(img_slice, cmap='gray')
        axes[row, 2].imshow(pred_slice, cmap='jet', alpha=0.5, vmin=0, vmax=2)
        axes[row, 2].set_title(f'{view_name} - Prediction')
        axes[row, 2].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'sample_{sample_idx + 1}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(checkpoint_path, output_dir):
    """
    Plot training history curves
    """
    import json
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load history
    history_path = os.path.join(os.path.dirname(checkpoint_path), 'training_history.json')
    
    if not os.path.exists(history_path):
        print(f"Training history not found at {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Dice plot
    axes[1].plot(epochs, history['val_dice'], 'g-', label='Val Dice', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Coefficient', fontsize=12)
    axes[1].set_title('Validation Dice Score', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch = np.argmax(history['val_dice']) + 1
    best_dice = max(history['val_dice'])
    axes[1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    axes[1].text(best_epoch, best_dice, f'  Best: {best_dice:.4f}\n  (Epoch {best_epoch})',
                 fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved training history plot to {output_path}")


def evaluate_pipeline(checkpoint_path, output_dir):
    """
    Evaluate trained model and generate visualizations
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        output_dir: Directory to save evaluation results
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first")
        return
    
    print("="*70)
    print("Model Evaluation")
    print("="*70)
    
    # Get device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(checkpoint_path, device)
    
    # Load validation data
    print("\nLoading validation data...")
    _, val_loader = get_train_val_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        target_shape=tuple(config['target_shape']),
        num_workers=0
    )
    
    # Evaluate
    metrics, predictions, labels = evaluate_model(
        model, val_loader, device, config['num_classes']
    )
    
    # Visualize predictions
    visualize_predictions(
        model, val_loader, device, output_dir, num_samples=5
    )
    
    # Plot training history
    plot_training_history(checkpoint_path, output_dir)
    
    # Save metrics to file
    import json
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    
    print("\n📊 Final Results:")
    print(f"  Mean Dice Coefficient: {metrics['dice']['mean_dice']:.4f}")
    print(f"  Mean IoU: {metrics['iou']['mean_iou']:.4f}")
    if np.isnan(metrics['hausdorff']['mean_hausdorff']):
        print("  Mean Hausdorff Distance: NaN")
    else:
        print(f"  Mean Hausdorff Distance: {metrics['hausdorff']['mean_hausdorff']:.4f}")


def main():
    """Main evaluation function"""
    # Configuration
    checkpoint_path = 'results/checkpoints/best_model.pth'
    output_dir = 'results/visualizations'
    
    # Call evaluate_pipeline with default config
    evaluate_pipeline(checkpoint_path, output_dir)


if __name__ == "__main__":
    main()
