"""
Test all components of the pipeline
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataset import get_train_val_dataloaders
from src.model import UNet3D
from src.losses import CombinedLoss
from src.metrics import MetricsTracker
import torch


def test_dataset():
    """Test 1: Dataset Loader"""
    print("\n" + "=" * 70)
    print("TEST 1: Dataset Loader")
    print("=" * 70)
    try:
        train_loader, val_loader = get_train_val_dataloaders(
            data_dir="data/raw/Task04_Hippocampus",
            batch_size=2,
            val_split=0.2,
            target_shape=(32, 48, 32)
        )
        
        # Test loading one batch
        images, labels = next(iter(train_loader))
        print(f"\n✓ Dataset test passed!")
        print(f"  Sample batch - Images: {images.shape}, Labels: {labels.shape}")
        
        return images, labels
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        sys.exit(1)


def test_model(images):
    """Test 2: 3D U-Net Model"""
    print("\n" + "=" * 70)
    print("TEST 2: 3D U-Net Model")
    print("=" * 70)
    try:
        model = UNet3D(in_channels=1, num_classes=3, base_channels=16)
        print(f"\n✓ Model created successfully!")
        print(f"  Parameters: {model.count_parameters():,}")
        
        # Test forward pass
        with torch.no_grad():
            output = model(images)
        print(f"  Forward pass - Input: {images.shape}, Output: {output.shape}")
        
        return model, output
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        sys.exit(1)


def test_loss_functions(output, labels):
    """Test 3: Loss Functions"""
    print("\n" + "=" * 70)
    print("TEST 3: Loss Functions")
    print("=" * 70)
    try:
        criterion = CombinedLoss(weight_ce=0.5, weight_dice=0.5)
        
        combined_loss, ce_loss, dice_loss = criterion(output, labels)
        print(f"\n✓ Loss functions working!")
        print(f"  Combined Loss: {combined_loss.item():.4f}")
        print(f"  CE Loss: {ce_loss.item():.4f}")
        print(f"  Dice Loss: {dice_loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ Loss test failed: {e}")
        sys.exit(1)


def test_metrics(output, labels):
    """Test 4: Evaluation Metrics"""
    print("\n" + "=" * 70)
    print("TEST 4: Evaluation Metrics")
    print("=" * 70)
    try:
        tracker = MetricsTracker(num_classes=3)
        tracker.update(output, labels)
        
        metrics = tracker.get_average_metrics()
        print(f"\n✓ Metrics working!")
        print(f"  Mean Dice: {metrics['dice']['mean_dice']:.4f}")
        print(f"  Mean IoU: {metrics['iou']['mean_iou']:.4f}")
        
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        sys.exit(1)


def test_device(model, images):
    """Test 5: Computing Device"""
    print("\n" + "=" * 70)
    print("TEST 5: Computing Device")
    print("=" * 70)
    try:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✓ MPS (Apple Silicon GPU) available!")
            
            # Test model on MPS
            model_test = model.to(device)
            images_test = images.to(device)
            with torch.no_grad():
                output_test = model_test(images_test)
            print(f"  MPS test passed - Output shape: {output_test.shape}")
            
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("✓ CUDA GPU available!")
        else:
            device = torch.device("cpu")
            print("✓ Using CPU")
        
        print(f"  Training will use: {device}")
        
    except Exception as e:
        print(f"✗ Device test failed: {e}")
        sys.exit(1)


def main():
    """Run all tests"""
    print("=" * 70)
    print("TESTING ALL PIPELINE COMPONENTS")
    print("=" * 70)
    
    # Run tests in sequence
    images, labels = test_dataset()
    model, output = test_model(images)
    test_loss_functions(output, labels)
    test_metrics(output, labels)
    test_device(model, images)
    
    # Summary
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    print("\nPipeline is ready for training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
