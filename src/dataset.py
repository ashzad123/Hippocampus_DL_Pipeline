"""
Dataset loader for Hippocampus segmentation
"""
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from pathlib import Path

class HippocampusDataset(Dataset):
    """
    PyTorch Dataset for Hippocampus MRI segmentation
    
    Args:
        data_dir: Path to Task04_Hippocampus directory
        mode: 'train' or 'val'
        indices: List of indices to use (for train/val split)
        target_shape: Tuple (D, H, W) to resize images to
        transform: Optional data augmentation transforms
    """
    
    def __init__(self, data_dir, mode='train', indices=None, 
                 target_shape=(32, 48, 32), transform=None):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.target_shape = target_shape
        self.transform = transform
        
        # Get image and label directories
        self.images_dir = self.data_dir / "imagesTr"
        self.labels_dir = self.data_dir / "labelsTr"
        
        # Get all files
        all_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.endswith('.nii.gz') and not f.startswith('._')
        ])
        
        # Use specified indices or all files
        if indices is not None:
            self.image_files = [all_files[i] for i in indices]
        else:
            self.image_files = all_files
        
        print(f"{mode.upper()} dataset: {len(self.image_files)} samples")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Load and preprocess a single sample
        
        Returns:
            image: Tensor of shape (1, D, H, W)
            label: Tensor of shape (D, H, W)
        """
        # Get filename
        img_filename = self.image_files[idx]
        label_filename = img_filename  # Labels have same name
        
        # Load image
        img_path = self.images_dir / img_filename
        img_nii = nib.load(img_path)
        image = img_nii.get_fdata().astype(np.float32)
        
        # Load label
        label_path = self.labels_dir / label_filename
        label_nii = nib.load(label_path)
        label = label_nii.get_fdata().astype(np.float32)
        
        # Normalize image to [0, 1]
        image = self.normalize_intensity(image)
        
        # Resize to target shape if needed
        if image.shape != self.target_shape:
            image = self.resize_volume(image, self.target_shape)
            label = self.resize_volume(label, self.target_shape, is_label=True)
        
        # Apply transforms (augmentation)
        if self.transform:
            image, label = self.transform(image, label)
        
        # Convert to tensors
        # Image: (1, D, H, W) - add channel dimension
        # Label: (D, H, W) - no channel dimension for segmentation
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dim, ensure float32
        label = torch.from_numpy(label).long()  # Long type for cross-entropy
        
        return image, label
    
    @staticmethod
    def normalize_intensity(volume):
        """
        Normalize intensity to [0, 1] range
        """
        # Clip extreme values (percentile normalization)
        p1, p99 = np.percentile(volume, (1, 99))
        volume = np.clip(volume, p1, p99)
        
        # Min-max normalization
        min_val = volume.min()
        max_val = volume.max()
        
        if max_val > min_val:
            volume = (volume - min_val) / (max_val - min_val)
        
        return volume
    
    @staticmethod
    def resize_volume(volume, target_shape, is_label=False):
        """
        Resize 3D volume to target shape using simple interpolation
        
        Args:
            volume: 3D numpy array
            target_shape: Tuple (D, H, W)
            is_label: If True, use nearest neighbor (for labels)
        """
        from scipy.ndimage import zoom
        
        current_shape = volume.shape
        zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
        
        if is_label:
            # Use nearest neighbor for labels to preserve integer values
            resized = zoom(volume, zoom_factors, order=0)
        else:
            # Use linear interpolation for images
            resized = zoom(volume, zoom_factors, order=1)
        
        return resized


class RandomFlip:
    """
    Randomly flip volume along each axis
    """
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, label):
        # Flip along each axis with probability
        for axis in range(3):
            if np.random.random() < self.prob:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
        
        return image, label


class RandomRotate90:
    """
    Randomly rotate volume by 90 degrees
    """
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, label):
        if np.random.random() < self.prob:
            # Random number of 90-degree rotations
            axes = (1, 2)  # Rotate in-plane (H, W) to preserve (D, H, W)
            if image.shape[1] == image.shape[2]:
                k = np.random.randint(1, 4)  # 1, 2, or 3 rotations
            else:
                k = 2  # 180-degree rotation to preserve shape when H != W
            image = np.rot90(image, k=k, axes=axes).copy()
            label = np.rot90(label, k=k, axes=axes).copy()
        
        return image, label


class Compose:
    """
    Compose multiple transforms
    """
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


def get_train_val_dataloaders(data_dir, batch_size=2, val_split=0.2, 
                               target_shape=(32, 48, 32), num_workers=0):
    """
    Create train and validation dataloaders with proper splitting
    
    Args:
        data_dir: Path to Task04_Hippocampus
        batch_size: Batch size for training
        val_split: Fraction of data for validation (0.2 = 20%)
        target_shape: Target shape for resizing
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    
    # Get total number of samples
    images_dir = Path(data_dir) / "imagesTr"
    all_files = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith('.nii.gz') and not f.startswith('._')
    ])
    n_samples = len(all_files)
    
    # Create indices
    indices = list(range(n_samples))
    
    # Split into train and validation
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=val_split, 
        random_state=42,
        shuffle=True
    )
    
    print(f"Total samples: {n_samples}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    # Create augmentation transforms for training
    train_transform = Compose([
        RandomFlip(prob=0.5),
        RandomRotate90(prob=0.5),
    ])
    
    # Create datasets
    train_dataset = HippocampusDataset(
        data_dir=data_dir,
        mode='train',
        indices=train_indices,
        target_shape=target_shape,
        transform=train_transform
    )
    
    val_dataset = HippocampusDataset(
        data_dir=data_dir,
        mode='val',
        indices=val_indices,
        target_shape=target_shape,
        transform=None  # No augmentation for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset loader
    print("Testing HippocampusDataset...")
    print("=" * 70)
    
    data_dir = "data/raw/Task04_Hippocampus"
    
    # Create dataloaders
    train_loader, val_loader = get_train_val_dataloaders(
        data_dir=data_dir,
        batch_size=2,
        val_split=0.2,
        target_shape=(32, 48, 32)
    )
    
    # Test loading a batch
    print("\nTesting data loading...")
    print("-" * 70)
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")  # Should be (batch, 1, D, H, W)
        print(f"  Labels shape: {labels.shape}")  # Should be (batch, D, H, W)
        print(f"  Images range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Unique labels: {torch.unique(labels)}")
        
        if batch_idx >= 2:  # Test 3 batches
            break
    
    print("\n" + "=" * 70)
    print("✓ Dataset loader working correctly!")
    print("=" * 70)
