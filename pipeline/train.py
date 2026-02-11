"""
Training script for Hippocampus segmentation
"""
import os
import sys
from pathlib import Path

# Add project root to path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
import time

from src.dataset import get_train_val_dataloaders
from src.model import UNet3D
from src.losses import CombinedLoss
from src.metrics import MetricsTracker


class Trainer:
    """
    Trainer class for medical image segmentation
    """
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        
        # Create directories
        Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(config['log_dir']).mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        # Load data
        print("Loading data...")
        self.train_loader, self.val_loader = get_train_val_dataloaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            val_split=config['val_split'],
            target_shape=tuple(config['target_shape']),
            num_workers=config['num_workers']
        )
        
        # Create model
        print(f"\nCreating model...")
        self.model = UNet3D(
            in_channels=config['in_channels'],
            num_classes=config['num_classes'],
            base_channels=config['base_channels']
        ).to(self.device)
        
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Training device: {self.device}")
        
        # Loss function
        self.criterion = CombinedLoss(
            weight_ce=config['weight_ce'],
            weight_dice=config['weight_dice']
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        # Metrics tracker
        self.train_metrics = MetricsTracker(num_classes=config['num_classes'])
        self.val_metrics = MetricsTracker(num_classes=config['num_classes'])
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'learning_rates': []
        }
        
        self.best_dice = 0.0
        self.start_epoch = 0
    
    def _get_device(self):
        """Get the best available device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_dice_loss = 0.0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate loss
            combined_loss, ce_loss, dice_loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            combined_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(outputs.detach().cpu(), labels.cpu())
            
            # Track losses
            total_loss += combined_loss.item()
            total_ce_loss += ce_loss.item()
            total_dice_loss += dice_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{combined_loss.item():.4f}",
                'ce': f"{ce_loss.item():.4f}",
                'dice': f"{dice_loss.item():.4f}"
            })
        
        # Calculate average losses
        avg_loss = total_loss / len(self.train_loader)
        avg_ce_loss = total_ce_loss / len(self.train_loader)
        avg_dice_loss = total_dice_loss / len(self.train_loader)
        
        # Get metrics
        train_metrics = self.train_metrics.get_average_metrics()
        train_dice = train_metrics['dice']['mean_dice']
        
        return avg_loss, avg_ce_loss, avg_dice_loss, train_dice
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_dice_loss = 0.0
        
        # Progress bar
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Val]  ")
        
        with torch.no_grad():
            for images, labels in pbar:
                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                combined_loss, ce_loss, dice_loss = self.criterion(outputs, labels)
                
                # Update metrics
                self.val_metrics.update(outputs.cpu(), labels.cpu())
                
                # Track losses
                total_loss += combined_loss.item()
                total_ce_loss += ce_loss.item()
                total_dice_loss += dice_loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{combined_loss.item():.4f}",
                    'ce': f"{ce_loss.item():.4f}",
                    'dice': f"{dice_loss.item():.4f}"
                })
        
        # Calculate average losses
        avg_loss = total_loss / len(self.val_loader)
        avg_ce_loss = total_ce_loss / len(self.val_loader)
        avg_dice_loss = total_dice_loss / len(self.val_loader)
        
        # Get metrics
        val_metrics = self.val_metrics.get_average_metrics()
        val_dice = val_metrics['dice']['mean_dice']
        
        return avg_loss, avg_ce_loss, avg_dice_loss, val_dice, val_metrics
    
    def save_checkpoint(self, epoch, val_dice, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dice': self.best_dice,
            'config': self.config,
            'history': self.history
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model (Dice: {val_dice:.4f})")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("Starting Training")
        print("="*70)
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss, train_ce, train_dice_loss, train_dice = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_ce, val_dice_loss, val_dice, val_metrics = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step(val_dice)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_dice)
            self.history['learning_rates'].append(current_lr)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Dice/train', train_dice, epoch)
            self.writer.add_scalar('Dice/val', val_dice, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} Summary:")
            print(f"  Time: {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f}")
            print(f"  Val Dice per class:")
            for key, value in val_metrics['dice'].items():
                if key != 'mean_dice':
                    class_num = key.split('_')[1]
                    print(f"    Class {class_num}: {value:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_dice > self.best_dice
            if is_best:
                self.best_dice = val_dice
            
            self.save_checkpoint(epoch, val_dice, is_best)
            
            print("-"*70)
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation Dice: {self.best_dice:.4f}")
        
        # Save final history
        history_path = os.path.join(self.config['checkpoint_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        self.writer.close()
        
        return self.history


def main():
    """Main training function"""
    # Configuration
    config = {
        # Data
        'data_dir': 'data/raw/Task04_Hippocampus',
        'batch_size': 2,  # Small batch size for Mac
        'val_split': 0.2,
        'target_shape': [32, 48, 32],
        'num_workers': 0,  # 0 for Mac compatibility
        
        # Model
        'in_channels': 1,
        'num_classes': 3,
        'base_channels': 16,
        
        # Training
        'num_epochs': 50,  # You can reduce this for quick testing
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        
        # Loss
        'weight_ce': 0.5,
        'weight_dice': 0.5,
        
        # Paths
        'checkpoint_dir': 'results/checkpoints',
        'log_dir': 'results/logs'
    }
    
    # Print configuration
    print("="*70)
    print("Training Configuration")
    print("="*70)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*70)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train
    history = trainer.train()
    
    print("\n✓ Training script completed successfully!")


if __name__ == "__main__":
    main()
