"""
Main Pipeline Orchestrator for Hippocampus Segmentation
Connects: Download → Explore → Train → Evaluate
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.download_data import download_dataset
from scripts.explore_data import explore_dataset, analyze_all_images
from pipeline.train import Trainer


def main():
    """Execute the complete pipeline"""

    # Step 1: Download dataset

    download_dataset()

    # Step 2: Explore dataset

    explore_dataset()
    analyze_all_images()

    # Step 3: Train model

    config = {
        "data_dir": "data/raw/Task04_Hippocampus",
        "batch_size": 2,
        "val_split": 0.2,
        "target_shape": [32, 48, 32],
        "num_workers": 0,
        "in_channels": 1,
        "num_classes": 3,
        "base_channels": 16,
        "num_epochs": 5,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "weight_ce": 0.5,
        "weight_dice": 0.5,
        "checkpoint_dir": "results/checkpoints",
        "log_dir": "results/logs",
    }
    trainer = Trainer(config)
    trainer.train()

    # Step 4: Evaluate model
    from pipeline.evaluate import evaluate_pipeline

    evaluate_pipeline(
        checkpoint_path="results/checkpoints/best_model.pth",
        output_dir="results/visualizations",
    )


if __name__ == "__main__":
    main()
