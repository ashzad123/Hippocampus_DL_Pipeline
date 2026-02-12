"""
Run inference on official test set (imagesTs) and save predictions.
"""
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataset import HippocampusDataset
from src.model import UNet3D


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = UNet3D(
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
        base_channels=config["base_channels"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
    print(f"Best validation Dice: {checkpoint['best_dice']:.4f}")

    return model, config


def _save_test_visualization(image, prediction, output_path, title):
    """Save a 3-view visualization (axial/coronal/sagittal) for a test sample."""
    d, h, w = image.shape
    mid_d = d // 2
    mid_h = h // 2
    mid_w = w // 2

    views = [
        (mid_d, slice(None), slice(None), "Axial"),
        (slice(None), mid_h, slice(None), "Coronal"),
        (slice(None), slice(None), mid_w, "Sagittal"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for row, (d_slice, h_slice, w_slice, view_name) in enumerate(views):
        img_slice = image[d_slice, h_slice, w_slice]
        pred_slice = prediction[d_slice, h_slice, w_slice]

        axes[row, 0].imshow(img_slice, cmap="gray")
        axes[row, 0].set_title(f"{view_name} - Image")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(img_slice, cmap="gray")
        axes[row, 1].imshow(pred_slice, cmap="jet", alpha=0.5, vmin=0, vmax=2)
        axes[row, 1].set_title(f"{view_name} - Prediction")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_test_set_inference(
    checkpoint_path,
    output_dir,
    test_dir=None,
    visualization_dir=None,
    num_visualizations=5,
):
    """
    Run inference on official test set (imagesTs) and save predictions as NIfTI.

    Args:
        checkpoint_path: Path to trained model checkpoint
        output_dir: Directory to save predictions
        test_dir: Optional path to imagesTs directory
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model, config = load_model(checkpoint_path, device)

    if test_dir is None:
        test_dir = os.path.join(config["data_dir"], "imagesTs")

    if not os.path.exists(test_dir):
        print(f"Test set directory not found: {test_dir}")
        return

    test_files = sorted(
        [f for f in os.listdir(test_dir) if f.endswith(".nii.gz") and not f.startswith("._")]
    )

    if len(test_files) == 0:
        print("No test files found.")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if visualization_dir is not None:
        Path(visualization_dir).mkdir(parents=True, exist_ok=True)

    print("\nRunning inference on official test set...")
    print("=" * 70)
    print(f"Test samples: {len(test_files)}")
    print(f"Saving predictions to: {output_dir}")

    with torch.no_grad():
        for idx, fname in enumerate(tqdm(test_files, desc="Test Inference")):
            img_path = os.path.join(test_dir, fname)
            img_nii = nib.load(img_path)
            image = img_nii.get_fdata().astype(np.float32)

            # Preprocess (same as training)
            image = HippocampusDataset.normalize_intensity(image)
            if image.shape != tuple(config["target_shape"]):
                image = HippocampusDataset.resize_volume(image, tuple(config["target_shape"]))

            # To tensor
            image_tensor = (
                torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
            )

            # Predict
            outputs = model(image_tensor)
            pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Save prediction with original affine
            pred_nii = nib.Nifti1Image(pred, affine=img_nii.affine)
            out_path = os.path.join(output_dir, fname)
            nib.save(pred_nii, out_path)

            if visualization_dir is not None and idx < num_visualizations:
                vis_path = os.path.join(
                    visualization_dir, fname.replace(".nii.gz", ".png")
                )
                _save_test_visualization(
                    image,
                    pred,
                    vis_path,
                    title=f"{fname} - Prediction",
                )

    print("✓ Test set inference complete")


def main():
    checkpoint_path = "results/checkpoints/best_model.pth"
    output_dir = "results/test_predictions"
    visualization_dir = "results/test_visualizations"
    run_test_set_inference(
        checkpoint_path,
        output_dir,
        visualization_dir=visualization_dir,
        num_visualizations=5,
    )


if __name__ == "__main__":
    main()
