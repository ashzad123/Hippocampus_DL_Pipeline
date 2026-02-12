"""
Explore and visualize the Hippocampus dataset
"""

import os
import json
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def _list_nii_gz(directory):
    return sorted(
        f
        for f in os.listdir(directory)
        if f.endswith(".nii.gz") and not f.startswith("._")
    )


def explore_dataset(data_dir="data/raw/Task04_Hippocampus"):
    """
    Explore the Hippocampus dataset structure and content
    """
    print("=" * 70)
    print("Hippocampus Dataset Exploration")
    print("=" * 70)

    # Check if dataset exists
    if not os.path.exists(data_dir):
        print(f"✗ Dataset not found at: {data_dir}")
        print("Please run: python scripts/download_data.py")
        return

    # Read dataset description
    dataset_json = os.path.join(data_dir, "dataset.json")
    if os.path.exists(dataset_json):
        print("\n📋 Dataset Information:")
        print("-" * 70)
        with open(dataset_json, "r") as f:
            info = json.load(f)
            print(f"Name: {info.get('name', 'N/A')}")
            print(f"Description: {info.get('description', 'N/A')}")
            print(f"Modality: {info.get('modality', 'N/A')}")
            print(f"Labels: {info.get('labels', 'N/A')}")
            print(f"Number of training samples: {info.get('numTraining', 'N/A')}")
            print(f"Number of test samples: {info.get('numTest', 'N/A')}")

    # Explore images
    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")

    image_files = _list_nii_gz(images_dir)
    label_files = _list_nii_gz(labels_dir)

    print(f"\n📁 Dataset Files:")
    print("-" * 70)
    print(f"Images: {len(image_files)}")
    print(f"Labels: {len(label_files)}")
    print(f"\nExample filenames:")
    for i, fname in enumerate(image_files[:3]):
        print(f"  {i+1}. {fname}")

    # Load and analyze first image
    if len(image_files) > 0:
        print(f"\n🔍 Analyzing Sample Image...")
        print("-" * 70)

        sample_image_path = os.path.join(images_dir, image_files[0])
        sample_label_path = os.path.join(labels_dir, label_files[0])

        # Load image
        img_nii = nib.load(sample_image_path)
        img_data = img_nii.get_fdata()

        # Load label
        label_nii = nib.load(sample_label_path)
        label_data = label_nii.get_fdata()

        print(f"Sample: {image_files[0]}")
        print(f"\nImage Properties:")
        print(f"  Shape: {img_data.shape}")
        print(f"  Data type: {img_data.dtype}")
        print(f"  Value range: [{img_data.min():.2f}, {img_data.max():.2f}]")
        print(f"  Mean: {img_data.mean():.2f}")
        print(f"  Std: {img_data.std():.2f}")
        print(f"  Voxel spacing: {img_nii.header.get_zooms()}")

        print(f"\nLabel Properties:")
        print(f"  Shape: {label_data.shape}")
        print(f"  Unique labels: {np.unique(label_data)}")
        print(f"  Background voxels (0): {(label_data == 0).sum()}")
        print(f"  Anterior hippocampus (1): {(label_data == 1).sum()}")
        print(f"  Posterior hippocampus (2): {(label_data == 2).sum()}")

        # Visualize
        print(f"\n📊 Creating visualization...")
        visualize_sample(img_data, label_data, image_files[0])

    print("\n" + "=" * 70)
    print("✓ Exploration complete!")
    print("=" * 70)


def visualize_sample(image, label, filename, output_dir="results/visualizations"):
    """
    Visualize a sample image and its label
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get middle slices
    d, h, w = image.shape
    mid_d = d // 2
    mid_h = h // 2
    mid_w = w // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Sample: {filename}", fontsize=16, fontweight="bold")

    # Axial view (top-down)
    axes[0, 0].imshow(image[mid_d, :, :], cmap="gray")
    axes[0, 0].set_title("Image - Axial View")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(label[mid_d, :, :], cmap="jet", vmin=0, vmax=2)
    axes[1, 0].set_title("Label - Axial View")
    axes[1, 0].axis("off")

    # Coronal view (front)
    axes[0, 1].imshow(image[:, mid_h, :], cmap="gray")
    axes[0, 1].set_title("Image - Coronal View")
    axes[0, 1].axis("off")

    axes[1, 1].imshow(label[:, mid_h, :], cmap="jet", vmin=0, vmax=2)
    axes[1, 1].set_title("Label - Coronal View")
    axes[1, 1].axis("off")

    # Sagittal view (side)
    axes[0, 2].imshow(image[:, :, mid_w], cmap="gray")
    axes[0, 2].set_title("Image - Sagittal View")
    axes[0, 2].axis("off")

    axes[1, 2].imshow(label[:, :, mid_w], cmap="jet", vmin=0, vmax=2)
    axes[1, 2].set_title("Label - Sagittal View")
    axes[1, 2].axis("off")

    plt.tight_layout()

    output_path = os.path.join(output_dir, "sample_exploration.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Visualization saved: {output_path}")
    plt.close()


def analyze_all_images(data_dir="data/raw/Task04_Hippocampus"):
    """
    Analyze all images in the dataset to understand dimensions
    """
    images_dir = os.path.join(data_dir, "imagesTr")
    image_files = _list_nii_gz(images_dir)

    print("\n📊 Analyzing all images...")
    print("-" * 70)

    shapes = []
    spacings = []

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        img_nii = nib.load(img_path)
        img_data = img_nii.get_fdata()

        shapes.append(img_data.shape)
        spacings.append(img_nii.header.get_zooms())

    shapes = np.array(shapes)
    spacings = np.array(spacings)

    print(f"Shape statistics (D, H, W):")
    print(f"  Min: {shapes.min(axis=0)}")
    print(f"  Max: {shapes.max(axis=0)}")
    print(f"  Mean: {shapes.mean(axis=0).astype(int)}")
    print(f"  Most common: {[tuple(s) for s in np.unique(shapes, axis=0)]}")

    print(f"\nSpacing statistics (D, H, W):")
    print(f"  Min: {spacings.min(axis=0)}")
    print(f"  Max: {spacings.max(axis=0)}")
    print(f"  Mean: {spacings.mean(axis=0)}")

    return shapes, spacings


if __name__ == "__main__":
    explore_dataset()
    analyze_all_images()

    print("\n" + "=" * 70)
    print(
        "✓ Data exploration complete! You can check the visualizations in the 'results/visualizations' folder."
    )
    print("=" * 70)
