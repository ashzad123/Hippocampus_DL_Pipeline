"""
Download script for Hippocampus dataset from Medical Segmentation Decathlon
"""

import os
import urllib.request
import tarfile
from pathlib import Path


# Download with progress
def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(downloaded * 100.0 / total_size, 100)
    print(
        f"\rDownloading: {percent:.1f}% ({downloaded / 1e6:.1f} MB / {total_size / 1e6:.1f} MB)",
        end="",
    )


def download_dataset(data_dir="data/raw"):
    """
    Download the Hippocampus dataset from Medical Decathlon
    """
    # Create directory if it doesn't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Dataset URL
    url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"
    tar_path = os.path.join(data_dir, "Task04_Hippocampus.tar")

    print("=" * 60)
    print("Downloading Hippocampus Dataset")
    print("=" * 60)
    print(f"Source: {url}")
    print(f"Destination: {tar_path}")
    print("Size: ~35 MB")
    print()

    try:
        if not os.path.exists(tar_path):
            print("Downloading...")
            urllib.request.urlretrieve(url, tar_path, show_progress)
            print("\n✓ Download complete!")
        else:
            print(f"File already exists: {tar_path}")
            print("Skipping download.")

        # Extract
        print("\nExtracting files...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(data_dir)
        print("✓ Extraction complete!")

        # Show structure
        task_dir = os.path.join(data_dir, "Task04_Hippocampus")
        if os.path.exists(task_dir):
            print("\n" + "=" * 60)
            print("Dataset Structure:")
            print("=" * 60)
            for root, dirs, files in os.walk(task_dir):
                level = root.replace(task_dir, "").count(os.sep)
                indent = " " * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files
                    print(f"{subindent}{file}")
                if len(files) > 5:
                    print(f"{subindent}... and {len(files) - 5} more files")
                if level > 1:  # Don't go too deep
                    break

        print("\n" + "=" * 60)
        print("✓ Dataset ready!")
        print("=" * 60)
        print(f"Location: {task_dir}")

        # Count files
        images_dir = os.path.join(task_dir, "imagesTr")
        labels_dir = os.path.join(task_dir, "labelsTr")

        if os.path.exists(images_dir):
            num_images = len(
                [f for f in os.listdir(images_dir) if f.endswith(".nii.gz")]
            )
            print(f"Training images: {num_images}")

        if os.path.exists(labels_dir):
            num_labels = len(
                [f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")]
            )
            print(f"Training labels: {num_labels}")

        return task_dir

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nIf download fails, you can manually download from:")
        print("http://medicaldecathlon.com/")
        print("Look for Task04_Hippocampus")
        return None


if __name__ == "__main__":
    dataset_path = download_dataset()

    if dataset_path:
        print("\n" + "=" * 60)
        print("Next we will be doing the data exploration:")
        print("=" * 60)
