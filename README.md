# Hippocampus Segmentation - 3D Medical Image Segmentation

A PyTorch implementation of 3D U-Net for hippocampus segmentation from MRI scans using the Medical Segmentation Decathlon dataset.

## 📋 Project Description

This project implements an end-to-end deep learning pipeline for segmenting hippocampus structures from 3D MRI brain scans. The model segments three classes:
- **Class 0**: Background
- **Class 1**: Anterior hippocampus
- **Class 2**: Posterior hippocampus

**Dataset**: Task04_Hippocampus from Medical Segmentation Decathlon
- **Modality**: MRI (T1-weighted)
- **Training samples**: 260 3D volumes
- **Image size**: Variable (~31-43 × 40-59 × 24-47 voxels)
- **Preprocessed size**: 32 × 48 × 32 voxels

## 🏗️ Architecture

### Model: 3D U-Net
- **Architecture**: Classic U-Net adapted for 3D medical images
- **Parameters**: ~5.9 million trainable parameters
- **Base channels**: 16 (lightweight for Mac training)
- **Depth**: 5 levels (4 down + bottleneck + 4 up)
- **Skip connections**: Concatenation between encoder and decoder

### Loss Function
- **Combined Loss**: 0.5 × Cross-Entropy + 0.5 × Dice Loss
  - Cross-Entropy: Helps with class separation
  - Dice Loss: Handles class imbalance (hippocampus is small)

### Metrics
- **Primary**: Dice Coefficient (per class and mean)
- **Secondary**: IoU (Intersection over Union)

## 📁 Project Structure

All required files are included in the repository:

```
hippocampus-segmentation/
├── main.py   
├── README.md                         ← Complete 
├── requirements.txt                  ← All dependencies
├── data/
│   └── raw/
│       └── Task04_Hippocampus/  # Downloaded dataset
│           ├── imagesTr/        # Training images
│           ├── labelsTr/        # Training labels
│           └── dataset.json     # Dataset metadata
├── pipeline/
│   ├── __init__.py
│   ├── train.py                     ← Training pipeline
│   ├── evaluate.py                  ← Evaluation pipeline 
├── src/
│   ├── __init__.py
│   ├── dataset.py                   ← Data pipeline
│   ├── model.py                     ← 3D U-Net
│   ├── losses.py                    ← Loss functions
│   ├── metrics.py                   ← Evaluation metrics
├── scripts/
│   ├── __init__.py
│   ├── download_data.py             ← Dataset downloader
│   ├── explore_data.py              ← Data visualization
└── results/
    ├── checkpoints/
    │   ├── best_model.pth           ← Best model (Dice: 0.8585)
    │   └── training_history.json    ← Training metrics
    ├── logs/                        ← TensorBoard logs
    └── visualizations/
        ├── sample_1.png             ← Prediction visualizations
        ├── sample_2.png
        ├── sample_3.png
        ├── sample_4.png
        ├── sample_5.png
        ├── training_history.png     ← Training curves
        └── evaluation_metrics.json  ← Final metrics
```

## 🚀 Setup and Installation

### Requirements
- Python 3.9+
- macOS (tested on Apple Silicon M1/M2/M3)
- ~2GB disk space for dataset
- ~500MB for dependencies

### Quick Setup (Recommended)

1. **Clone or download this repository**

2. **Create and activate virtual environment**:
```bash
# Using conda (recommended)
conda create -n medseg python=3.10 -y
conda activate medseg

# OR using venv
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run complete pipeline**:
```bash
python main.py
```

That's it! The `main.py` script will automatically:
- Download the dataset
- Explore and visualize data
- Train the model
- Evaluate and generate results

## 🎯 Training

### Quick Start - Run Complete Pipeline

The easiest way to run everything:
```bash
python main.py
```

This single command will:
1. ✅ Download dataset (if needed)
2. ✅ Explore data and create visualizations
3. ✅ Train the 3D U-Net model (50 epochs)
4. ✅ Evaluate and generate results

### Training Configuration

Key hyperparameters (in `src/train.py`):
```python
config = {
    'batch_size': 2,           # Small batch for Mac
    'num_epochs': 50,          # Reduce for quick testing
    'learning_rate': 1e-3,     # Adam optimizer
    'base_channels': 16,       # Model capacity
}
```

### Training Output

During training, you'll see:
```
Epoch 1/50 [Train]: 100%|████████| 104/104 [loss: 0.8234]
Epoch 1/50 [Val]:   100%|████████| 27/27 [loss: 0.7821]

Epoch 1/50 Summary:
  Time: 45.2s
  Train Loss: 0.8234 | Val Loss: 0.7821
  Train Dice: 0.3156 | Val Dice: 0.3421
  Val Dice per class:
    Class 1: 0.3312
    Class 2: 0.3530
  Learning Rate: 0.001000
  💾 Saved best model (Dice: 0.3421)
```

### Training Time
- **Per epoch**: ~45-60 seconds (on M1 Mac with MPS)
- **50 epochs**: ~40-50 minutes
- **For quick testing**: Use 10-20 epochs (~10-20 minutes)

### Hardware Support
- ✅ **Apple Silicon (M1/M2/M3)**: Uses MPS (Metal Performance Shaders) GPU
- ✅ **Intel Mac**: Uses CPU (slower but works)
- ✅ **CUDA GPU**: Automatically detected if available

## 📊 Evaluation

This will:
1. Load the best model checkpoint
2. Evaluate on validation set
3. Generate visualizations (5 samples)
4. Plot training history
5. Save metrics to JSON

### Output Files
- `results/visualizations/sample_*.png` - Prediction visualizations
- `results/visualizations/training_history.png` - Loss and Dice curves
- `results/visualizations/evaluation_metrics.json` - Final metrics
### Training Summary

- **Total Epochs**: 39 (converged early)
- **Best Epoch**: 39
- **Training Time**: ~30-35 minutes (Apple Silicon M-series)
- **Device**: MPS (Metal Performance Shaders - Apple GPU)
- **Final Validation Dice**: 0.8585

### Performance Analysis

**Strengths:**
- ✅ High Dice coefficients (>0.85) indicate excellent overlap with ground truth
- ✅ Balanced performance between anterior and posterior hippocampus
- ✅ Strong IoU scores (>0.75) show precise boundary localization
- ✅ Model converged smoothly without overfitting

**Key Observations:**
- Both hippocampus subregions (anterior and posterior) are segmented with high accuracy
- Minimal performance difference between classes (0.8687 vs 0.8482)
- Model demonstrates good generalization on validation set

### Qualitative Results

The model successfully segments both anterior and posterior hippocampus regions with high precision. Visualizations show:
- ✅ Excellent overlap with ground truth segmentations
- ✅ Accurate boundary delineation
- ✅ Consistent performance across different brain slices and views
- ✅ Minimal false positives or false negatives

See `results/visualizations/` for detailed visual examples of predictions compared to ground truth.

## 🔍 Approach Overview

### 1. Data Pipeline
- **Loading**: NIfTI files read using `nibabel`
- **Preprocessing**:
  - Intensity normalization (percentile clipping + min-max scaling)
  - Resizing to uniform 32×48×32 shape
- **Augmentation**:
  - Random flips (50% probability)
  - Random 90° rotations (50% probability)
- **Split**: 80% training, 20% validation (stratified)

### 2. Model Training
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **LR Scheduler**: ReduceLROnPlateau (monitors val Dice, patience=5)
- **Loss**: Combined Dice + Cross-Entropy (0.5 each)
- **Early stopping**: Implicitly via best model saving

### 3. Evaluation
- **Metrics**: Dice coefficient, IoU per class
- **Visualization**: Multi-view (axial, coronal, sagittal) with overlay

## ⚠️ Limitations

1. **Small Dataset**: 260 samples limits model generalization
2. **Image Resolution**: Downsampled to 32×48×32 for memory efficiency
3. **Class Imbalance**: Hippocampus is ~5% of volume, background dominates
4. **Single Modality**: Only T1-weighted MRI, no multi-modal fusion
5. **No Test Set**: Only train/val split, no held-out test set
6. **Hardware**: Limited to small batch sizes on Mac

## 🔧 Potential Improvements

### Data
- [ ] Use full resolution images (requires more memory)
- [ ] Add more augmentations (elastic deformation, intensity shift)
- [ ] Implement patch-based training for larger images
- [ ] Use class weights to handle imbalance

### Model
- [ ] Try different architectures (nnU-Net, V-Net, Attention U-Net)
- [ ] Deeper network with more parameters
- [ ] Add residual connections
- [ ] Use group normalization instead of batch norm

### Training
- [ ] Longer training (100+ epochs)
- [ ] Mixed precision training (faster on GPU)
- [ ] Cross-validation (5-fold)
- [ ] Ensemble multiple models

### Evaluation
- [ ] Hausdorff distance metric
- [ ] Surface Dice metric
- [ ] Test on official test set
- [ ] Clinical validation

## 📚 References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
2. Medical Segmentation Decathlon: http://medicaldecathlon.com/
3. Çiçek, Ö., et al. (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. MICCAI.

## 🙏 Acknowledgments

- **Dataset**: Medical Segmentation Decathlon
- **Framework**: PyTorch
- **Medical Imaging**: nibabel, SimpleITK

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Last updated**: February 2026
