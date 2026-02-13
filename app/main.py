"""
FastAPI backend for training and prediction.
"""
import base64
import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent

from pipeline.train import Trainer
from src.dataset import HippocampusDataset
from src.model import UNet3D
from src.vnet import VNet3D

app = FastAPI(title="Hippocampus Segmentation API")


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_model(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_type = config.get("model_type", "unet").lower()

    if model_type == "vnet":
        model = VNet3D(
            in_channels=config["in_channels"],
            num_classes=config["num_classes"],
            base_channels=config["base_channels"],
        ).to(device)
    else:
        model = UNet3D(
            in_channels=config["in_channels"],
            num_classes=config["num_classes"],
            base_channels=config["base_channels"],
        ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def _predict_volume(model, config, device, image_np: np.ndarray) -> np.ndarray:
    image_np = HippocampusDataset.normalize_intensity(image_np.astype(np.float32))

    if image_np.shape != tuple(config["target_shape"]):
        image_np = HippocampusDataset.resize_volume(image_np, tuple(config["target_shape"]))

    image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    return image_np, pred


def _render_prediction_png(image, prediction) -> bytes:
    d, h, w = image.shape
    mid_d, mid_h, mid_w = d // 2, h // 2, w // 2

    views = [
        (mid_d, slice(None), slice(None), "Axial"),
        (slice(None), mid_h, slice(None), "Coronal"),
        (slice(None), slice(None), mid_w, "Sagittal"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle("Prediction Preview", fontsize=14, fontweight="bold")

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
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()


def _train_job(config: dict):
    trainer = Trainer(config)
    trainer.train()


@app.get("/train", response_class=HTMLResponse)
def train_page():
    return """
    <html>
      <head><title>Train Model</title></head>
      <body>
        <h2>Start Training</h2>
        <form action="/train" method="post">
          <label>Epochs:</label>
          <input name="epochs" type="number" value="5" min="1" />
          <button type="submit">Start Training</button>
        </form>
      </body>
    </html>
    """


@app.post("/train")
async def train(background_tasks: BackgroundTasks, epochs: int = 5):
    config = {
        "data_dir": "data/raw/Task04_Hippocampus",
        "batch_size": 2,
        "val_split": 0.2,
        "target_shape": [32, 48, 32],
        "num_workers": 0,
        "in_channels": 1,
        "num_classes": 3,
        "base_channels": 16,
        "model_type": "vnet",
        "num_epochs": int(epochs),
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "weight_ce": 0.5,
        "weight_dice": 0.5,
        "checkpoint_dir": "results/checkpoints",
        "log_dir": "results/logs",
    }

    background_tasks.add_task(_train_job, config)
    return {"status": "training_started", "epochs": epochs}


@app.get("/predict", response_class=HTMLResponse)
def predict_page():
    return """
    <html>
      <head><title>Predict</title></head>
      <body>
        <h2>Upload NIfTI (.nii.gz) for Prediction</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
          <input name="file" type="file" accept=".nii,.nii.gz" />
          <button type="submit">Predict</button>
        </form>
      </body>
    </html>
    """


@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=400, detail="Please upload a NIfTI file (.nii or .nii.gz).")

    device = _get_device()
    checkpoint_path = "results/checkpoints/best_model.pth"

    try:
        model, config = _load_model(checkpoint_path, device)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    data = await file.read()
    img_nii = nib.load(io.BytesIO(data))
    image = img_nii.get_fdata().astype(np.float32)

    image_proc, pred = _predict_volume(model, config, device, image)
    png_bytes = _render_prediction_png(image_proc, pred)
    png_b64 = base64.b64encode(png_bytes).decode("utf-8")

    return f"""
    <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 24px; }}
                .preview {{ max-width: 720px; }}
                img {{ width: 100%; height: auto; display: block; border: 1px solid #e5e7eb; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <h2>Prediction Result</h2>
            <div class="preview">
                <img src="data:image/png;base64,{png_b64}" />
            </div>
            <p><a href="/predict">Try another file</a></p>
        </body>
    </html>
    """


@app.get("/predict-path", response_class=HTMLResponse)
def predict_path_page():
    return """
    <html>
        <head><title>Predict by Path</title></head>
        <body>
            <h2>Predict from Local Path</h2>
            <p>Example: data/raw/Task04_Hippocampus/imagesTs/hippocampus_002.nii.gz</p>
            <form action="/predict-path" method="post">
                <input name="file_path" type="text" size="80" />
                <button type="submit">Predict</button>
            </form>
        </body>
    </html>
    """


@app.post("/predict-path", response_class=HTMLResponse)
async def predict_path(file_path: str = Form(...)):
        file_path = file_path.strip()
        if not file_path:
                raise HTTPException(status_code=400, detail="file_path is required.")

        # Restrict to project directory for safety
        abs_path = os.path.abspath(file_path)
        project_root = os.path.abspath(PROJECT_ROOT)
        if not abs_path.startswith(project_root):
                raise HTTPException(status_code=400, detail="file_path must be inside the project directory.")

        if not abs_path.endswith((".nii", ".nii.gz")):
                raise HTTPException(status_code=400, detail="file_path must be a .nii or .nii.gz file.")

        if not os.path.exists(abs_path):
                raise HTTPException(status_code=404, detail=f"File not found: {abs_path}")

        device = _get_device()
        checkpoint_path = "results/checkpoints/best_model.pth"

        try:
                model, config = _load_model(checkpoint_path, device)
        except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc))

        img_nii = nib.load(abs_path)
        image = img_nii.get_fdata().astype(np.float32)

        image_proc, pred = _predict_volume(model, config, device, image)
        png_bytes = _render_prediction_png(image_proc, pred)
        png_b64 = base64.b64encode(png_bytes).decode("utf-8")

        return f"""
        <html>
            <head>
                <title>Prediction Result</title>
                <style>
                    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 24px; }}
                    .preview {{ max-width: 720px; }}
                    img {{ width: 100%; height: auto; display: block; border: 1px solid #e5e7eb; border-radius: 8px; }}
                </style>
            </head>
            <body>
                <h2>Prediction Result</h2>
                <div class="preview">
                    <img src="data:image/png;base64,{png_b64}" />
                </div>
                <p><a href="/predict-path">Try another file</a></p>
            </body>
        </html>
        """
