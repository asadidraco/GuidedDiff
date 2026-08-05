import os

import albumentations as A
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_loss_to_log,
    plot_loss,
)


# ============================================================
# Configuration
# ============================================================

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

PIN_MEMORY = DEVICE == "cuda"
LOAD_MODEL = False

CHECKPOINT_PATH = "best_checkpoint.pth.tar"
PREDICTION_FOLDER = "saved_images"

TRAIN_IMG_DIR = "data/ImageTr/"
TRAIN_MASK_DIR = "data/LabelTr/"
VAL_IMG_DIR = "data/ImageVal/"
VAL_MASK_DIR = "data/LabelVal/"


def prepare_targets(targets: torch.Tensor) -> torch.Tensor:
    """
    Convert segmentation masks to shape [B, 1, H, W]
    and binary float values in {0, 1}.
    """

    # ToTensorV2 may return masks as [B, H, W]
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)

    # Catch unexpected mask dimensions
    if targets.ndim != 4:
        raise ValueError(
            f"Expected mask shape [B, H, W] or [B, 1, H, W], "
            f"but received {tuple(targets.shape)}"
        )

    targets = targets.float()

    # Handle masks stored as 0 and 255
    if targets.max() > 1:
        targets = targets / 255.0

    # Ensure strictly binary masks
    targets = (targets > 0.5).float()

    return targets


def train_fn(
    loader,
    model,
    optimizer,
    loss_fn,
    scaler,
    device,
):
    model.train()

    loop = tqdm(loader, desc="Training", leave=False)
    running_loss = 0.0
    total_samples = 0

    for data, targets in loop:
        data = data.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        targets = prepare_targets(targets)
        targets = targets.to(
            device=device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision is enabled only when CUDA is available
        with torch.amp.autocast(
            device_type=device,
            enabled=(device == "cuda"),
        ):
            predictions = model(data)

            if predictions.shape != targets.shape:
                raise ValueError(
                    "Prediction and target shapes do not match. "
                    f"Prediction: {tuple(predictions.shape)}, "
                    f"Target: {tuple(targets.shape)}"
                )

            loss = loss_fn(predictions, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = data.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        loop.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(total_samples, 1)


def validate_fn(
    loader,
    model,
    loss_fn,
    device,
):
    model.eval()

    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data, targets in tqdm(
            loader,
            desc="Validation",
            leave=False,
        ):
            data = data.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            targets = prepare_targets(targets)
            targets = targets.to(
                device=device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type=device,
                enabled=(device == "cuda"),
            ):
                predictions = model(data)

                if predictions.shape != targets.shape:
                    raise ValueError(
                        "Prediction and target shapes do not match. "
                        f"Prediction: {tuple(predictions.shape)}, "
                        f"Target: {tuple(targets.shape)}"
                    )

                loss = loss_fn(predictions, targets)

            batch_size = data.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    return running_loss / max(total_samples, 1)


def main():
    os.makedirs(PREDICTION_FOLDER, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print("Training started.")

    train_transform = A.Compose(
        [
            A.Resize(
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
            ),

            A.Rotate(
                limit=35,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),

            A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                max_pixel_value=255.0,
            ),

            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
            ),

            A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                max_pixel_value=255.0,
            ),

            ToTensorV2(),
        ]
    )

    model = UNET(
        in_channels=3,
        out_channels=1,
    ).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if LOAD_MODEL and os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
        )

        load_checkpoint(
            checkpoint,
            model,
            optimizer,
        )

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf"),
        )

        print(
            f"Resuming from epoch {start_epoch + 1}. "
            f"Best validation loss: {best_val_loss:.4f}"
        )

    scaler = torch.amp.GradScaler(
        device=DEVICE,
        enabled=(DEVICE == "cuda"),
    )

    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(
            f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
        )

        train_loss = train_fn(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=DEVICE,
        )

        val_loss = validate_fn(
            loader=val_loader,
            model=model,
            loss_fn=loss_fn,
            device=DEVICE,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Training loss   : {train_loss:.4f}")
        print(f"Validation loss : {val_loss:.4f}")

        # Save only the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }

            save_checkpoint(
                checkpoint,
                filename=CHECKPOINT_PATH,
            )

            print(
                f"Best model saved with validation loss "
                f"{best_val_loss:.4f}"
            )

        # Compute Dice, IoU and other validation metrics
        check_accuracy(
            val_loader,
            model,
            device=DEVICE,
        )

        # Save sample predictions periodically
        if (epoch + 1) % 10 == 0 or epoch == 0:
            epoch_folder = os.path.join(
                PREDICTION_FOLDER,
                f"epoch_{epoch + 1}",
            )

            os.makedirs(epoch_folder, exist_ok=True)

            save_predictions_as_imgs(
                val_loader,
                model,
                folder=epoch_folder,
                device=DEVICE,
            )

        print(f"Finished epoch {epoch + 1}")

    save_loss_to_log(
        train_losses,
        val_losses,
    )

    plot_loss(
        train_losses,
        val_losses,
    )

    print("Training completed.")


if __name__ == "__main__":
    main()