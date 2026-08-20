import backend.torch_fix
import os
import sys
import time
import json
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from backend.model import (
    build_model,
    set_stage,
    count_parameters,
    device,
    MODEL_PATH,
    BEST_QWK_MODEL_PATH,
    BEST_LOSS_MODEL_PATH,
)
from backend.config import (
    APTOS_TRAIN_CSV,
    APTOS_TEST_CSV,
    APTOS_TRAIN_IMG_DIR,
    APTOS_TEST_IMG_DIR,
    BATCH_SIZE,
    VAL_SIZE,
    RANDOM_SEED,
)
from backend.preprocessing import set_seed, get_val_transforms
from backend.dataset_loader import (
    APTOSDataset,
    build_dataloaders,
    compute_class_weights,
)
from backend.metrics import compute_metrics

TRAINING_RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_runs")
os.makedirs(TRAINING_RUNS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def train_epoch(model, train_loader, criterion, optimizer, scaler, is_cuda):
    """Run one epoch of training using mixed precision if CUDA is available."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=is_cuda):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, is_cuda):
    """Run validation epoch, collect predictions, and calculate metrics."""
    model.eval()
    val_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.cuda.amp.autocast(enabled=is_cuda):
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_val_loss = val_loss / total
    metrics = compute_metrics(all_labels, all_preds, loss_val=avg_val_loss)
    return metrics


def train_model(
    epochs_stage1: int = 3,
    epochs_stage2: int = 12,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-4,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
    early_stopping_patience: int = 5,
):
    """
    Two-stage training pipeline for 5-class Diabetic Retinopathy classification on APTOS dataset.
    - Stage 1: Freeze backbone, train custom FC head.
    - Stage 2: Unfreeze upper layers (layer3, layer4, fc), fine-tune with smaller LR.
    - Uses weighted NLLLoss, AdamW, ReduceLROnPlateau, Mixed Precision, and QWK model selection.
    """
    set_seed(seed)
    is_cuda = torch.cuda.is_available()

    print("=" * 70, flush=True)
    print("      RetinaX ResNet152 2-Stage Training Pipeline", flush=True)
    print("=" * 70, flush=True)
    print(f"Device: {device} (AMP Mixed Precision: {'Enabled' if is_cuda else 'Disabled'})")
    print(f"Train CSV: {APTOS_TRAIN_CSV}")
    print(f"Train Images: {APTOS_TRAIN_IMG_DIR}")

    # Build DataLoaders
    train_loader, val_loader, class_weights, train_df, val_df = build_dataloaders(
        batch_size=batch_size,
        val_size=VAL_SIZE,
        seed=seed
    )

    criterion = nn.NLLLoss(weight=class_weights.to(device))
    scaler = torch.cuda.amp.GradScaler(enabled=is_cuda)

    # Instantiate model
    model = build_model(pretrained=True, stage=1)
    tot_params, train_params = count_parameters(model)
    print(f"Model Initialized: Stage 1 (Total Params: {tot_params:,} | Trainable: {train_params:,})")

    best_qwk = -1.0
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    history_csv_path = os.path.join(TRAINING_RUNS_DIR, "history.csv")
    history_json_path = os.path.join(TRAINING_RUNS_DIR, "history.json")

    # CSV Logging Header
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "stage", "train_loss", "train_acc", "val_loss",
            "val_acc", "macro_f1", "qwk", "lr"
        ])

    total_epochs = epochs_stage1 + epochs_stage2

    for stage, epochs, lr in [(1, epochs_stage1, lr_stage1), (2, epochs_stage2, lr_stage2)]:
        print(f"\n" + "-" * 70)
        print(f" >>> STARTING STAGE {stage} ({'Head Warmup' if stage == 1 else 'Fine-Tuning'}) - {epochs} Epochs | LR={lr}")
        print("-" * 70, flush=True)

        set_stage(model, stage=stage)
        tot_params, train_params = count_parameters(model)
        print(f"Stage {stage} Active (Trainable Parameters: {train_params:,} / {tot_params:,})", flush=True)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )

        stage_start_epoch = 1 if stage == 1 else (epochs_stage1 + 1)
        stage_end_epoch = epochs_stage1 if stage == 1 else total_epochs

        for epoch in range(stage_start_epoch, stage_end_epoch + 1):
            start_time = time.time()

            # Train & Validate
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, is_cuda)
            val_metrics = validate_epoch(model, val_loader, criterion, is_cuda)

            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_metrics['qwk'])

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch:02d}/{total_epochs:02d} [Stage {stage}] [{elapsed:.1f}s] - "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}% | "
                f"Macro F1: {val_metrics['macro_f1']:.4f} | QWK: {val_metrics['qwk']:.4f} | "
                f"LR: {current_lr:.6f}",
                flush=True
            )

            # Record metrics history
            epoch_log = {
                "epoch": epoch,
                "stage": stage,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 2),
                "val_loss": val_metrics['val_loss'],
                "val_acc": val_metrics['accuracy'],
                "macro_f1": val_metrics['macro_f1'],
                "qwk": val_metrics['qwk'],
                "lr": current_lr,
                "per_class": val_metrics['per_class'],
                "confusion_matrix": val_metrics['confusion_matrix'],
            }
            history.append(epoch_log)

            # Write to CSV
            with open(history_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, stage, round(train_loss, 4), round(train_acc, 2),
                    val_metrics['val_loss'], val_metrics['accuracy'],
                    val_metrics['macro_f1'], val_metrics['qwk'], current_lr
                ])

            # Write to JSON
            with open(history_json_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            # Checkpoint 1: Best QWK Model
            if val_metrics['qwk'] > best_qwk:
                best_qwk = val_metrics['qwk']
                patience_counter = 0
                checkpoint_data = {
                    "epoch": epoch,
                    "stage": stage,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "class_weights": class_weights.tolist(),
                    "seed": seed,
                    "val_metrics": val_metrics,
                }
                torch.save(checkpoint_data, BEST_QWK_MODEL_PATH)
                torch.save(model.state_dict(), MODEL_PATH)  # Serving checkpoint
                print(f"  --> Saved BEST QWK Checkpoint (QWK: {best_qwk:.4f}) to {BEST_QWK_MODEL_PATH}", flush=True)
            else:
                if stage == 2:
                    patience_counter += 1

            # Checkpoint 2: Best Loss Model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                torch.save(model.state_dict(), BEST_LOSS_MODEL_PATH)
                print(f"  --> Saved BEST Loss Checkpoint (Val Loss: {best_val_loss:.4f}) to {BEST_LOSS_MODEL_PATH}", flush=True)

            # Early Stopping Check (Stage 2)
            if stage == 2 and patience_counter >= early_stopping_patience:
                print(f"\n[!] Early Stopping triggered after {patience_counter} epochs without QWK improvement.", flush=True)
                break

    print("\n" + "=" * 70, flush=True)
    print(f"Training Completed! Best Validation QWK: {best_qwk:.4f} | Best Val Loss: {best_val_loss:.4f}", flush=True)
    print(f"Best QWK Model saved to: {BEST_QWK_MODEL_PATH}", flush=True)
    print(f"Serving Model saved to: {MODEL_PATH}", flush=True)
    print(f"Metrics History saved to: {history_json_path} and {history_csv_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    train_model(epochs_stage1=3, epochs_stage2=12, batch_size=BATCH_SIZE)
