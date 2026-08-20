"""
RetinaX Dataset Loader & Stratified Splitting Module.
Handles APTOS dataset metadata loading, stratified train/validation splitting,
batch-wise PyTorch DataLoader creation, and class imbalance strategies.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from backend.config import (
    APTOS_TRAIN_CSV,
    APTOS_TEST_CSV,
    APTOS_TRAIN_IMG_DIR,
    APTOS_TEST_IMG_DIR,
    VAL_SIZE,
    RANDOM_SEED,
    BATCH_SIZE,
    NUM_WORKERS,
)
from backend.preprocessing import set_seed, get_train_transforms, get_val_transforms

class APTOSDataset(Dataset):
    """
    PyTorch Dataset for APTOS 2019 fundus images.
    Loads single images dynamically on-demand from disk during iteration (never loads full dataset into RAM).
    """
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['id_code'])

        # Resolve image file extension (.png or .jpg)
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image '{img_id}' missing in directory: {self.img_dir}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, img_id

        label = int(row['diagnosis'])
        return image, torch.tensor(label, dtype=torch.long)


def compute_class_weights(df: pd.DataFrame, num_classes: int = 5) -> torch.Tensor:
    """
    Calculate inverse class frequency weights for NLLLoss / CrossEntropyLoss.
    Formula: w_c = N / (num_classes * N_c)
    """
    class_counts = df['diagnosis'].value_counts().to_dict()
    total_samples = len(df)
    
    weights = []
    for c in range(num_classes):
        count = class_counts.get(c, 0)
        if count > 0:
            w = total_samples / (num_classes * count)
        else:
            w = 1.0
        weights.append(w)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize weights so mean weight is 1.0
    weights_tensor = weights_tensor / weights_tensor.mean()
    return weights_tensor


def create_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler for mini-batch class-aware over-sampling during training.
    """
    class_counts = df['diagnosis'].value_counts().to_dict()
    class_weights = {c: 1.0 / count for c, count in class_counts.items()}
    sample_weights = [class_weights[int(label)] for label in df['diagnosis']]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def get_stratified_split(csv_path: str = APTOS_TRAIN_CSV, val_size: float = VAL_SIZE, seed: int = RANDOM_SEED):
    """
    Perform a reproducible, stratified train/validation split based on the diagnosis label.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df['diagnosis']
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_dataloaders(
    batch_size: int = BATCH_SIZE,
    val_size: float = VAL_SIZE,
    seed: int = RANDOM_SEED,
    use_sampler: bool = False,
    num_workers: int = NUM_WORKERS
):
    """
    Full pipeline builder for training & validation DataLoaders.
    - Sets reproducible seed
    - Performs stratified split
    - Applies stochastic augmentation to train set and deterministic preprocessing to val set
    - Computes class weights for loss weighting
    - Optionally builds WeightedRandomSampler
    """
    set_seed(seed)

    train_df, val_df = get_stratified_split(csv_path=APTOS_TRAIN_CSV, val_size=val_size, seed=seed)

    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    train_dataset = APTOSDataset(train_df, APTOS_TRAIN_IMG_DIR, transform=train_transforms)
    val_dataset = APTOSDataset(val_df, APTOS_TRAIN_IMG_DIR, transform=val_transforms)

    class_weights = compute_class_weights(train_df)

    if use_sampler:
        sampler = create_weighted_sampler(train_df)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, class_weights, train_df, val_df
