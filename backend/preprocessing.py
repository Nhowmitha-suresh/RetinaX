"""
RetinaX Image Preprocessing & Augmentation Pipeline Module.
Implements reproducible seeding, training data augmentation, and deterministic validation transforms.
"""

import random
import numpy as np
import torch
from torchvision import transforms

from backend.config import (
    RANDOM_SEED,
    IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD,
)

def set_seed(seed: int = RANDOM_SEED):
    """
    Set fixed random seed across all libraries to ensure full reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_train_transforms(image_size=IMAGE_SIZE, mean=NORM_MEAN, std=NORM_STD):
    """
    Returns stochastic data augmentation pipeline ONLY for the training dataset.
    Includes random flips, rotations, color jittering, to_tensor, and normalization.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_val_transforms(image_size=IMAGE_SIZE, mean=NORM_MEAN, std=NORM_STD):
    """
    Returns deterministic preprocessing pipeline ONLY for the validation/testing dataset.
    Applies only scaling, tensor conversion, and normalization. No random augmentations.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
