# preprocess.py - Corrected and Complete Version
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_image(image_path):
    """Loads an image with error checking"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    
    except Exception as e:
        raise ValueError(f"Error loading {image_path}: {e}")

def get_train_transforms(input_size=(224, 224)):
    """Training transformations"""
     
    return transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

def get_val_transforms(input_size=(224, 224)):
    """Validation transformations"""

    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_test_transforms(input_size=(224, 224)):
    """Test transformations"""

    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def denormalize_tensor(tensor):
    """Denormalizes a tensor for visualization"""

    if isinstance(tensor, torch.Tensor):
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(tensor.device)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(tensor.device)
        return tensor * std + mean
    else:
        raise TypeError(f"Unsupported type: {type(tensor)}")

class DriverDistractionDataset(Dataset):

    def __init__(self, csv_file, images_dir, transform=None):
        """
        Args : 
            images_dir: Path to images (class-based structure)
            csv_file: CSV file with columns [filename, label]
            transform: Transformations to apply
          
        """
        if csv_file and os.path.exists(csv_file):
            self.data = pd.read_csv(csv_file)
        else:
            raise FileNotFoundError("File not found")
        
        if images_dir and os.path.exists(images_dir):
            self.images_dir = images_dir
        else:
            raise FileNotFoundError('Diroctory not found')

        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.images_dir, row['image_path'])
        label = int(row["label_id"])
        image = load_image(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label    

def create_dataloader(
        csv_file, images_dir,
        transform, batch_size,
        shuffle, num_workers):
    dataset = DriverDistractionDataset(
        csv_file=csv_file,
        images_dir=images_dir,
        transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader

    