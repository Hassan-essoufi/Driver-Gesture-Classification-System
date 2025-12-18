import os
import cv2
import pandas as pd
import numpy as np
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


#Image loading
def load_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert BGR (OpenCV default) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def resize_image(image, size=(224, 224)):
    """
    Resize image to target size.
    """
    if image is None:
        raise ValueError("Input image is None")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array")

    # width, height
    target_h, target_w = size[0] ,size[1]

    resized_image = cv2.resize(
        image,
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR
    )

    return resized_image
