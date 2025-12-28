import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

import pandas as pd 
import numpy as np
import random
import yaml
import preprocess


def load_training_config(
    training_cfg_path="config/training.yaml",
    model_cfg_path="config/model.yaml"
):
    """
    Load training and model configuration files
    and merge them into a single config dictionary.
    """

    training_cfg_path = Path(training_cfg_path)
    model_cfg_path = Path(model_cfg_path)

    if not training_cfg_path.exists():
        raise FileNotFoundError(f"Training config not found: {training_cfg_path}")

    if not model_cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_cfg_path}")

    # Load YAML files
    with open(training_cfg_path, "r") as f:
        training_cfg = yaml.safe_load(f)

    with open(model_cfg_path, "r") as f:
        model_cfg = yaml.safe_load(f)

    # Merge configs
    config = {
        **training_cfg,
        **model_cfg
    }

    return config

def setup_environment(config):
    """
    Setup training environment:
    - device (CPU or GPU)
    - reproducibility
    - output directories
    """

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Reproducibility
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Random seed set to {seed}")

    # Output directories
    output_root = Path(config.get("output_dir", "results"))
    checkpoints_dir = output_root / "checkpoints"
    logs_dir = output_root / "training_logs"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Checkpoints directory: {checkpoints_dir}")
    print(f"[INFO] Logs directory: {logs_dir}")

    # Save paths in config for later use
    config["checkpoints_dir"] = checkpoints_dir
    config["logs_dir"] = logs_dir

    return device

def build_model(config, model_name):
    """
    Build a model based on the configuration.
    
    Args:
        config (dict)

    Returns:
        PyTorch model
    """
    model_cfg = config.get(model_name, "resnet50")
    num_classes = config.get("num_classes", 10)
    pretrained = config.get("pretrained", True)
    classifier_cfg = config.get("classifier", {})
    dropout_p = classifier_cfg.get("dropout", 0.5)
    hidden_dim = classifier_cfg.get("hidden_dim", None)
    fine_tuning_cfg = config.get("fine_tuning", {})
    freeze_backbone = fine_tuning_cfg.get("freeze_backbone", False)
    unfreeze_from_layer = fine_tuning_cfg.get("unfreeze_from_layer", None)
    use_bn = config.get("use_batch_norm", True)

    # Load backbone
    if model_name.lower() == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features  
    elif model_name.lower() == "efficientnet_b3":
        model = models.efficientnet_b3(pretrained=pretrained)
        in_features = model.classifier[1].in_features  
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Fine-tuning 
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    elif unfreeze_from_layer and model_name.lower() == "resnet50":
        unfreeze_flag = False
        for name, param in model.named_parameters():
            if unfreeze_from_layer in name:
                unfreeze_flag = True
            param.requires_grad = unfreeze_flag

    # Replace classifier head
    layers = []
    if hidden_dim:  # Optional hidden FC layer
        layers.append(nn.Linear(in_features, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(hidden_dim, num_classes))
    else:  # Directly map to num_classes
        layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(in_features, num_classes))

    if model_name.lower() == "resnet50":
        model.fc = nn.Sequential(*layers)
    elif model_name.lower() == "efficientnet_b3":
        model.classifier = nn.Sequential(*layers)

    return model

