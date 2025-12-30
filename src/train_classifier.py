import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau
)

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
    num_classes = model_cfg.get("num_classes", 10)
    pretrained = model_cfg.get("pretrained", True)
    classifier_cfg = model_cfg.get("classifier", {})
    dropout_p = classifier_cfg.get("dropout", 0.5)
    hidden_dim = classifier_cfg.get("hidden_dim", None)
    fine_tuning_cfg = model_cfg.get("fine_tuning", {})
    freeze_backbone = fine_tuning_cfg.get("freeze_backbone", False)
    unfreeze_from_layer = fine_tuning_cfg.get("unfreeze_from_layer", None)
    use_bn = model_cfg.get("use_batch_norm", True)

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
    if hidden_dim:  
        # Optional hidden FC layer
        layers.append(nn.Linear(in_features, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(hidden_dim, num_classes))
    else:  
        # map to num_classes
        layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(in_features, num_classes))

    if model_name.lower() == "resnet50":
        model.fc = nn.Sequential(*layers)
    elif model_name.lower() == "efficientnet_b3":
        model.classifier = nn.Sequential(*layers)

    return model

def build_loss_function(config):
    """
    CrossEntropyLoss
  
    """
    loss_cfg = config.get("loss", {})
    
    loss_name = loss_cfg.get("name", "cross_entropy")
    label_smoothing = loss_cfg.get("label_smoothing", 0.0)
    class_weights = loss_cfg.get("class_weights", None)

    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float)

    if loss_name == "cross_entropy":
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

    return criterion


def build_optimizer(model, config):
    """
    Optimizer (Adam / AdamW).
    """
    optim_cfg = config.get("optimizer", {})
    
    optimizer_name = optim_cfg.get("name", "adamw").lower()
    lr = optim_cfg.get("lr", 1e-3)
    weight_decay = optim_cfg.get("weight_decay", 1e-4)

    params = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_name == "adam":
        optimizer = optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay
        )

    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay
        )

    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer



def build_scheduler(optimizer, config):
    """
    Learning rate scheduler.
    """
    sched_cfg = config.get("scheduler", {})
    
    scheduler_name = sched_cfg.get("name", None)

    if scheduler_name is None:
        return None

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "step":
        scheduler = StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 10),
            gamma=sched_cfg.get("gamma", 0.1)
        )

    elif scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("t_max", 20),
            eta_min=sched_cfg.get("eta_min", 1e-6)
        )

    elif scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=sched_cfg.get("factor", 0.1),
            patience=sched_cfg.get("patience", 5),
            min_lr=sched_cfg.get("min_lr", 1e-6)
        )

    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return scheduler

