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
from tqdm.notebook import tqdm

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

    if class_weights is not None and class_weights != "null":
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

    if scheduler_name == "step_lr":
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

    elif scheduler_name == "reduce_on_plateau":
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


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Perform one training epoch.
    """

    model.train() 
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        optimizer.step()
        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Training epoch metrics
    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def validate_one_epoch(model, val_loader, criterion, device):
    """
    Perform one validation epoch.
    """

    model.eval() 
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(val_loader, desc="Validation", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Validation epoch metrics
    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

def save_checkpoint(state, config, is_best):
    """
    Save model checkpoints.
    Keep best model based on validation performance.
    """

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    last_ckpt_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    torch.save(state, last_ckpt_path)

    # Save best_model 
    if is_best:
        best_model_path = os.path.join(checkpoint_dir, "best_classifier.pth")
        torch.save(state, best_model_path)

        print(f"Best model updated and saved at: {best_model_path}")

    print(f"Checkpoint saved: {last_ckpt_path}")

def train_classifier(images_dir, model_name):
    """
    Full training pipeline
    """
    # Loading config
    config = load_training_config(
        training_cfg_path="config/training.yaml",
        model_cfg_path="config/model.yaml")
    
    # Device
    device = setup_environment(config)

    # Dataloaders
    loader_conf = config.get("dataloader", "")
    batch_size = loader_conf.get("batch_size", 2)
    num_workers = loader_conf.get("num_workers", 2)
    shuffle = loader_conf.get("shuffle", True)

    model_conf = config.get(model_name, "resnet50")
    input_size = model_conf.get("input_size", (224,224))

    train_transform = preprocess.get_train_transforms(input_size)
    val_transform = preprocess.get_val_transforms(input_size)

    train_loader = preprocess.create_dataloader(
        "data/annotations/train_labels.csv", images_dir,
        train_transform, batch_size,
        shuffle, num_workers
    )

    val_loader = preprocess.create_dataloader(
        "data/annotations/val_labels.csv", images_dir,
        val_transform, batch_size,
        shuffle, num_workers
    )

    # Model
    model = build_model(config, model_name)
    model.to(device)

    criterion = build_loss_function(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # Training
    best_val_acc = 0.0
    num_epochs = config["training"]["epochs"]

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device)

        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device)
        #  Scheduler
        if scheduler is not None:
            if config["scheduler"]["type"] == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        checkpoint_state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "config": config
        }

        save_checkpoint(
            state=checkpoint_state,
            is_best=is_best,
            config=config
        )

    print("\nTraining completed.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")    

def load_best_model(config, model_name, device):
    """
    Load the best trained model from checkpoints.
    """

    model = build_model(config, model_name)
    model.to(device)

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{model_name}_best.pth"
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Best model checkpoint not found: {checkpoint_path}"
        )

    # Loading checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Loading model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluation mode
    model.eval()

    print(f"Loaded best model from: {checkpoint_path}")

    return model







