"""Training functions for driver distraction detection"""

import os
import sys
from pathlib import Path
from datetime import datetime

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

def load_training_config(training_cfg_path, model_cfg_path):
    """
    Load and merge configuration files.
    """
    # Load YAML files
    with open(training_cfg_path, "r") as f:
        training_cfg = yaml.safe_load(f)

    with open(model_cfg_path, "r") as f:
        model_cfg = yaml.safe_load(f)

    # Merge configurations
    config = {
        **training_cfg,
        **model_cfg
    }

    return config

def setup_environment(config):
    """
    Setup training environment with reproducibility.
    Returns only the device.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds for reproducibility
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return device

def build_model(config, model_name):
    """
    Build a model based on the configuration.
    """
    # Get model configuration from the models section
    if 'models' in config and model_name in config['models']:
        model_cfg = config['models'][model_name]
    else:
        raise ValueError(f"Model '{model_name}' not found in configuration")
    
    # Extract model parameters
    num_classes = model_cfg.get("num_classes", 10)
    pretrained = model_cfg.get("pretrained", True)
    classifier_cfg = model_cfg.get("classifier", {})
    dropout_p = classifier_cfg.get("dropout", 0.5)
    hidden_dim = classifier_cfg.get("hidden_dim", None)
    fine_tuning_cfg = model_cfg.get("fine_tuning", {})
    freeze_backbone = fine_tuning_cfg.get("freeze_backbone", False)
    unfreeze_from_layer = fine_tuning_cfg.get("unfreeze_from_layer", None)
    use_bn = model_cfg.get("use_batch_norm", True)

    # Load backbone based on model name
    if model_name.lower() == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
    elif model_name.lower() == "efficientnet_b3":
        model = models.efficientnet_b3(pretrained=pretrained)
        in_features = model.classifier[1].in_features
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Apply fine-tuning settings
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    elif unfreeze_from_layer and model_name.lower() == "resnet50":
        unfreeze_flag = False
        for name, param in model.named_parameters():
            if unfreeze_from_layer in name:
                unfreeze_flag = True
            param.requires_grad = unfreeze_flag

    # Build classifier layers
    layers = []
    if hidden_dim:
        # With hidden layer
        layers.append(nn.Linear(in_features, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(hidden_dim, num_classes))
    else:
        # Direct classification
        layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(in_features, num_classes))

    # Replace the classifier
    if model_name.lower() == "resnet50":
        model.fc = nn.Sequential(*layers)
    elif model_name.lower() == "efficientnet_b3":
        model.classifier = nn.Sequential(*layers)

    return model

def build_loss_function(config):
    """
    Build loss function from configuration.
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
    Build optimizer from configuration.
    """
    optim_cfg = config.get("optimizer", {})
    
    optimizer_name = optim_cfg.get("name", "adamw").lower()
    lr = optim_cfg.get("lr", 1e-3)
    weight_decay = optim_cfg.get("weight_decay", 1e-4)

    # Filter parameters that require gradients
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
    Build learning rate scheduler from configuration.
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
    Train for one epoch.
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

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Calculate epoch metrics
    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy

def validate_one_epoch(model, val_loader, criterion, device):
    """
    Validate for one epoch.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Calculate epoch metrics
    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy

def save_checkpoint(state, config, is_best):
    """
    Save model checkpoint.
    """
    # Get model name from state
    model_name = state.get('model_name', 'model')
    
    # Get checkpoint directory
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save last checkpoint
    last_checkpoint_path = checkpoint_dir / f"{model_name}_last.pth"
    torch.save(state, last_checkpoint_path)
    
    # Save best checkpoint if it's the best model
    if is_best:
        best_checkpoint_path = checkpoint_dir / f"{model_name}_best.pth"
        torch.save(state, best_checkpoint_path)

def train_classifier(config, model_name, train_loader, val_loader, checkpoints_dir):
    """
    Main training function for a single model.
    """
    # Setup environment
    device = setup_environment(config)
    
    # Build model
    model = build_model(config, model_name)
    model.to(device)
    
    # Calculate model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Trainable percentage: {trainable_params/total_params*100:.1f}%")
    
    # Build loss function
    criterion = build_loss_function(config)
    
    # Build optimizer
    optimizer = build_optimizer(model, config)
    
    # Build scheduler
    scheduler = build_scheduler(optimizer, config)
    
    # Training parameters
    training_params = config.get('training', {})
    num_epochs = training_params.get('epochs', 30)
    early_stopping_patience = training_params.get('early_stopping_patience', 7)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training state
    best_val_acc = 0.0
    best_epoch = 0
    early_stopping_counter = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        # Training phase
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation phase
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Display metrics
        print(f"  Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Check for best model
        is_best = val_acc > best_val_acc
        
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            early_stopping_counter = 0
            print(f"  ✓ New best model! Validation accuracy: {val_acc:.4f}")
        else:
            early_stopping_counter += 1
            print(f"  Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
        
        # Create checkpoint
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'config': config,
            'model_name': model_name,
            'history': history
        }
        
        # Save checkpoint
        save_checkpoint(checkpoint_state, config, is_best)
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\nTraining completed!")
    print(f"  - Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"  - Total epochs trained: {len(history['train_loss'])}")
    
    return {
        'model': model,
        'history': history,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch
    }

def load_best_model(config, model_name, device):
    """
    Load the best trained model from checkpoints.
    """
    # Build model architecture
    model = build_model(config, model_name)
    model.to(device)
    
    # Get checkpoint path
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
    checkpoint_path = checkpoint_dir / f"{model_name}_best.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best model checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    print(f"Loaded best model from: {checkpoint_path}")
    print(f"  - Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
    print(f"  - Training epoch: {checkpoint['epoch']}")
    
    return model