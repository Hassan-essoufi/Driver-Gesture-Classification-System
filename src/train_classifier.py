import yaml
import random
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler


def load_training_config(training_cfg_path, model_cfg_path):
    """
    Load configuration files.
    """
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
    Setup training environment.
    """
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Random seeds
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
    if 'models' in config and model_name in config['models']:
        model_cfg = config['models'][model_name]
    else:
        raise ValueError(f"'{model_name}' not found ")
    
    # Model's parameters
    num_classes = model_cfg.get("num_classes", 10)
    pretrained = model_cfg.get("pretrained", True)
    classifier_cfg = model_cfg.get("classifier", {})
    dropout_p = classifier_cfg.get("dropout", 0.5)
    hidden_dim = classifier_cfg.get("hidden_dim", None)
    ft_cfg = model_cfg.get("fine_tuning", {})
    freeze_backbone = ft_cfg.get("freeze_backbone", False)
    unfreeze_from_layer = ft_cfg.get("unfreeze_from_layer", None)
    use_bn = model_cfg.get("use_batch_norm", True)
    
    # Loading backbone
    if model_name.lower() == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
    elif model_name.lower() == "efficientnet_b3":
        model = models.efficientnet_b3(pretrained=pretrained)
        in_features = model.classifier[1].in_features
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Finetuning
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
        layers.append(nn.Linear(in_features, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(hidden_dim, num_classes))
    else:
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
    Build optimizer from configuration.
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
    Learning rate sheduler.
    """
    sched_cfg = config.get("scheduler", {})
    
    scheduler_name = sched_cfg.get("name", None)

    if scheduler_name is None:
        return None

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "step_lr":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 10),
            gamma=sched_cfg.get("gamma", 0.1)
        )
    elif scheduler_name == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("t_max", 20),
            eta_min=sched_cfg.get("eta_min", 1e-6)
        )
    elif scheduler_name == "reduce_on_plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
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
    Training for one epoch.
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

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy

def save_checkpoint(state, config, is_best):
    """
    Save model checkpoint.
    """
    model_name = state.get("model_name", None)
    
    # checkpoint directory
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if is_best:
        # Save the best checkpoint
        best_checkpoint_path = checkpoint_dir / f"{model_name}_best.pth"
        torch.save(state, best_checkpoint_path)
    else :
        # Save last checkpoint
        last_checkpoint_path = checkpoint_dir / f"{model_name}_last.pth"
        torch.save(state, last_checkpoint_path)



def train_classifier(config, model_name, train_loader, val_loader):
    """
    Main training function.
    """

    device = setup_environment(config)
    model = build_model(config, model_name)
    model.to(device)
    
    criterion = build_loss_function(config)
    
    # Build optimizer
    optimizer = build_optimizer(model, config)
    
    # Build scheduler
    scheduler = build_scheduler(optimizer, config)
    
    training_params = config.get('training', {})
    num_epochs = training_params.get('epochs', 15)
    patience = training_params.get('patience', 5)
    

    best_val_acc = 0.0
    best_epoch = 0
    counter = 0
        
    # Training loop
    for epoch in range(num_epochs):        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
              
        # Best model check
        is_best = val_acc > best_val_acc
        
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            counter = 0
        else:
            counter += 1

        # Create checkpoint
        checkpoint_state = {
            'model_name': model_name,
            'epoch': epoch +1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_acc': best_val_acc
        }
        
        # Early stopping
        if counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
            break
    
    print(f"\nTraining completed!")

    save_checkpoint(checkpoint_state, config, is_best)
    
    return {
        'model': model,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch
    }

def load_model(config, model_name, device):
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
    model.load_state_dict(checkpoint['model_state'])
    
    model.eval()
    
    print(f"Loaded best model from: {checkpoint_path}")
    print(f"  - Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
    print(f"  - Training epoch: {checkpoint['epoch']}")
    
    return model