import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import os
from datetime import datetime
import io
from contextlib import redirect_stdout

def compute_model_metrics(y_true, y_pred, model_name):
    """Compute metrics for a single model."""
    return {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'confusion': confusion_matrix(y_true, y_pred),
        'precision_per_class': precision_score(y_true, y_pred, average=None, zero_division=0),
        'recall_per_class': recall_score(y_true, y_pred, average=None, zero_division=0)
    }


def print_results(metrics_dict):
    """Display comparative results table."""
    model_names = list(metrics_dict.keys())
    
    print(f"\n{'Metric':<15}", end="")
    for name in model_names:
        print(f"{name:<15}", end="")
    print(f"{'Best':<10}")
    print("-" * (15 + 15*len(model_names) + 10))
    
    for metric_key, metric_name in [('accuracy', 'Accuracy'), 
                                   ('precision', 'Precision'),
                                   ('recall', 'Recall'), 
                                   ('f1', 'F1-Score')]:
        print(f"{metric_name:<15}", end="")
        
        values = []
        for name in model_names:
            value = metrics_dict[name][metric_key]
            values.append(value)
            print(f"{value:.4f}{'':<11}", end="")
        
        best_idx = np.argmax(values)
        best_name = model_names[best_idx]
        print(f"{best_name:<10}")
    
    # Determine overall best model
    avg_scores = {}
    for name in model_names:
        scores = [metrics_dict[name][k] for k in ['accuracy', 'precision', 'recall', 'f1']]
        avg_scores[name] = np.mean(scores)
    
    best_model = max(avg_scores, key=avg_scores.get)
    print(f"\nBest model: {best_model} (average score: {avg_scores[best_model]:.4f})")


def get_best_model(metrics_dict):
    """Return the best model based on average score."""
    avg_scores = {}
    for name, metrics in metrics_dict.items():
        avg_scores[name] = np.mean([
            metrics['accuracy'],
            metrics['precision'], 
            metrics['recall'],
            metrics['f1']
        ])
    
    return max(avg_scores, key=avg_scores.get)


def plot_metrics_comparison(metrics_dict, model_names, class_names):
    """Compare global metrics across models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metric_titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#2E86AB', '#A23B72', '#73AB84', '#F18F01']
    
    for idx, (ax, title, key, color) in enumerate(zip(axes.flat, metric_titles, metric_keys, colors)):
        values = [metrics_dict[name][key] for name in model_names]
        
        bars = ax.bar(model_names, values, color=color, alpha=0.8)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{height:.3f}', ha='center', fontsize=9)
    
    plt.suptitle('Global Metrics Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_confusion_matrix(metrics_dict, model_names, class_names):
    """Display confusion matrices side by side."""
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, name in zip(axes, model_names):
        conf_matrix = metrics_dict[name]['confusion']
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar_kws={'label': 'Count'})
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Prediction')
        ax.set_ylabel('True Label')
    
    plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_acc_class_metrics(metrics_dict, model_names, class_names):
    """Display accuracy per class for each model."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(class_names))
    width = 0.8 / len(model_names)
    
    # Calculate accuracy per class
    for i, name in enumerate(model_names):
        conf_matrix = metrics_dict[name]['confusion']
        accuracy_per_class = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, accuracy_per_class, 
                     width, label=name, alpha=0.8)
        
        # Add values on bars
        for bar_idx, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{height:.2f}', ha='center', fontsize=8)
    
    ax.set_title('Accuracy per Class', fontweight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    return fig


def plot_train_val_test_acc(train_acc, val_acc, test_acc, model_names):
    """Plot train, validation, and test accuracy for each model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data - ensure all models have the same keys
    metrics_data = {
        'Train': [train_acc.get(name, 0) for name in model_names],
        'Validation': [val_acc.get(name, 0) for name in model_names],
        'Test': [test_acc.get(name, 0) for name in model_names]
    }
    
    # Bar chart
    x = np.arange(len(model_names))
    width = 0.25
    
    for i, (phase, values) in enumerate(metrics_data.items()):
        offset = (i - 1) * width
        bars = axes[0].bar(x + offset, values, width, label=phase, alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.3f}', ha='center', fontsize=8)
    
    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy by Phase', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1.1])
    
    # Line chart comparison
    for i, name in enumerate(model_names):
        values = [
            train_acc.get(name, 0),
            val_acc.get(name, 0),
            test_acc.get(name, 0)
        ]
        axes[1].plot(['Train', 'Validation', 'Test'], values, 
                    marker='o', label=name, linewidth=2)
    
    axes[1].set_xlabel('Phase')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Progression', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.1])
    
    plt.suptitle('Model Performance Across Phases', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def generate_visualizations(metrics_dict, class_names, train_acc=None, 
                          val_acc=None, test_acc=None):
    """Generate all visualizations."""
    model_names = list(metrics_dict.keys())
    
    figs = []
    figs.append(plot_metrics_comparison(metrics_dict, model_names, class_names))
    figs.append(plot_confusion_matrices(metrics_dict, model_names, class_names))
    figs.append(plot_acc_class_metrics(metrics_dict, model_names, class_names))
    
    # Only add phase comparison if all accuracies are provided
    if train_acc and val_acc and test_acc:
        figs.append(plot_train_val_test_acc(train_acc, val_acc, test_acc, model_names))
    
    return figs


def save_results_simple(metrics, figures, results_dir="results"):
    """Save visualizations and metrics summary."""
    # Create directories
    vis_dir = os.path.join(results_dir, "visualizations")
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths = {}
    
    # 1. Save visualizations
    if figures:
        vis_names = [
            "global_metrics_comparison.png",
            "confusion_matrices.png", 
            "accuracy_per_class.png",
            "phase_comparison.png"
        ]
        
        for i, (fig, name) in enumerate(zip(figures, vis_names)):
            if i < len(figures):
                filepath = os.path.join(vis_dir, f"{timestamp}_{name}")
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                saved_paths[name] = filepath
    
    # 2. Save metrics summary
    metrics_file = os.path.join(metrics_dir, f"{timestamp}_metrics_summary.json")
    
    summary = {
        "timestamp": timestamp,
        "models": {},
        "best_model": get_best_model(metrics)
    }
    
    for model_name, model_metrics in metrics.items():
        summary["models"][model_name] = {
            "accuracy": float(model_metrics['accuracy']),
            "precision": float(model_metrics['precision']),
            "recall": float(model_metrics['recall']),
            "f1_score": float(model_metrics['f1']),
            "confusion_matrix": model_metrics['confusion'].tolist()
        }
    
    with open(metrics_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    saved_paths['metrics_summary'] = metrics_file
    
    # 3. Save results table
    txt_file = os.path.join(metrics_dir, f"{timestamp}_results_table.txt")
    
    f = io.StringIO()
    with redirect_stdout(f):
        print_results(metrics)
    output = f.getvalue()
    
    with open(txt_file, 'w') as f:
        f.write(output)
    
    saved_paths['results_table'] = txt_file
    
    # Print summary
    print(f"\n📁 Results saved to: {results_dir}/")
    print(f"   • {len(figures)} visualizations")
    print(f"   • 2 metric files")
    
    return saved_paths


def evaluate_models(y_true, predictions_dict, class_names=None, 
                   train_acc=None, val_acc=None, test_acc=None,
                   save_results=True, results_dir="results"):
    """
    Evaluate multiple models with phase comparison.
    
    Args:
        y_true: True labels for test set
        predictions_dict: Dict {model_name: test_predictions}
        class_names: List of class names
        train_acc: Dict {model_name: train_accuracy}
        val_acc: Dict {model_name: validation_accuracy}
        test_acc: Dict {model_name: test_accuracy}
        save_results: Whether to save results
        results_dir: Directory to save results
    """
    if class_names is None:
        class_names = [f'c{i}' for i in range(10)]
    
    # Compute test metrics
    metrics = {}
    for name, pred in predictions_dict.items():
        metrics[name] = compute_model_metrics(y_true, pred, name)
    
    # Display results
    print_results(metrics)
    
    # Generate visualizations
    figs = generate_visualizations(metrics, class_names, train_acc, val_acc, test_acc)
    
    # Save results
    if save_results:
        saved_paths = save_results_simple(metrics, figs, results_dir)
        return {'metrics': metrics, 'figures': figs, 'saved_paths': saved_paths}
    
    return {'metrics': metrics, 'figures': figs}