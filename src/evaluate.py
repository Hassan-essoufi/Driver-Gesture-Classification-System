import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix




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


def evaluate_models(y_true, predictions_dict, class_names=None):
    """
    Evaluate multiple models.
    
    Args:
        y_true: True labels
        predictions_dict: Dict {model_name: predictions}
        class_names: List of class names
    """
    if class_names is None:
        class_names = [f'c{i}' for i in range(10)]
    
    # Compute metrics for each model
    metrics = {}
    for name, pred in predictions_dict.items():
        metrics[name] = compute_model_metrics(y_true, pred, name)
    
    # Display results
    print_results(metrics)
    
    # Generate visualizations
    figs = generate_visualizations(metrics, class_names, y_true, predictions_dict)
    
    return {'metrics': metrics, 'figures': figs}


# ==> VISUALIZATION FUNCTIONS <==

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



def plot_confusion_matrices(metrics_dict, model_names, class_names):
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



def plot_per_class_metrics(metrics_dict, model_names, class_names):
    """Display precision and recall per class for each model."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    x = np.arange(len(class_names))
    width = 0.8 / len(model_names)
    
    # Precision per class
    for i, name in enumerate(model_names):
        offset = (i - len(model_names)/2 + 0.5) * width
        axes[0].bar(x + offset, metrics_dict[name]['precision_per_class'], 
                   width, label=name, alpha=0.8)
    
    axes[0].set_title('Precision per Class', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1.05])
    
    # Recall per class
    for i, name in enumerate(model_names):
        offset = (i - len(model_names)/2 + 0.5) * width
        axes[1].bar(x + offset, metrics_dict[name]['recall_per_class'], 
                   width, label=name, alpha=0.8)
    
    axes[1].set_title('Recall per Class', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    return fig


def plot_error_distribution(y_true, predictions_dict, model_names):
    """Analyze error distribution between models."""
    # Compute correct predictions for each model
    correct_masks = {}
    for name in model_names:
        correct_masks[name] = (y_true == predictions_dict[name])
    
    # Count cases
    n_models = len(model_names)
    if n_models == 2:
        name1, name2 = model_names
        
        both_correct = correct_masks[name1] & correct_masks[name2]
        both_wrong = ~correct_masks[name1] & ~correct_masks[name2]
        only_first_correct = correct_masks[name1] & ~correct_masks[name2]
        only_second_correct = ~correct_masks[name1] & correct_masks[name2]
        
        labels = [f'{name1}\nonly', f'{name2}\nonly', 'Both\nwrong', 'Both\ncorrect']
        sizes = [only_first_correct.sum(), only_second_correct.sum(), 
                both_wrong.sum(), both_correct.sum()]
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#73AB84']
    else:
        # General case for n models
        labels = ['All correct', 'At least one error']
        all_correct = np.all([mask for mask in correct_masks.values()], axis=0)
        sizes = [all_correct.sum(), len(y_true) - all_correct.sum()]
        colors = ['#73AB84', '#F18F01']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                     startangle=90, explode=[0.1]*len(labels))
    
    ax.set_title('Prediction Distribution', fontweight='bold')
    return fig



# ==> UTILITIES <==

def print_results(metrics_dict):
    """Display clean comparative results table."""
    
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




def generate_visualizations(metrics_dict, class_names, y_true, predictions_dict):
    """Generate all visualizations."""
    model_names = list(metrics_dict.keys())
    
    figs = []
    figs.append(plot_metrics_comparison(metrics_dict, model_names, class_names))
    figs.append(plot_confusion_matrices(metrics_dict, model_names, class_names))
    figs.append(plot_per_class_metrics(metrics_dict, model_names, class_names))
    figs.append(plot_error_distribution(y_true, predictions_dict, model_names))
    
    return figs


