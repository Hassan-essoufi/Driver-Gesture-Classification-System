import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')



def evaluate_model_performance(y_true: np.ndarray,
                              efficient_pred: np.ndarray,
                              resnet_pred: np.ndarray,
                              efficient_probs: Optional[np.ndarray] = None,
                              resnet_probs: Optional[np.ndarray] = None) -> Dict:
    
    results = {}
    
    # Métriques pour EfficientNet
    results['efficientnet'] = {
        'accuracy': accuracy_score(y_true, efficient_pred),
        'precision': precision_score(y_true, efficient_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, efficient_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, efficient_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, efficient_pred),
        'precision_per_class': precision_score(y_true, efficient_pred, average=None, zero_division=0),
        'recall_per_class': recall_score(y_true, efficient_pred, average=None, zero_division=0)
    }
    
    # Métriques pour ResNet18
    results['resnet18'] = {
        'accuracy': accuracy_score(y_true, resnet_pred),
        'precision': precision_score(y_true, resnet_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, resnet_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, resnet_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, resnet_pred),
        'precision_per_class': precision_score(y_true, resnet_pred, average=None, zero_division=0),
        'recall_per_class': recall_score(y_true, resnet_pred, average=None, zero_division=0)
    }
    
    return results
def plot_model_comparison(metrics_dict: Dict,
                         class_names: List[str] = None) -> Tuple[plt.Figure, plt.Figure]:
    
    # 1. Barres comparatives des métriques
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    efficient_scores = [metrics_dict['efficientnet'][m] for m in metrics_to_plot]
    resnet_scores = [metrics_dict['resnet18'][m] for m in metrics_to_plot]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    ax1.bar(x - width/2, efficient_scores, width, label='EfficientNet', color='#2E86AB', alpha=0.8)
    ax1.bar(x + width/2, resnet_scores, width, label='ResNet18', color='#A23B72', alpha=0.8)
    
    # Améliorer l'affichage
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Comparaison EfficientNet vs ResNet18', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in metrics_to_plot])
    ax1.legend(loc='lower right')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs
    for i, (eff, res) in enumerate(zip(efficient_scores, resnet_scores)):
        ax1.text(i - width/2, eff + 0.01, f'{eff:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax1.text(i + width/2, res + 0.01, f'{res:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # 2. Matrices de confusion comparées
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(15, 6))
    
    # EfficientNet confusion matrix
    conf_efficient = metrics_dict['efficientnet']['confusion_matrix']
    sns.heatmap(conf_efficient, annot=True, fmt='d', cmap='Blues', ax=ax2,
                cbar_kws={'label': 'Nombre'})
    ax2.set_title('Matrice de Confusion - EfficientNet', fontweight='bold')
    ax2.set_xlabel('Prédiction')
    ax2.set_ylabel('Vérité Terrain')
    
    # ResNet18 confusion matrix
    conf_resnet = metrics_dict['resnet18']['confusion_matrix']
    sns.heatmap(conf_resnet, annot=True, fmt='d', cmap='Reds', ax=ax3,
                cbar_kws={'label': 'Nombre'})
    ax3.set_title('Matrice de Confusion - ResNet18', fontweight='bold')
    ax3.set_xlabel('Prédiction')
    ax3.set_ylabel('Vérité Terrain')
    
    plt.tight_layout()
    
    return fig1, fig2


def plot_per_class_comparison(metrics_dict: Dict,
                             class_names: List[str]) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Precision par classe
    x = np.arange(len(class_names))
    width = 0.35
    
    # Precision
    axes[0].bar(x - width/2, metrics_dict['efficientnet']['precision_per_class'], 
                width, label='EfficientNet', color='#2E86AB', alpha=0.8)
    axes[0].bar(x + width/2, metrics_dict['resnet18']['precision_per_class'], 
                width, label='ResNet18', color='#A23B72', alpha=0.8)
    
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_title('Precision par Classe', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1.05])
    
    # Recall par classe
    axes[1].bar(x - width/2, metrics_dict['efficientnet']['recall_per_class'], 
                width, label='EfficientNet', color='#2E86AB', alpha=0.8)
    axes[1].bar(x + width/2, metrics_dict['resnet18']['recall_per_class'], 
                width, label='ResNet18', color='#A23B72', alpha=0.8)
    
    axes[1].set_ylabel('Recall', fontsize=12)
    axes[1].set_title('Recall par Classe', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    return fig


def plot_error_analysis(y_true: np.ndarray,
                       efficient_pred: np.ndarray,
                       resnet_pred: np.ndarray,
                       class_names: List[str]) -> plt.Figure:
    
    # Identifier les erreurs uniques à chaque modèle
    efficient_correct = (y_true == efficient_pred)
    resnet_correct = (y_true == resnet_pred)
    
    # Cas où seul EfficientNet a raison
    only_efficient_correct = efficient_correct & ~resnet_correct
    
    # Cas où seul ResNet a raison
    only_resnet_correct = ~efficient_correct & resnet_correct
    
    # Les deux ont tort
    both_wrong = ~efficient_correct & ~resnet_correct
    
    # Les deux ont raison
    both_correct = efficient_correct & resnet_correct
    
    labels = ['EfficientNet\nseul correct', 'ResNet18\nseul correct', 
              'Les deux\nincorrects', 'Les deux\ncorrects']
    sizes = [only_efficient_correct.sum(), only_resnet_correct.sum(), 
             both_wrong.sum(), both_correct.sum()]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#73AB84']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Diagramme circulaire
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                       startangle=90, explode=(0.1, 0.1, 0.1, 0))
    ax1.set_title('Distribution des Erreurs', fontweight='bold')
    
    # Barres par classe pour les erreurs uniques
    if only_efficient_correct.sum() > 0 or only_resnet_correct.sum() > 0:
        eff_errors_by_class = []
        res_errors_by_class = []
        
        for i in range(len(class_names)):
            mask = (y_true == i)
            eff_errors_by_class.append((only_efficient_correct & mask).sum())
            res_errors_by_class.append((only_resnet_correct & mask).sum())
        
        x = np.arange(len(class_names))
        ax2.bar(x - 0.2, eff_errors_by_class, 0.4, label='EfficientNet seul correct', color='#2E86AB')
        ax2.bar(x + 0.2, res_errors_by_class, 0.4, label='ResNet18 seul correct', color='#A23B72')
        
        ax2.set_xlabel('Classe Réelle')
        ax2.set_ylabel('Nombre d\'échantillons')
        ax2.set_title('Erreurs Uniques par Classe', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def evaluate_both_models(y_true: np.ndarray,
                        efficient_pred: np.ndarray,
                        resnet_pred: np.ndarray,
                        class_names: List[str] = None) -> Dict:

    # Noms par défaut si non fournis
    if class_names is None:
        class_names = [f'c{i}' for i in range(10)]
    
    # 1. Calcul des métriques
    print("Calcul des métriques...")
    metrics = evaluate_model_performance(y_true, efficient_pred, resnet_pred)
    
    # 2. Affichage des scores
    print("\n" + "="*50)
    print(" RÉSULTATS - EFFICIENTNET vs RESNET18")
    print("="*50)
    
    print(f"\n{'Métrique':<15} {'EfficientNet':<15} {'ResNet18':<15} {'Différence':<12}")
    print("-" * 57)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        eff = metrics['efficientnet'][metric]
        res = metrics['resnet18'][metric]
        diff = eff - res
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        color = "\033[92m" if diff > 0 else "\033[91m" if diff < 0 else ""
        
        print(f"{metric:<15} {eff:<15.4f} {res:<15.4f} {color}{diff_str:<12}\033[0m")
    
    # 3. Déterminer le meilleur modèle
    eff_score = np.mean([metrics['efficientnet'][m] for m in ['accuracy', 'f1']])
    res_score = np.mean([metrics['resnet18'][m] for m in ['accuracy', 'f1']])
    
    
    if eff_score > res_score:
        print(f"EfficientNet est meilleur (score: {eff_score:.4f} vs {res_score:.4f})")
    elif res_score > eff_score:
        print(f"ResNet18 est meilleur (score: {res_score:.4f} vs {eff_score:.4f})")
    else:
        print("Les deux modèles sont équivalents")
    print("="*50)
    
    # 4. Générer les visualisations
    
    fig1, fig2 = plot_model_comparison(metrics, class_names)
    fig3 = plot_per_class_comparison(metrics, class_names)
    fig4 = plot_error_analysis(y_true, efficient_pred, resnet_pred, class_names)
    
    # 5. Résumé des résultats
    results = {
        'metrics': metrics,
        'best_model': 'efficientnet' if eff_score > res_score else 'resnet18',
        'figures': [fig1, fig2, fig3, fig4]
    }
    
    return results
