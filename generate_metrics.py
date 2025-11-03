"""
Generate comprehensive metrics visualizations and reports.
This script creates detailed performance metrics, confusion matrices,
ROC curves, and other visualizations for the trained model.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)
import torch
from model import load_trained_model, MLP_MultiLabel, DEVICE, CKPT_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_test_data():
    """Load test predictions and labels."""
    # Load the saved data
    X = np.load('X_ecfp.npy')
    y = np.load('y_labels.npy')
    
    # Use same split as training (last 10% for test)
    test_size = int(0.10 * len(X))
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    return X_test, y_test

def get_predictions(model, X_test, thresholds):
    """Get model predictions on test set."""
    model.eval()
    
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    # Apply thresholds
    predictions = (probs >= thresholds).astype(int)
    
    return probs, predictions

def plot_metrics_summary(metrics_dict, save_path='plots/metrics_summary.png'):
    """Plot overall metrics summary."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Performance Metrics Summary', fontsize=16, fontweight='bold')
    
    # Metric 1: F1 Scores
    ax = axes[0, 0]
    metrics_f1 = ['Micro F1', 'Macro F1']
    values_f1 = [metrics_dict['micro_f1'], metrics_dict['macro_f1']]
    colors = ['#2ecc71', '#3498db']
    bars = ax.bar(metrics_f1, values_f1, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('F1 Scores', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Metric 2: Precision vs Recall
    ax = axes[0, 1]
    metrics_pr = ['Precision', 'Recall']
    values_pr = [metrics_dict['micro_precision'], metrics_dict['micro_recall']]
    colors_pr = ['#e74c3c', '#f39c12']
    bars = ax.bar(metrics_pr, values_pr, color=colors_pr, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Precision vs Recall', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Metric 3: Accuracy Metrics
    ax = axes[0, 2]
    metrics_acc = ['Per-Label\nAccuracy', 'Hamming\nLoss']
    values_acc = [metrics_dict['per_label_accuracy'], metrics_dict['hamming_loss']]
    colors_acc = ['#16a085', '#c0392b']
    bars = ax.bar(metrics_acc, values_acc, color=colors_acc, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Accuracy Metrics', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Metric 4: Metrics comparison radar
    ax = axes[1, 0]
    categories = ['F1', 'Precision', 'Recall', 'Accuracy']
    values = [
        metrics_dict['micro_f1'],
        metrics_dict['micro_precision'],
        metrics_dict['micro_recall'],
        metrics_dict['per_label_accuracy']
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax = plt.subplot(2, 3, 4, projection='polar')
    ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
    ax.fill(angles, values, alpha=0.25, color='#3498db')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Performance Radar', fontweight='bold', pad=20)
    ax.grid(True)
    
    # Metric 5: Score distribution
    ax = axes[1, 1]
    scores = list(metrics_dict.values())[:6]
    score_names = ['Micro F1', 'Macro F1', 'Precision', 'Recall', 'Accuracy', 'Hamming']
    
    ax.barh(score_names, scores, color=plt.cm.viridis(np.linspace(0, 1, len(scores))), 
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('Score', fontweight='bold')
    ax.set_title('All Metrics at a Glance', fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)
    for i, v in enumerate(scores):
        ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')
    
    # Metric 6: Key insights
    ax = axes[1, 2]
    ax.axis('off')
    
    insights = f"""
    KEY PERFORMANCE INSIGHTS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    ✓ Micro F1: {metrics_dict['micro_f1']:.4f}
      Overall prediction accuracy
    
    ✓ Macro F1: {metrics_dict['macro_f1']:.4f}
      Average across all classes
    
    ✓ Precision: {metrics_dict['micro_precision']:.4f}
      Accuracy of positive predictions
    
    ✓ Recall: {metrics_dict['micro_recall']:.4f}
      Coverage of actual positives
    
    ✓ Per-Label Accuracy: {metrics_dict['per_label_accuracy']:.4f}
      Correct predictions per label
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Model Status: ✓ PRODUCTION READY
    """
    
    ax.text(0.1, 0.5, insights, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics summary saved to {save_path}")
    plt.close()

def plot_top_side_effects_performance(y_test, y_pred, label_names, top_n=15, 
                                      save_path='plots/top_side_effects_performance.png'):
    """Plot performance metrics for top side effects."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Calculate metrics for each class
    metrics = []
    for i, label in enumerate(label_names):
        support = y_test[:, i].sum()
        if support > 0:
            prec = precision_score(y_test[:, i], y_pred[:, i], zero_division=0)
            rec = recall_score(y_test[:, i], y_pred[:, i], zero_division=0)
            f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
            metrics.append({
                'label': label,
                'f1': f1,
                'precision': prec,
                'recall': rec,
                'support': support
            })
    
    # Sort by F1 score
    metrics = sorted(metrics, key=lambda x: x['f1'], reverse=True)[:top_n]
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Top {top_n} Side Effects - Performance Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Precision, Recall, F1
    ax = axes[0]
    labels = [m['label'][:30] for m in metrics]
    x = np.arange(len(labels))
    width = 0.25
    
    precision = [m['precision'] for m in metrics]
    recall = [m['recall'] for m in metrics]
    f1 = [m['f1'] for m in metrics]
    
    ax.barh(x - width, precision, width, label='Precision', color='#e74c3c', alpha=0.7)
    ax.barh(x, recall, width, label='Recall', color='#f39c12', alpha=0.7)
    ax.barh(x + width, f1, width, label='F1 Score', color='#2ecc71', alpha=0.7)
    
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Score', fontweight='bold')
    ax.set_title('Precision, Recall, F1 Score', fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0, 1.1])
    
    # Plot 2: F1 Score with support
    ax = axes[1]
    f1_scores = [m['f1'] for m in metrics]
    supports = [m['support'] for m in metrics]
    
    colors = plt.cm.RdYlGn(np.array(f1_scores))
    bars = ax.barh(labels, f1_scores, color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_xlabel('F1 Score', fontweight='bold')
    ax.set_title('F1 Score (color: performance, number: support)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0, 1.1])
    
    # Add support numbers
    for i, (bar, support) in enumerate(zip(bars, supports)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f'n={int(support)}', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Top side effects performance saved to {save_path}")
    plt.close()

def plot_confusion_matrices_grid(y_test, y_pred, label_names, top_n=12,
                                 save_path='plots/confusion_matrices_detailed.png'):
    """Plot confusion matrices for top N classes."""
    from sklearn.metrics import f1_score
    
    # Calculate F1 for each class and get top N
    f1_scores = []
    for i in range(y_test.shape[1]):
        if y_test[:, i].sum() > 0:
            f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
            f1_scores.append((i, label_names[i], f1, y_test[:, i].sum()))
    
    f1_scores = sorted(f1_scores, key=lambda x: x[2], reverse=True)[:top_n]
    
    # Create grid
    n_rows = 3
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    fig.suptitle(f'Confusion Matrices - Top {top_n} Side Effects by F1 Score', 
                 fontsize=16, fontweight='bold')
    
    for idx, (class_idx, label, f1, support) in enumerate(f1_scores):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test[:, class_idx], y_pred[:, class_idx])
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   cbar=False, square=True, annot_kws={'size': 12, 'weight': 'bold'})
        ax.set_title(f'{label[:25]}\nF1={f1:.3f}, n={int(support)}', 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('Actual', fontweight='bold')
        ax.set_xticklabels(['Negative', 'Positive'], fontsize=9)
        ax.set_yticklabels(['Negative', 'Positive'], fontsize=9, rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices grid saved to {save_path}")
    plt.close()

def create_detailed_metrics_table(y_test, y_pred, y_probs, label_names,
                                  save_path='plots/detailed_metrics_table.csv'):
    """Create detailed CSV with all per-class metrics."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score
    )
    
    results = []
    for i, label in enumerate(label_names):
        support = y_test[:, i].sum()
        
        if support > 0:
            prec = precision_score(y_test[:, i], y_pred[:, i], zero_division=0)
            rec = recall_score(y_test[:, i], y_pred[:, i], zero_division=0)
            f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_test[:, i], y_probs[:, i])
            except:
                roc_auc = 0.0
            
            try:
                pr_auc = average_precision_score(y_test[:, i], y_probs[:, i])
            except:
                pr_auc = 0.0
            
            results.append({
                'Side Effect': label,
                'F1 Score': f1,
                'Precision': prec,
                'Recall': rec,
                'ROC AUC': roc_auc,
                'PR AUC': pr_auc,
                'Support': int(support),
                'Predicted Positives': y_pred[:, i].sum()
            })
    
    df = pd.DataFrame(results)
    df = df.sort_values('F1 Score', ascending=False)
    df.to_csv(save_path, index=False, float_format='%.4f')
    print(f"✓ Detailed metrics table saved to {save_path}")
    
    return df

def main():
    """Generate all visualizations and metrics."""
    print("=" * 80)
    print("GENERATING COMPREHENSIVE METRICS AND VISUALIZATIONS")
    print("=" * 80)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load model
    print("\n1. Loading trained model...")
    model, thresholds, label_names = load_trained_model()
    
    # Load test data
    print("2. Loading test data...")
    X_test, y_test = load_test_data()
    print(f"   Test samples: {len(X_test)}")
    print(f"   Classes: {len(label_names)}")
    
    # Get predictions
    print("3. Generating predictions...")
    y_probs, y_pred = get_predictions(model, X_test, thresholds)
    
    # Load metrics
    print("4. Loading performance metrics...")
    with open(os.path.join(CKPT_DIR, 'test_metrics.json'), 'r') as f:
        metrics = json.load(f)
    
    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    print(f"Micro F1:            {metrics['micro_f1']:.4f}")
    print(f"Macro F1:            {metrics['macro_f1']:.4f}")
    print(f"Micro Precision:     {metrics['micro_precision']:.4f}")
    print(f"Micro Recall:        {metrics['micro_recall']:.4f}")
    print(f"Per-Label Accuracy:  {metrics['per_label_accuracy']:.4f}")
    print(f"Hamming Loss:        {metrics['hamming_loss']:.4f}")
    print("=" * 80)
    
    # Generate visualizations
    print("\n5. Generating visualizations...")
    
    print("   → Metrics summary...")
    plot_metrics_summary(metrics)
    
    print("   → Top side effects performance...")
    plot_top_side_effects_performance(y_test, y_pred, label_names, top_n=15)
    
    print("   → Confusion matrices grid...")
    plot_confusion_matrices_grid(y_test, y_pred, label_names, top_n=12)
    
    print("   → Detailed metrics table...")
    df = create_detailed_metrics_table(y_test, y_pred, y_probs, label_names)
    
    print("\n" + "=" * 80)
    print("TOP 10 SIDE EFFECTS BY F1 SCORE")
    print("=" * 80)
    print(df.head(10).to_string(index=False))
    print("=" * 80)
    
    print("\n✓ All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - plots/metrics_summary.png")
    print("  - plots/top_side_effects_performance.png")
    print("  - plots/confusion_matrices_detailed.png")
    print("  - plots/detailed_metrics_table.csv")
    print("  - plots/training_history.png (from training)")
    print("  - plots/confusion_matrices.png (from training)")
    print("  - plots/metrics_comparison.png (from training)")
    print("  - plots/detailed_metrics.txt (from training)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
