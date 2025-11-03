"""
Script to view training results, metrics, and visualizations.

Usage:
    python view_results.py              # Show all results
    python view_results.py --metrics    # Show metrics only
    python view_results.py --plots      # Open plot files
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def show_metrics():
    """Display test metrics."""
    metrics_path = "checkpoints/test_metrics.json"
    
    if not os.path.exists(metrics_path):
        print(f"❌ Metrics file not found: {metrics_path}")
        print("   Run training first: python model.py")
        return False
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    print_section("TEST SET PERFORMANCE METRICS")
    print(f"\n{'Metric':<25} {'Value':<15}")
    print("-" * 80)
    print(f"{'Micro F1 Score':<25} {metrics['micro_f1']:<15.4f}")
    print(f"{'Macro F1 Score':<25} {metrics['macro_f1']:<15.4f}")
    print(f"{'Micro Precision':<25} {metrics['micro_precision']:<15.4f}")
    print(f"{'Micro Recall':<25} {metrics['micro_recall']:<15.4f}")
    print(f"{'Accuracy':<25} {metrics['accuracy']:<15.4f}")
    print(f"{'Hamming Loss':<25} {metrics['hamming_loss']:<15.4f}")
    print("-" * 80)
    
    return True


def show_training_history():
    """Display training history summary."""
    history_path = "checkpoints/training_history.json"
    
    if not os.path.exists(history_path):
        print(f"❌ Training history not found: {history_path}")
        return False
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    print_section("TRAINING HISTORY SUMMARY")
    
    n_epochs = len(history['train_loss'])
    best_val_f1_idx = history['val_f1'].index(max(history['val_f1']))
    
    print(f"\nTotal Epochs: {n_epochs}")
    print(f"Best Epoch: {best_val_f1_idx + 1}")
    print(f"\nBest Validation Metrics (Epoch {best_val_f1_idx + 1}):")
    print(f"  - F1 Score: {history['val_f1'][best_val_f1_idx]:.4f}")
    print(f"  - Accuracy: {history['val_accuracy'][best_val_f1_idx]:.4f}")
    print(f"  - Hamming Loss: {history['val_hamming'][best_val_f1_idx]:.4f}")
    
    print(f"\nFinal Epoch Metrics (Epoch {n_epochs}):")
    print(f"  - Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  - Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"  - Val F1: {history['val_f1'][-1]:.4f}")
    print(f"  - Val Accuracy: {history['val_accuracy'][-1]:.4f}")
    
    return True


def show_detailed_report():
    """Display detailed classification report."""
    report_path = "plots/detailed_metrics.txt"
    
    if not os.path.exists(report_path):
        print(f"❌ Detailed report not found: {report_path}")
        return False
    
    print_section("DETAILED CLASSIFICATION REPORT")
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    print(content)
    return True


def list_plot_files():
    """List available plot files."""
    plots_dir = "plots"
    
    if not os.path.exists(plots_dir):
        print(f"❌ Plots directory not found: {plots_dir}")
        return []
    
    plot_files = list(Path(plots_dir).glob("*.png"))
    
    if not plot_files:
        print(f"❌ No plot files found in {plots_dir}")
        return []
    
    print_section("AVAILABLE VISUALIZATIONS")
    print()
    for idx, plot_file in enumerate(plot_files, 1):
        print(f"{idx}. {plot_file.name}")
    print()
    
    return plot_files


def open_plot(plot_path):
    """Open a plot file with default viewer."""
    if sys.platform == 'win32':
        os.startfile(plot_path)
    elif sys.platform == 'darwin':  # macOS
        subprocess.run(['open', plot_path])
    else:  # Linux
        subprocess.run(['xdg-open', plot_path])


def show_model_info():
    """Display model configuration."""
    bundle_path = "checkpoints/inference_bundle.joblib"
    
    if not os.path.exists(bundle_path):
        print(f"❌ Model bundle not found: {bundle_path}")
        return False
    
    import joblib
    bundle = joblib.load(bundle_path)
    
    print_section("MODEL CONFIGURATION")
    
    config = bundle['model_config']
    print(f"\nArchitecture:")
    print(f"  - Input Dimension: {config['input_dim']}")
    print(f"  - Hidden Layers: {config['hidden_dims']}")
    print(f"  - Output Dimension: {config['output_dim']}")
    print(f"  - Dropout: {config['dropout']}")
    
    print(f"\nDataset:")
    print(f"  - Number of Side Effects: {len(bundle['label_names'])}")
    print(f"  - Threshold Strategy: Per-class optimized")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="View training results and metrics"
    )
    
    parser.add_argument(
        '--metrics',
        action='store_true',
        help='Show test metrics only'
    )
    
    parser.add_argument(
        '--history',
        action='store_true',
        help='Show training history only'
    )
    
    parser.add_argument(
        '--plots',
        action='store_true',
        help='List and open plot files'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Show detailed classification report'
    )
    
    parser.add_argument(
        '--model',
        action='store_true',
        help='Show model configuration'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Show everything'
    )
    
    args = parser.parse_args()
    
    # If no specific flag, show all
    show_all = args.all or not any([args.metrics, args.history, args.plots, args.report, args.model])
    
    success = True
    
    if show_all or args.model:
        success &= show_model_info()
    
    if show_all or args.metrics:
        success &= show_metrics()
    
    if show_all or args.history:
        success &= show_training_history()
    
    if show_all or args.plots:
        plot_files = list_plot_files()
        if plot_files:
            response = input("\nOpen plots? (y/n): ")
            if response.lower() == 'y':
                for plot_file in plot_files:
                    print(f"Opening {plot_file.name}...")
                    open_plot(str(plot_file))
    
    if args.report:
        success &= show_detailed_report()
    
    if not success:
        print("\n⚠️  Some results are missing. Have you run training yet?")
        print("   Run: python model.py")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("For more details, check:")
    print("  - plots/training_history.png")
    print("  - plots/confusion_matrices.png")
    print("  - plots/metrics_comparison.png")
    print("  - plots/detailed_metrics.txt")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
