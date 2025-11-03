"""
Helper script for common project tasks.
Provides shortcuts for training, validation, and prediction.

Usage:
    python run.py validate          # Check setup
    python run.py train             # Train model
    python run.py train --epochs 50 # Train with custom epochs
    python run.py predict "SMILES"  # Predict for a SMILES string
    python run.py info              # Show project info
"""

import argparse
import os
import sys
import subprocess


def run_validate():
    """Run validation script."""
    print("Running setup validation...\n")
    return subprocess.call([sys.executable, "validate_setup.py"])


def run_train(epochs=None):
    """Run training script."""
    print("Starting model training...\n")
    
    # Optionally modify epochs in model.py
    if epochs:
        print(f"Note: To change epochs, edit EPOCHS in model.py (currently set to 20)")
        print(f"You requested {epochs} epochs.\n")
    
    return subprocess.call([sys.executable, "model.py"])


def run_predict(smiles, top_k=10):
    """Run prediction for a SMILES string."""
    if not smiles:
        print("Error: No SMILES string provided")
        print("Usage: python run.py predict 'CC(=O)OC1=CC=CC=C1C(=O)O'")
        return 1
    
    print(f"Predicting side effects for: {smiles}\n")
    return subprocess.call([
        sys.executable, "predict.py", 
        "--smiles", smiles,
        "--top_k", str(top_k)
    ])


def show_info():
    """Show project information."""
    print("=" * 80)
    print("DRUG SIDE-EFFECT PREDICTION PROJECT")
    print("=" * 80)
    print("\n📊 Dataset:")
    print("  - Samples: 1,105 drugs")
    print("  - Features: 2,048 ECFP bits")
    print("  - Labels: 299 side effects (multi-label)")
    
    print("\n🏗️  Model Architecture:")
    print("  - Input: ECFP fingerprint (2048)")
    print("  - Hidden: [1024, 512]")
    print("  - Output: 299 side effects")
    print("  - Loss: Multi-label Focal Loss")
    
    print("\n📁 Files:")
    files = {
        "model.py": "Main training script",
        "predict.py": "Inference script",
        "validate_setup.py": "Setup validation",
        "requirements.txt": "Dependencies",
        "README.md": "Documentation",
        "top300_sideeffects_dataset.csv": "Dataset",
        "X_ecfp.npy": "Features",
        "y_labels.npy": "Labels",
    }
    
    for file, desc in files.items():
        exists = "✓" if os.path.exists(file) else "✗"
        print(f"  {exists} {file:35s} - {desc}")
    
    print("\n🚀 Quick Commands:")
    print("  python run.py validate           # Check setup")
    print("  python run.py train              # Train model")
    print("  python run.py predict 'SMILES'   # Make prediction")
    
    print("\n📚 Documentation:")
    print("  - Full guide: README.md")
    print("  - Project status: PROJECT_STATUS.md")
    print("=" * 80)


def show_examples():
    """Show example SMILES strings."""
    print("=" * 80)
    print("EXAMPLE DRUG SMILES STRINGS")
    print("=" * 80)
    
    examples = [
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),
        ("Metformin", "CN(C)C(=N)NC(=N)N"),
    ]
    
    print("\nCopy these SMILES to test predictions:\n")
    for name, smiles in examples:
        print(f"{name:15s}: {smiles}")
    
    print("\nUsage:")
    print(f'  python run.py predict "{examples[0][1]}"')
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Helper script for drug side-effect prediction project"
    )
    
    parser.add_argument(
        "command",
        choices=["validate", "train", "predict", "info", "examples"],
        help="Command to run"
    )
    
    parser.add_argument(
        "smiles",
        nargs="?",
        help="SMILES string for prediction (required for 'predict' command)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (for 'train' command)"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top predictions to show (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == "validate":
        return run_validate()
    
    elif args.command == "train":
        return run_train(args.epochs)
    
    elif args.command == "predict":
        return run_predict(args.smiles, args.top_k)
    
    elif args.command == "info":
        show_info()
        return 0
    
    elif args.command == "examples":
        show_examples()
        return 0
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
