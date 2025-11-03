"""
Quick validation script to test data loading and basic functionality.
Run this before training to ensure everything is set up correctly.
"""

import os
import sys
import numpy as np
import pandas as pd

def check_dependencies():
    """Check if all required packages are installed."""
    print("=" * 80)
    print("CHECKING DEPENDENCIES")
    print("=" * 80)
    
    dependencies = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'torch': 'PyTorch',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm',
        'joblib': 'joblib',
        'rdkit': 'RDKit'
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name:20s} installed")
        except ImportError:
            print(f"✗ {name:20s} NOT FOUND")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        print("For RDKit: conda install -c conda-forge rdkit")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True


def check_data_files():
    """Check if data files exist and have correct structure."""
    print("\n" + "=" * 80)
    print("CHECKING DATA FILES")
    print("=" * 80)
    
    # Check CSV
    csv_path = "top300_sideeffects_dataset.csv"
    if not os.path.exists(csv_path):
        print(f"✗ {csv_path} NOT FOUND")
        return False
    
    df = pd.read_csv(csv_path)
    print(f"✓ {csv_path}")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {list(df.columns[:5])}...")
    print(f"  - Drugs: {len(df)}")
    print(f"  - Side effects: {len(df.columns) - 3}")
    
    # Check required columns
    required_cols = ['CID', 'SMILES', 'Drug']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        return False
    
    # Check npy files
    x_path = "X_ecfp.npy"
    y_path = "y_labels.npy"
    
    if os.path.exists(x_path) and os.path.exists(y_path):
        X = np.load(x_path)
        y = np.load(y_path)
        print(f"\n✓ {x_path}")
        print(f"  - Shape: {X.shape}")
        print(f"  - Dtype: {X.dtype}")
        print(f"  - Non-zero features: {(X > 0).sum() / X.size * 100:.1f}%")
        
        print(f"\n✓ {y_path}")
        print(f"  - Shape: {y.shape}")
        print(f"  - Dtype: {y.dtype}")
        print(f"  - Positive labels: {y.sum() / y.size * 100:.1f}%")
        
        # Verify alignment
        if X.shape[0] != y.shape[0]:
            print(f"✗ Sample count mismatch: X has {X.shape[0]}, y has {y.shape[0]}")
            return False
        
        if X.shape[0] != len(df):
            print(f"⚠️  Warning: CSV has {len(df)} rows, but npy files have {X.shape[0]}")
    else:
        print(f"\n⚠️  {x_path} and/or {y_path} not found")
        print("   They will be generated during first training run")
    
    print("\n✓ Data files OK!")
    return True


def check_rdkit_functionality():
    """Test RDKit SMILES parsing."""
    print("\n" + "=" * 80)
    print("CHECKING RDKIT FUNCTIONALITY")
    print("=" * 80)
    
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        # Test SMILES parsing
        test_smiles = [
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin"),
            ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen"),
            ("INVALID_SMILES", "Invalid (should fail)")
        ]
        
        for smiles, name in test_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"✗ {name:20s} - Invalid SMILES")
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                n_bits = len([b for b in fp.GetOnBits()])
                print(f"✓ {name:20s} - {n_bits} bits set in fingerprint")
        
        print("\n✓ RDKit working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ RDKit error: {e}")
        return False


def check_pytorch_setup():
    """Check PyTorch installation and device availability."""
    print("\n" + "=" * 80)
    print("CHECKING PYTORCH SETUP")
    print("=" * 80)
    
    try:
        import torch
        
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU device: {torch.cuda.get_device_name(0)}")
            print(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"  - Running on CPU (training will be slower)")
        
        # Test basic operations
        x = torch.randn(10, 2048)
        model = torch.nn.Linear(2048, 299)
        y = model(x)
        print(f"\n✓ Basic PyTorch operations working!")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {y.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ PyTorch error: {e}")
        return False


def estimate_training_time():
    """Estimate training time based on device."""
    print("\n" + "=" * 80)
    print("TRAINING TIME ESTIMATE")
    print("=" * 80)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print("With GPU: ~3-5 minutes for 20 epochs")
        else:
            print("With CPU: ~5-10 minutes for 20 epochs")
        
        print("\nRecommended settings:")
        print("  - BATCH_SIZE: 32 (decrease if out of memory)")
        print("  - EPOCHS: 20 (increase for better performance)")
        print("  - LR: 1e-3 (decrease if loss is unstable)")
        
    except:
        pass


def main():
    """Run all validation checks."""
    print("\n" + "=" * 80)
    print("DRUG SIDE-EFFECT PREDICTION - VALIDATION SCRIPT")
    print("=" * 80)
    
    all_ok = True
    
    # Run checks
    all_ok &= check_dependencies()
    all_ok &= check_data_files()
    all_ok &= check_rdkit_functionality()
    all_ok &= check_pytorch_setup()
    
    estimate_training_time()
    
    # Final summary
    print("\n" + "=" * 80)
    if all_ok:
        print("✓ ALL CHECKS PASSED")
        print("=" * 80)
        print("\nYou're ready to train!")
        print("Run: python model.py")
    else:
        print("✗ SOME CHECKS FAILED")
        print("=" * 80)
        print("\nPlease fix the issues above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
