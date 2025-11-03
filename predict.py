"""
Inference script for drug side-effect prediction.

This script loads a trained model and predicts potential side effects
for a given drug SMILES string.

Usage:
    python predict.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
    python predict.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" --top_k 20
    
Interactive mode:
    python predict.py --interactive
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
import joblib

# Import fingerprint function from model.py
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, MolSurf
except ImportError as e:
    print("RDKit not available. Please install: conda install -c conda-forge rdkit")
    raise e


def smiles_to_ecfp_vector(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Convert SMILES to ECFP fingerprint vector."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        
        # Use Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        for bit in fp.GetOnBits():
            if bit < n_bits:
                arr[bit] = 1.0
        return arr
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)


def smiles_to_rdkit_descriptors(smiles: str) -> np.ndarray:
    """Extract RDKit molecular descriptors from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(200, dtype=np.float32)
        
        descriptors = []
        
        # Basic molecular properties
        descriptors.append(Descriptors.MolWt(mol))
        descriptors.append(Crippen.MolLogP(mol))
        descriptors.append(Descriptors.TPSA(mol))
        descriptors.append(Lipinski.NumHDonors(mol))
        descriptors.append(Lipinski.NumHAcceptors(mol))
        descriptors.append(Lipinski.NumRotatableBonds(mol))
        descriptors.append(Descriptors.NumAromaticRings(mol))
        descriptors.append(Descriptors.NumAliphaticRings(mol))
        descriptors.append(Descriptors.NumSaturatedRings(mol))
        descriptors.append(Descriptors.FractionCSP3(mol))
        descriptors.append(Descriptors.NumHeteroatoms(mol))
        descriptors.append(Descriptors.HeavyAtomCount(mol))
        descriptors.append(Descriptors.RingCount(mol))
        
        # Kappa indices
        descriptors.append(Descriptors.Kappa1(mol))
        descriptors.append(Descriptors.Kappa2(mol))
        descriptors.append(Descriptors.Kappa3(mol))
        
        # Graph-based descriptors
        descriptors.append(Descriptors.BalabanJ(mol))
        descriptors.append(Descriptors.BertzCT(mol))
        
        # Chi indices
        descriptors.append(Descriptors.Chi0(mol))
        descriptors.append(Descriptors.Chi1(mol))
        descriptors.append(Descriptors.Chi2n(mol))
        descriptors.append(Descriptors.Chi3n(mol))
        descriptors.append(Descriptors.Chi4n(mol))
        
        # VSA descriptors
        descriptors.append(MolSurf.LabuteASA(mol))
        descriptors.extend(Descriptors.PEOE_VSA_(mol))
        descriptors.extend(Descriptors.SMR_VSA_(mol))
        descriptors.extend(Descriptors.SlogP_VSA_(mol))
        
        # Partial charges
        try:
            descriptors.append(Descriptors.MaxPartialCharge(mol))
            descriptors.append(Descriptors.MinPartialCharge(mol))
        except:
            descriptors.extend([0.0, 0.0])
        
        # Pad or trim to exactly 200 features
        desc_array = np.array(descriptors, dtype=np.float32)
        if len(desc_array) < 200:
            desc_array = np.pad(desc_array, (0, 200 - len(desc_array)), mode='constant')
        else:
            desc_array = desc_array[:200]
        
        # Replace any NaN or inf values
        desc_array = np.nan_to_num(desc_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return desc_array
        
    except Exception:
        return np.zeros(200, dtype=np.float32)


def smiles_to_combined_features(smiles: str, n_bits: int = 2048) -> np.ndarray:
    """Combine ECFP fingerprint with RDKit descriptors."""
    ecfp = smiles_to_ecfp_vector(smiles, n_bits=n_bits)
    descriptors = smiles_to_rdkit_descriptors(smiles)
    return np.concatenate([ecfp, descriptors])


class MLP_MultiLabel(nn.Module):
    """Multi-label classification model."""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim
        
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev = h
        
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_model(checkpoint_dir: str = "checkpoints"):
    """Load trained model and metadata."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try to load from inference bundle first
    bundle_path = os.path.join(checkpoint_dir, 'inference_bundle.joblib')
    use_combined = False
    
    if os.path.exists(bundle_path):
        bundle = joblib.load(bundle_path)
        model_config = bundle['model_config']
        thresholds = bundle['thresholds']
        label_names = bundle['label_names']
        # Infer use_combined from input_dim if not explicitly set
        use_combined = model_config.get('use_combined_features', model_config['input_dim'] == 2248)
        print(f"✓ Loaded configuration from inference bundle")
    else:
        # Fallback to loading individual files
        label_names_path = os.path.join(checkpoint_dir, 'label_names.json')
        thresholds_path = os.path.join(checkpoint_dir, 'thresholds.npy')
        
        if not os.path.exists(label_names_path):
            raise FileNotFoundError(f"Label names not found at {label_names_path}")
        if not os.path.exists(thresholds_path):
            raise FileNotFoundError(f"Thresholds not found at {thresholds_path}")
        
        with open(label_names_path, 'r') as f:
            label_names = json.load(f)
        
        thresholds = np.load(thresholds_path)
        
        # Try to detect input dimension from model weights
        model_path = os.path.join(checkpoint_dir, 'final_model.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        
        if os.path.exists(model_path):
            # Load model state dict to check dimensions
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                
                # Get input dimension from first layer weight
                first_layer_key = 'net.0.weight'
                if first_layer_key in state_dict:
                    input_dim = state_dict[first_layer_key].shape[1]
                    use_combined = (input_dim == 2248)
                    print(f"✓ Detected input dimension from model: {input_dim}")
                else:
                    # Default to 2248 (combined features)
                    input_dim = 2248
                    use_combined = True
                    print(f"⚠ Could not detect input dimension, defaulting to {input_dim}")
            except Exception as e:
                print(f"⚠ Error detecting dimensions: {e}")
                input_dim = 2248
                use_combined = True
        else:
            raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")
        
        # Infer configuration
        model_config = {
            'input_dim': input_dim,
            'hidden_dims': [1024, 512],
            'output_dim': len(label_names),
            'dropout': 0.3
        }
    
    # Initialize model
    model = MLP_MultiLabel(
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        output_dim=model_config['output_dim'],
        dropout=model_config['dropout']
    ).to(device)
    
    # Load weights
    model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded model from best_model.pt (epoch {checkpoint['epoch']})")
        else:
            raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Loaded model from final_model.pt")
    
    model.eval()
    
    feature_type = "ECFP + RDKit Descriptors (2248 dim)" if use_combined else "ECFP only (2048 dim)"
    print(f"✓ Feature type: {feature_type}")
    
    return model, thresholds, label_names, device, use_combined


def predict_side_effects(
    smiles: str, 
    model: nn.Module, 
    thresholds: np.ndarray, 
    label_names: List[str], 
    device: torch.device,
    top_k: int = 10,
    use_combined: bool = False
) -> List[Tuple[str, float, float]]:
    """
    Predict side effects for a given SMILES string.
    
    Args:
        smiles: Drug SMILES string
        model: Trained PyTorch model
        thresholds: Per-class probability thresholds
        label_names: List of side effect names
        device: Device to run inference on
        top_k: Number of top predictions to return
        use_combined: Whether to use combined features (ECFP + descriptors)
        
    Returns:
        List of tuples: (side_effect_name, probability, threshold)
    """
    # Validate SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"WARNING: Invalid SMILES string: {smiles}")
        return []
    
    # Convert SMILES to fingerprint
    if use_combined:
        fp = smiles_to_combined_features(smiles, n_bits=2048)
    else:
        n_bits = 2248 if use_combined else 2048  
        fp = smiles_to_ecfp_vector(smiles, n_bits=n_bits)
    
    X = torch.tensor(fp, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    
    # Get predictions above threshold
    predictions = []
    for i, (prob, threshold) in enumerate(zip(probs, thresholds)):
        if prob >= threshold:
            predictions.append((label_names[i], float(prob), float(threshold)))
    
    # Sort by probability (descending)
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k
    return predictions[:top_k]


def main():
    parser = argparse.ArgumentParser(
        description='Predict drug side effects from SMILES string'
    )
    parser.add_argument(
        '--smiles', 
        type=str, 
        help='SMILES string of the drug molecule'
    )
    parser.add_argument(
        '--top_k', 
        type=int, 
        default=10,
        help='Number of top predictions to show (default: 10)'
    )
    parser.add_argument(
        '--checkpoint_dir', 
        type=str, 
        default='checkpoints',
        help='Directory containing model checkpoints (default: checkpoints)'
    )
    parser.add_argument(
        '--show_all',
        action='store_true',
        help='Show all predictions above threshold (ignores --top_k)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    # Load model
    print("Loading trained model...")
    model, thresholds, label_names, device, use_combined = load_model(args.checkpoint_dir)
    print(f"✓ Model loaded successfully. Device: {device}")
    print(f"✓ Number of side effects: {len(label_names)}\n")
    
    if args.interactive:
        # Interactive mode
        print("=" * 80)
        print("Interactive Prediction Mode")
        print("=" * 80)
        print("Enter a SMILES string to predict side effects (or 'quit' to exit)\n")
        
        while True:
            try:
                smiles = input("SMILES: ").strip()
                
                if smiles.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not smiles:
                    continue
                
                top_k = None if args.show_all else args.top_k
                predictions = predict_side_effects(
                    smiles, 
                    model, 
                    thresholds, 
                    label_names, 
                    device,
                    top_k=top_k if top_k else len(label_names),
                    use_combined=use_combined
                )
                
                if predictions:
                    print(f"\nPredicted side effects (top {len(predictions)}):")
                    print("-" * 80)
                    for i, (side_effect, prob, threshold) in enumerate(predictions, 1):
                        print(f"{i:3d}. {side_effect:45s} | Prob: {prob:.4f} | Threshold: {threshold:.4f}")
                    print("-" * 80)
                else:
                    print("No side effects predicted above threshold.\n")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")
    else:
        # Single prediction mode
        if not args.smiles:
            parser.error("--smiles is required when not in interactive mode")
        
        print(f"Input SMILES: {args.smiles}\n")
        
        top_k = None if args.show_all else args.top_k
        predictions = predict_side_effects(
            args.smiles, 
            model, 
            thresholds, 
            label_names, 
            device,
            top_k=top_k if top_k else len(label_names),
            use_combined=use_combined
        )
        
        if predictions:
            print(f"Predicted side effects (showing top {len(predictions)}):")
            print("-" * 80)
            for i, (side_effect, prob, threshold) in enumerate(predictions, 1):
                print(f"{i:3d}. {side_effect:45s} | Prob: {prob:.4f} | Threshold: {threshold:.4f}")
            print("-" * 80)
            print(f"\nTotal predicted side effects above threshold: {len(predictions)}")
        else:
            print("No side effects predicted above threshold.")
            print("This could mean:")
            print("  1. The drug is predicted to be relatively safe")
            print("  2. The SMILES string is invalid")
            print("  3. The model needs more training data for this type of molecule")


if __name__ == "__main__":
    main()
