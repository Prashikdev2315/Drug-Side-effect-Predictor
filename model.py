import os
import time
import math
import json
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import (
    f1_score, hamming_loss, accuracy_score, 
    precision_score, recall_score, multilabel_confusion_matrix,
    classification_report
)
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Config
# -------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# File paths (using workspace relative paths)
CSV_PATH = "top300_sideeffects_dataset.csv"  # Your dataset
X_PATH = "X_ecfp.npy"   # Will be created if doesn't exist
Y_PATH = "y_labels.npy"  # Will be created if doesn't exist

# Training mode control
TRAIN_MODE = False  # Set to True to train a new model, False to use existing saved model

# Feature engineering options
USE_COMBINED_FEATURES = True  # Set to True to use ECFP + RDKit descriptors (2248 dim)
                               # Set to False to use only ECFP (2048 dim)

# Training hyperparams
BATCH_SIZE = 32  # Reduced for safety
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5
VALID_FRAC = 0.15
TEST_FRAC = 0.10
NUM_WORKERS = 0  # Changed to 0 to avoid multiprocessing issues on Windows

# Model architecture
INPUT_DIM = 2248 if USE_COMBINED_FEATURES else 2048  # Auto-adjust based on features
HIDDEN_DIMS = [1024, 512]  # Reduced for stability
DROPOUT = 0.3

# Loss function options
USE_BCE_LOSS = False  # Set to True for BCEWithLogitsLoss, False for Focal Loss

# Pos weight clipping
POS_WEIGHT_MAX = 10.0  # Reduced to prevent extreme values

# Checkpointing
CKPT_DIR = "checkpoints"
PLOTS_DIR = "plots"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

USE_AMP = True  # enable mixed precision if available

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
print(f"Feature mode: {'ECFP + Descriptors (2248 dim)' if USE_COMBINED_FEATURES else 'ECFP only (2048 dim)'}")
print(f"Loss function: {'BCEWithLogitsLoss' if USE_BCE_LOSS else 'Focal Loss'}")

# -------------------------
# Utilities: RDKit fingerprint + Descriptors
# -------------------------
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
    """
    Extract important RDKit molecular descriptors.
    Returns a 200-dimensional feature vector with key physicochemical properties.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(200, dtype=np.float32)
        
        # Create descriptor list with safe computation
        desc_list = []
        
        # Basic molecular properties (10 descriptors)
        desc_list.append(Descriptors.MolWt(mol))  # Molecular weight
        desc_list.append(Descriptors.MolLogP(mol))  # LogP
        desc_list.append(Descriptors.TPSA(mol))  # Topological polar surface area
        desc_list.append(Descriptors.NumHDonors(mol))  # H-bond donors
        desc_list.append(Descriptors.NumHAcceptors(mol))  # H-bond acceptors
        desc_list.append(Descriptors.NumRotatableBonds(mol))  # Rotatable bonds
        desc_list.append(Descriptors.NumAromaticRings(mol))  # Aromatic rings
        desc_list.append(Descriptors.NumAliphaticRings(mol))  # Aliphatic rings
        desc_list.append(Descriptors.FractionCsp3(mol))  # Fraction sp3 carbons
        desc_list.append(Descriptors.RingCount(mol))  # Total rings
        
        # Lipinski descriptors (5)
        desc_list.append(Lipinski.NumHeteroatoms(mol))
        desc_list.append(Lipinski.NumHDonors(mol))
        desc_list.append(Lipinski.NumHAcceptors(mol))
        desc_list.append(Lipinski.NumRotatableBonds(mol))
        desc_list.append(Lipinski.NumSaturatedRings(mol))
        
        # Crippen parameters (2)
        desc_list.append(Crippen.MolLogP(mol))
        desc_list.append(Crippen.MolMR(mol))  # Molar refractivity
        
        # Surface area descriptors (3)
        desc_list.append(MolSurf.TPSA(mol))
        desc_list.append(MolSurf.LabuteASA(mol))
        try:
            desc_list.append(MolSurf.PEOE_VSA1(mol))
        except:
            desc_list.append(0.0)
        
        # Additional important descriptors (30)
        desc_list.append(Descriptors.BalabanJ(mol))
        desc_list.append(Descriptors.BertzCT(mol))
        desc_list.append(Descriptors.Chi0(mol))
        desc_list.append(Descriptors.Chi1(mol))
        desc_list.append(Descriptors.HallKierAlpha(mol))
        desc_list.append(Descriptors.Kappa1(mol))
        desc_list.append(Descriptors.Kappa2(mol))
        desc_list.append(Descriptors.Kappa3(mol))
        desc_list.append(Descriptors.LabuteASA(mol))
        desc_list.append(Descriptors.PEOE_VSA1(mol))
        desc_list.append(Descriptors.PEOE_VSA2(mol))
        desc_list.append(Descriptors.PEOE_VSA3(mol))
        desc_list.append(Descriptors.SMR_VSA1(mol))
        desc_list.append(Descriptors.SMR_VSA2(mol))
        desc_list.append(Descriptors.SMR_VSA3(mol))
        desc_list.append(Descriptors.SlogP_VSA1(mol))
        desc_list.append(Descriptors.SlogP_VSA2(mol))
        desc_list.append(Descriptors.SlogP_VSA3(mol))
        desc_list.append(Descriptors.EState_VSA1(mol))
        desc_list.append(Descriptors.EState_VSA2(mol))
        desc_list.append(Descriptors.VSA_EState1(mol))
        desc_list.append(Descriptors.VSA_EState2(mol))
        desc_list.append(Descriptors.NumValenceElectrons(mol))
        desc_list.append(Descriptors.MaxAbsPartialCharge(mol))
        desc_list.append(Descriptors.MinAbsPartialCharge(mol))
        desc_list.append(Descriptors.MaxPartialCharge(mol))
        desc_list.append(Descriptors.MinPartialCharge(mol))
        desc_list.append(Descriptors.MolWt(mol))
        desc_list.append(Descriptors.ExactMolWt(mol))
        desc_list.append(Descriptors.HeavyAtomMolWt(mol))
        
        # Normalize and pad to 200 dimensions
        desc_array = np.array(desc_list, dtype=np.float32)
        
        # Handle any NaN or Inf values
        desc_array = np.nan_to_num(desc_array, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Pad to 200 dimensions with zeros if needed
        if len(desc_array) < 200:
            desc_array = np.pad(desc_array, (0, 200 - len(desc_array)), 'constant')
        elif len(desc_array) > 200:
            desc_array = desc_array[:200]
            
        return desc_array
        
    except Exception as e:
        # Return zeros on any error
        return np.zeros(200, dtype=np.float32)

def smiles_to_combined_features(smiles: str, n_bits: int = 2048) -> np.ndarray:
    """
    Combine Morgan fingerprints with RDKit descriptors.
    Returns: concatenated vector of ECFP (2048) + Descriptors (200) = 2248 dimensions
    """
    ecfp = smiles_to_ecfp_vector(smiles, n_bits=n_bits)
    descriptors = smiles_to_rdkit_descriptors(smiles)
    return np.concatenate([ecfp, descriptors])

# -------------------------
# Dataset classes
# -------------------------
class NumpyMultiLabelDataset(Dataset):
    def __init__(self, x_path: str, y_path: str):
        self.X = np.load(x_path)
        self.y = np.load(y_path)
        assert self.X.shape[0] == self.y.shape[0], f"X and y row mismatch: {self.X.shape[0]} vs {self.y.shape[0]}"
        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FingerprintFromCSV(Dataset):
    """Compute ECFP on-the-fly from SMILES."""
    def __init__(self, csv_path: str, n_bits: int = 2048):
        df = pd.read_csv(csv_path)
        # Keep label names for inference
        self.label_names = list(df.columns[3:])  # All columns after CID, SMILES, Drug
        self.smiles = df['SMILES'].fillna('').tolist()
        self.y = df.iloc[:, 3:].values.astype(np.float32)
        self.n_bits = n_bits
        
        print(f"Loaded {len(self.smiles)} samples with {len(self.label_names)} side effects")

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        s = self.smiles[idx]
        fp = smiles_to_ecfp_vector(s, n_bits=self.n_bits)
        return fp, self.y[idx]

# -------------------------
# Model
# -------------------------
class MLP_MultiLabel(nn.Module):
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
        
        # Final layer without activation (we'll use sigmoid in loss)
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------
# Loss: Multi-label Focal Loss
# -------------------------
class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight: Optional[torch.Tensor] = None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        # Compute probabilities
        prob = torch.sigmoid(logits)
        
        # Binary cross entropy with logits
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        
        # Focal loss components
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = alpha_factor * modulating_factor * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# -------------------------
# Helper functions
# -------------------------
def compute_pos_weight(y_np: np.ndarray, clip_max: float = POS_WEIGHT_MAX) -> torch.Tensor:
    """Compute positive weights for imbalanced classes."""
    pos = y_np.sum(axis=0).astype(np.float64)
    neg = y_np.shape[0] - pos
    pos_safe = np.where(pos == 0, 1.0, pos)
    pw = neg / pos_safe
    pw = np.clip(pw, a_min=1.0, a_max=clip_max)
    return torch.tensor(pw, dtype=torch.float32)

def split_dataset(dataset, valid_frac: float, test_frac: float, seed: int = SEED):
    """Split dataset into train/validation/test."""
    N = len(dataset)
    n_test = int(math.floor(N * test_frac))
    n_valid = int(math.floor(N * valid_frac))
    n_train = N - n_valid - n_test
    lengths = [n_train, n_valid, n_test]
    return random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed))

def evaluate_model(model, data_loader, device, threshold=0.5):
    """Evaluate model on given data loader."""
    model.eval()
    all_probs = []
    all_trues = []
    
    with torch.no_grad():
        for Xb, yb in data_loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_trues.append(yb.numpy())
    
    all_probs = np.vstack(all_probs)
    all_trues = np.vstack(all_trues)
    preds = (all_probs >= threshold).astype(int)
    
    # Compute all metrics
    micro_f1 = f1_score(all_trues, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(all_trues, preds, average='macro', zero_division=0)
    micro_precision = precision_score(all_trues, preds, average='micro', zero_division=0)
    micro_recall = recall_score(all_trues, preds, average='micro', zero_division=0)
    ham_loss = hamming_loss(all_trues, preds)
    
    # Calculate per-label accuracy (more meaningful than subset accuracy)
    per_label_accuracy = 1.0 - ham_loss  # This is the real accuracy!
    
    metrics = {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'per_label_accuracy': per_label_accuracy,  # Use this instead of subset accuracy
        'hamming_loss': ham_loss
    }
    
    return metrics, all_probs, all_trues, preds

def tune_thresholds_per_class(valid_loader, model, device, num_classes, steps=50):
    """Tune per-class thresholds for optimal F1 score."""
    model.eval()
    all_probs = []
    all_trues = []
    
    with torch.no_grad():
        for Xb, yb in valid_loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_trues.append(yb.numpy())
    
    all_probs = np.vstack(all_probs)
    all_trues = np.vstack(all_trues)
    
    thresholds = np.full(num_classes, 0.5, dtype=np.float32)
    pos_counts = all_trues.sum(axis=0)
    min_pos = 5  # Minimum positives required for tuning
    
    for c in range(num_classes):
        if pos_counts[c] < min_pos:
            continue  # Keep default 0.5 for rare classes
            
        y_true = all_trues[:, c]
        y_prob = all_probs[:, c]
        
        best_f1 = -1
        best_thresh = 0.5
        
        for threshold in np.linspace(0.1, 0.9, steps):
            y_pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = threshold
                
        thresholds[c] = best_thresh
    
    return thresholds

def create_fingerprint_files(csv_path: str, x_path: str, y_path: str, n_bits: int = 2048, use_combined: bool = False):
    """Create fingerprint files from CSV if they don't exist."""
    if os.path.exists(x_path) and os.path.exists(y_path):
        print("Fingerprint files already exist. Loading...")
        return
        
    print("Creating fingerprint files from CSV...")
    df = pd.read_csv(csv_path)
    
    # Extract features and labels
    fingerprints = []
    feature_func = smiles_to_combined_features if use_combined else smiles_to_ecfp_vector
    
    for smiles in tqdm(df['SMILES'], desc="Generating fingerprints"):
        fp = feature_func(smiles, n_bits=n_bits)
        fingerprints.append(fp)
    
    X = np.array(fingerprints, dtype=np.float32)
    y = df.iloc[:, 3:].values.astype(np.float32)  # All columns after CID, SMILES, Drug
    
    # Save files
    np.save(x_path, X)
    np.save(y_path, y)
    print(f"Saved fingerprints: X.shape={X.shape}, y.shape={y.shape}")
    print(f"Feature type: {'ECFP + Descriptors' if use_combined else 'ECFP only'}")

# -------------------------
# Visualization functions
# -------------------------
def plot_training_history(history, save_path='plots/training_history.png'):
    """Plot training and validation loss/metrics over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training vs Validation Loss', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 Score plot
    axes[0, 1].plot(epochs, history['val_f1'], 'g-', label='Validation F1', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('F1 Score', fontsize=12)
    axes[0, 1].set_title('Validation F1 Score over Epochs', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Per-Label Accuracy plot
    axes[1, 0].plot(epochs, history['val_per_label_accuracy'], 'orange', label='Validation Per-Label Accuracy', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Per-Label Accuracy', fontsize=12)
    axes[1, 0].set_title('Validation Per-Label Accuracy over Epochs', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Hamming Loss plot
    axes[1, 1].plot(epochs, history['val_hamming'], 'purple', label='Validation Hamming Loss', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Hamming Loss', fontsize=12)
    axes[1, 1].set_title('Validation Hamming Loss over Epochs', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix_multilabel(y_true, y_pred, label_names, 
                                     top_n=10, save_path='plots/confusion_matrices.png'):
    """Plot confusion matrices for top N most common side effects."""
    # Find top N most common side effects
    positive_counts = y_true.sum(axis=0)
    top_indices = np.argsort(positive_counts)[-top_n:][::-1]
    
    n_rows = (top_n + 2) // 3
    n_cols = min(3, top_n)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle(f'Confusion Matrices - Top {top_n} Most Common Side Effects', 
                 fontsize=16, fontweight='bold')
    
    if top_n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, class_idx in enumerate(top_indices):
        cm = multilabel_confusion_matrix(y_true, y_pred)[class_idx]
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['Actual 0', 'Actual 1'],
                   ax=axes[idx], cbar_kws={'label': 'Count'})
        
        # Calculate metrics for this class
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        title = f"{label_names[class_idx]}\n"
        title += f"P:{precision:.3f} R:{recall:.3f} F1:{f1:.3f}"
        axes[idx].set_title(title, fontsize=10)
    
    # Hide extra subplots
    for idx in range(top_n, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved to {save_path}")
    plt.close()


def plot_metrics_comparison(metrics_dict, save_path='plots/metrics_comparison.png'):
    """Plot bar chart comparing different metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_to_plot = {
        'Micro F1': metrics_dict.get('micro_f1', 0),
        'Macro F1': metrics_dict.get('macro_f1', 0),
        'Precision': metrics_dict.get('micro_precision', 0),
        'Recall': metrics_dict.get('micro_recall', 0),
        'Per-Label Acc': metrics_dict.get('per_label_accuracy', 0),
    }
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = ax.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color=colors, alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison saved to {save_path}")
    plt.close()


def save_detailed_metrics_report(y_true, y_pred, label_names, save_path='plots/detailed_metrics.txt'):
    """Save detailed classification report to file."""
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Micro F1 Score:      {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}\n")
        f.write(f"Macro F1 Score:      {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}\n")
        f.write(f"Micro Precision:     {precision_score(y_true, y_pred, average='micro', zero_division=0):.4f}\n")
        f.write(f"Micro Recall:        {recall_score(y_true, y_pred, average='micro', zero_division=0):.4f}\n")
        ham_loss_val = hamming_loss(y_true, y_pred)
        f.write(f"Per-Label Accuracy:  {1.0 - ham_loss_val:.4f}\n")
        f.write(f"Hamming Loss:        {ham_loss_val:.4f}\n\n")
        
        # Per-class metrics
        f.write("PER-CLASS METRICS (Top 20 by F1 Score):\n")
        f.write("-" * 80 + "\n")
        
        # Calculate per-class F1 scores
        per_class_f1 = []
        for i in range(y_true.shape[1]):
            f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
            recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
            support = y_true[:, i].sum()
            per_class_f1.append((i, f1, precision, recall, support))
        
        # Sort by F1 score
        per_class_f1.sort(key=lambda x: x[1], reverse=True)
        
        f.write(f"{'Rank':<5} {'Side Effect':<40} {'F1':<8} {'Prec':<8} {'Recall':<8} {'Support':<8}\n")
        f.write("-" * 80 + "\n")
        
        for rank, (idx, f1, prec, rec, support) in enumerate(per_class_f1[:20], 1):
            name = label_names[idx][:38]  # Truncate long names
            f.write(f"{rank:<5} {name:<40} {f1:<8.4f} {prec:<8.4f} {rec:<8.4f} {support:<8.0f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Detailed metrics report saved to {save_path}")


# -------------------------
# Training function
# -------------------------
def train_and_tune():
    """Main training function."""
    # Create fingerprint files if they don't exist
    create_fingerprint_files(CSV_PATH, X_PATH, Y_PATH, n_bits=2048, use_combined=USE_COMBINED_FEATURES)
    
    # Load dataset
    dataset = NumpyMultiLabelDataset(X_PATH, Y_PATH)
    
    # Get label names from CSV
    df = pd.read_csv(CSV_PATH)
    label_names = list(df.columns[3:])
    
    n_samples, n_features = dataset.X.shape
    n_classes = dataset.y.shape[1]
    print(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
    
    # Verify feature dimension matches configuration
    expected_features = 2248 if USE_COMBINED_FEATURES else 2048
    if n_features != expected_features:
        print(f"⚠️  Warning: Feature dimension mismatch!")
        print(f"   Expected: {expected_features}, Got: {n_features}")
        print(f"   Deleting old features and regenerating...")
        os.remove(X_PATH)
        os.remove(Y_PATH)
        create_fingerprint_files(CSV_PATH, X_PATH, Y_PATH, n_bits=2048, use_combined=USE_COMBINED_FEATURES)
        dataset = NumpyMultiLabelDataset(X_PATH, Y_PATH)
        n_samples, n_features = dataset.X.shape
        print(f"   Regenerated: {n_samples} samples, {n_features} features")
    
    # Split dataset
    train_set, valid_set, test_set = split_dataset(dataset, VALID_FRAC, TEST_FRAC)
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                           num_workers=NUM_WORKERS, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=NUM_WORKERS, pin_memory=True)
    
    # Initialize model
    model = MLP_MultiLabel(
        input_dim=n_features, 
        hidden_dims=HIDDEN_DIMS, 
        output_dim=n_classes, 
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Compute positive weights
    pos_weight = compute_pos_weight(dataset.y)
    pos_weight = pos_weight.to(DEVICE)
    print(f"Positive weights - min: {pos_weight.min():.2f}, max: {pos_weight.max():.2f}")
    
    # Loss function selection
    if USE_BCE_LOSS:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("Using BCEWithLogitsLoss with pos_weight")
    else:
        criterion = MultiLabelFocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
        print("Using Focal Loss with pos_weight")
        
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    scaler = GradScaler(enabled=USE_AMP)
    
    # Training variables
    best_val_f1 = 0.0
    best_epoch = 0
    
    # History tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_per_label_accuracy': [],
        'val_hamming': [],
        'val_precision': [],
        'val_recall': []
    }
    
    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS}')
        for Xb, yb in pbar:
            Xb = Xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast(enabled=USE_AMP):
                logits = model(Xb)
                loss = criterion(logits, yb)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / batch_count
        
        # Validation phase
        val_metrics, val_probs, val_trues, val_preds = evaluate_model(model, valid_loader, DEVICE)
        
        # Calculate validation loss
        model.eval()
        val_loss_total = 0.0
        val_batch_count = 0
        with torch.no_grad():
            for Xb, yb in valid_loader:
                Xb = Xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(Xb)
                loss = criterion(logits, yb)
                val_loss_total += loss.item()
                val_batch_count += 1
        avg_val_loss = val_loss_total / val_batch_count
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_metrics['micro_f1'])
        history['val_per_label_accuracy'].append(val_metrics['per_label_accuracy'])
        history['val_hamming'].append(val_metrics['hamming_loss'])
        history['val_precision'].append(val_metrics['micro_precision'])
        history['val_recall'].append(val_metrics['micro_recall'])
        
        print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, '
              f'Val F1 = {val_metrics["micro_f1"]:.4f}, Val Per-Label Acc = {val_metrics["per_label_accuracy"]:.4f}, '
              f'Val Hamming = {val_metrics["hamming_loss"]:.4f}')
        
        # Update learning rate
        scheduler.step(val_metrics['micro_f1'])
        
        # Save best model
        if val_metrics['micro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['micro_f1']
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'history': history
            }
            
            torch.save(checkpoint, os.path.join(CKPT_DIR, 'best_model.pt'))
            print(f'New best model saved with F1: {val_metrics["micro_f1"]:.4f}')
    
    # Load best model for final evaluation
    print(f"\nLoading best model from epoch {best_epoch}...")
    checkpoint = torch.load(os.path.join(CKPT_DIR, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot training history
    print("\nGenerating training history plots...")
    plot_training_history(history, save_path=os.path.join(PLOTS_DIR, 'training_history.png'))
    
    # Tune thresholds
    print("Tuning per-class thresholds...")
    thresholds = tune_thresholds_per_class(valid_loader, model, DEVICE, n_classes)
    
    # Final evaluation with tuned thresholds
    print("\nEvaluating on test set...")
    model.eval()
    test_probs = []
    test_trues = []
    
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(DEVICE)
            logits = model(Xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.append(probs)
            test_trues.append(yb.numpy())
    
    test_probs = np.vstack(test_probs)
    test_trues = np.vstack(test_trues)
    
    # Evaluate with tuned thresholds
    test_preds = (test_probs >= thresholds).astype(int)
    
    # Calculate comprehensive metrics
    test_metrics = {
        'micro_f1': f1_score(test_trues, test_preds, average='micro', zero_division=0),
        'macro_f1': f1_score(test_trues, test_preds, average='macro', zero_division=0),
        'micro_precision': precision_score(test_trues, test_preds, average='micro', zero_division=0),
        'micro_recall': recall_score(test_trues, test_preds, average='micro', zero_division=0),
        'per_label_accuracy': 1 - hamming_loss(test_trues, test_preds),
        'hamming_loss': hamming_loss(test_trues, test_preds)
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Micro F1 Score:      {test_metrics['micro_f1']:.4f}")
    print(f"Macro F1 Score:      {test_metrics['macro_f1']:.4f}")
    print(f"Micro Precision:     {test_metrics['micro_precision']:.4f}")
    print(f"Micro Recall:        {test_metrics['micro_recall']:.4f}")
    print(f"Per-Label Accuracy:  {test_metrics['per_label_accuracy']:.4f} (i.e., {test_metrics['per_label_accuracy']*100:.2f}% of labels correct)")
    print(f"Hamming Loss:        {test_metrics['hamming_loss']:.4f}")
    print("=" * 80)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix_multilabel(
        test_trues, test_preds, label_names, top_n=12,
        save_path=os.path.join(PLOTS_DIR, 'confusion_matrices.png')
    )
    
    plot_metrics_comparison(
        test_metrics,
        save_path=os.path.join(PLOTS_DIR, 'metrics_comparison.png')
    )
    
    save_detailed_metrics_report(
        test_trues, test_preds, label_names,
        save_path=os.path.join(PLOTS_DIR, 'detailed_metrics.txt')
    )
    
    # Save final artifacts
    print("\nSaving model artifacts...")
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'final_model.pt'))
    np.save(os.path.join(CKPT_DIR, 'thresholds.npy'), thresholds)
    
    with open(os.path.join(CKPT_DIR, 'label_names.json'), 'w') as f:
        json.dump(label_names, f)
    
    # Save test metrics
    with open(os.path.join(CKPT_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Save training history
    with open(os.path.join(CKPT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save inference bundle
    joblib.dump({
        'model_config': {
            'input_dim': n_features,
            'hidden_dims': HIDDEN_DIMS,
            'output_dim': n_classes,
            'dropout': DROPOUT
        },
        'thresholds': thresholds,
        'label_names': label_names,
        'test_metrics': test_metrics,
        'feature_scaler': None  # No scaling for binary fingerprints
    }, os.path.join(CKPT_DIR, 'inference_bundle.joblib'))
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nModel artifacts saved to: {CKPT_DIR}/")
    print(f"Visualizations saved to: {PLOTS_DIR}/")
    print("\nGenerated files:")
    print(f"  - {CKPT_DIR}/final_model.pt")
    print(f"  - {CKPT_DIR}/best_model.pt")
    print(f"  - {CKPT_DIR}/thresholds.npy")
    print(f"  - {CKPT_DIR}/test_metrics.json")
    print(f"  - {CKPT_DIR}/training_history.json")
    print(f"  - {PLOTS_DIR}/training_history.png")
    print(f"  - {PLOTS_DIR}/confusion_matrices.png")
    print(f"  - {PLOTS_DIR}/metrics_comparison.png")
    print(f"  - {PLOTS_DIR}/detailed_metrics.txt")
    print("=" * 80 + "\n")
    
    return model, thresholds, label_names

# -------------------------
# Inference function
# -------------------------
def predict_side_effects(smiles: str, model: nn.Module, thresholds: np.ndarray, 
                       label_names: List[str], top_k: int = 10):
    """Predict side effects for a given SMILES string."""
    model.eval()
    
    # Convert SMILES to fingerprint
    fp = smiles_to_ecfp_vector(smiles, n_bits=INPUT_DIM)
    X = torch.tensor(fp).unsqueeze(0).to(DEVICE)
    
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


def load_trained_model():
    """Load a previously trained model and its artifacts."""
    model_path = os.path.join(CKPT_DIR, 'final_model.pt')
    thresholds_path = os.path.join(CKPT_DIR, 'thresholds.npy')
    label_names_path = os.path.join(CKPT_DIR, 'label_names.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            "Please train the model first by setting TRAIN_MODE = True"
        )
    
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds file not found at {thresholds_path}")
    
    if not os.path.exists(label_names_path):
        raise FileNotFoundError(f"Label names file not found at {label_names_path}")
    
    print("Loading trained model from checkpoints...")
    
    # Load label names to determine number of classes
    with open(label_names_path, 'r') as f:
        label_names = json.load(f)
    
    n_classes = len(label_names)
    
    # Initialize model with same architecture
    model = MLP_MultiLabel(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, 
                        output_dim=n_classes, dropout=DROPOUT).to(DEVICE)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Load thresholds
    thresholds = np.load(thresholds_path)
    
    print(f"✓ Model loaded successfully")
    print(f"✓ Input dimension: {INPUT_DIM}")
    print(f"✓ Number of side effects: {n_classes}")
    print(f"✓ Thresholds loaded: {len(thresholds)}")
    
    return model, thresholds, label_names


# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    # Fix for multiprocessing on Windows
    torch.multiprocessing.freeze_support()
    
    try:
        if TRAIN_MODE:
            # Training mode: Train a new model
            print("=" * 80)
            print("TRAINING MODE: Training a new model...")
            print("=" * 80)
            start_time = time.time()
            
            model, thresholds, label_names = train_and_tune()
            
            end_time = time.time()
            print(f"\nTotal training time: {(end_time - start_time) / 60:.2f} minutes")
        else:
            # Inference mode: Load existing model
            print("=" * 80)
            print("INFERENCE MODE: Loading saved model...")
            print("=" * 80)
            model, thresholds, label_names = load_trained_model()
        
        # Test prediction with example
        print("\n" + "=" * 80)
        print("TESTING PREDICTION")
        print("=" * 80)
        test_smiles = "CC1(C)SCC(N1C(=O)OCC2=CC=CC=C2)C(=O)O"  # Penicillin-like structure
        print(f"SMILES: {test_smiles}")
        
        predictions = predict_side_effects(test_smiles, model, thresholds, label_names)
        
        if predictions:
            print(f"\nPredicted side effects (top {len(predictions)}):")
            for i, (side_effect, prob, threshold) in enumerate(predictions, 1):
                print(f"{i:2d}. {side_effect:30s} (prob: {prob:.3f}, threshold: {threshold:.3f})")
        else:
            print("No side effects predicted above threshold.")
        
        print("\n" + "=" * 80)
        if TRAIN_MODE:
            print("Training complete! Set TRAIN_MODE = False to use the saved model.")
        else:
            print("Prediction complete! To retrain, set TRAIN_MODE = True.")
        print("=" * 80)
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()