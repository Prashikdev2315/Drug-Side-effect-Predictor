"""
Flask Web Application for Drug Side Effect Prediction
Provides a modern web interface for predicting drug side effects from SMILES strings.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
import json
import os
from datetime import datetime

# Import from your existing model
from model import MLP_MultiLabel, DEVICE, CKPT_DIR, INPUT_DIM, HIDDEN_DIMS, DROPOUT
from predict import (
    load_model, 
    predict_side_effects,
    smiles_to_combined_features,
    smiles_to_ecfp_vector
)
from rdkit import Chem
from drug_lookup import drug_name_to_smiles

app = Flask(__name__)

# Global variables to store loaded model
model = None
thresholds = None
label_names = None
device = None
use_combined = None

def initialize_model():
    """Load model on startup."""
    global model, thresholds, label_names, device, use_combined
    print("Loading model...")
    model, thresholds, label_names, device, use_combined = load_model()
    print("✓ Model loaded successfully!")

@app.route('/')
def home():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests - accepts both SMILES and drug names."""
    try:
        data = request.get_json()
        input_value = data.get('input', '').strip()
        top_k = int(data.get('top_k', 10))
        
        if not input_value:
            return jsonify({
                'success': False,
                'error': 'Please provide a SMILES string or drug name'
            })
        
        # Check if input is a SMILES string or drug name
        smiles = None
        input_source = 'SMILES (direct)'
        
        # Try to parse as SMILES first
        mol = Chem.MolFromSmiles(input_value)
        
        if mol is None:
            # Not a valid SMILES, try to look up as drug name
            lookup_result = drug_name_to_smiles(input_value)
            
            if lookup_result['success']:
                smiles = lookup_result['smiles']
                input_source = f"Drug name → {lookup_result['source']}"
                mol = Chem.MolFromSmiles(smiles)
            else:
                return jsonify({
                    'success': False,
                    'error': f'Invalid input. Not a valid SMILES string and {lookup_result["error"]}'
                })
        else:
            # Valid SMILES
            smiles = input_value
        
        # Validate final SMILES
        if mol is None:
            return jsonify({
                'success': False,
                'error': 'Could not process the molecular structure'
            })
        
        # Get molecular properties
        from rdkit.Chem import Descriptors
        mol_weight = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        num_atoms = mol.GetNumAtoms()
        
        # Make prediction
        predictions = predict_side_effects(
            smiles, model, thresholds, label_names, device,
            top_k=top_k, use_combined=use_combined
        )
        
        # Format results
        results = []
        for side_effect, prob, threshold in predictions:
            results.append({
                'name': side_effect,
                'probability': float(prob),
                'threshold': float(threshold),
                'confidence': 'High' if prob > 0.7 else 'Medium' if prob > 0.5 else 'Low'
            })
        
        return jsonify({
            'success': True,
            'input': input_value,
            'smiles': smiles,
            'input_source': input_source,
            'molecular_properties': {
                'molecular_weight': round(mol_weight, 2),
                'logp': round(logp, 2),
                'num_atoms': num_atoms
            },
            'predictions': results,
            'total_predicted': len(results),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/stats')
def stats():
    """Get model statistics."""
    try:
        # Load test metrics
        with open(os.path.join(CKPT_DIR, 'test_metrics.json'), 'r') as f:
            metrics = json.load(f)
        
        return jsonify({
            'success': True,
            'metrics': {
                'micro_f1': round(metrics['micro_f1'], 4),
                'macro_f1': round(metrics['macro_f1'], 4),
                'precision': round(metrics['micro_precision'], 4),
                'recall': round(metrics['micro_recall'], 4),
                'accuracy': round(metrics['per_label_accuracy'], 4)
            },
            'model_info': {
                'total_side_effects': len(label_names),
                'input_features': INPUT_DIM,
                'feature_type': 'ECFP + RDKit Descriptors' if use_combined else 'ECFP Only'
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/examples')
def examples():
    """Get example SMILES."""
    example_drugs = [
        {
            'name': 'Aspirin',
            'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'description': 'Common pain reliever and anti-inflammatory'
        },
        {
            'name': 'Ibuprofen',
            'smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
            'description': 'Non-steroidal anti-inflammatory drug (NSAID)'
        },
        {
            'name': 'Caffeine',
            'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'description': 'Stimulant found in coffee and tea'
        },
        {
            'name': 'Paracetamol (Acetaminophen)',
            'smiles': 'CC(=O)NC1=CC=C(C=C1)O',
            'description': 'Common pain reliever and fever reducer'
        },
        {
            'name': 'Penicillin V',
            'smiles': 'CC1(C)SC2C(NC(=O)COC3=CC=CC=C3)C(=O)N2C1C(=O)O',
            'description': 'Antibiotic medication'
        }
    ]
    
    return jsonify({
        'success': True,
        'examples': example_drugs
    })

if __name__ == '__main__':
    # Initialize model before starting server
    initialize_model()
    
    # Run the app
    print("\n" + "=" * 80)
    print("🚀 Starting Drug Side Effect Prediction Web App")
    print("=" * 80)
    print("\n📱 Open your browser and go to: http://localhost:5000")
    print("\n✓ Model loaded and ready for predictions!")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
