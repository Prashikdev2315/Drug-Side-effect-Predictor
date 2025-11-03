"""
Drug Name to SMILES Lookup Utility
Converts drug names to SMILES strings using PubChem and ChEMBL APIs.
"""

import requests
import time
from urllib.parse import quote_plus

REQUEST_TIMEOUT = 10
SLEEP_BETWEEN_REQUESTS = 0.3
MAX_RETRIES = 2


def get_smiles_from_pubchem(drug_name):
    """
    Query PubChem for SMILES string by drug name.
    Returns (smiles, source) or (None, error_msg).
    """
    try:
        # Try PubChem REST API
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote_plus(drug_name)}/property/CanonicalSMILES/JSON"
        
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                return smiles, 'PubChem'
        
        return None, 'Not found in PubChem'
        
    except Exception as e:
        return None, f'PubChem error: {str(e)}'


def get_smiles_from_chembl(drug_name):
    """
    Query ChEMBL for SMILES string by drug name.
    Returns (smiles, source) or (None, error_msg).
    """
    try:
        base_url = "https://www.ebi.ac.uk/chembl/api/data/molecule"
        
        # Try exact preferred name match first
        params = {"pref_name__iexact": drug_name, "format": "json"}
        response = requests.get(base_url, params=params, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("molecules"):
                mol = data["molecules"][0]
                if mol.get("molecule_structures"):
                    smiles = mol["molecule_structures"].get("canonical_smiles")
                    if smiles:
                        return smiles, 'ChEMBL (exact)'
        
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        
        # Try synonym search
        params = {"molecule_synonyms__icontains": drug_name, "format": "json"}
        response = requests.get(base_url, params=params, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("molecules"):
                mol = data["molecules"][0]
                if mol.get("molecule_structures"):
                    smiles = mol["molecule_structures"].get("canonical_smiles")
                    if smiles:
                        return smiles, 'ChEMBL (synonym)'
        
        return None, 'Not found in ChEMBL'
        
    except Exception as e:
        return None, f'ChEMBL error: {str(e)}'


def drug_name_to_smiles(drug_name):
    """
    Convert drug name to SMILES string.
    Tries PubChem first, then ChEMBL if needed.
    
    Args:
        drug_name (str): Name of the drug (e.g., 'aspirin', 'ibuprofen')
    
    Returns:
        dict: {
            'success': bool,
            'smiles': str or None,
            'source': str (e.g., 'PubChem', 'ChEMBL'),
            'original_name': str,
            'error': str or None
        }
    """
    drug_name = drug_name.strip()
    
    if not drug_name:
        return {
            'success': False,
            'smiles': None,
            'source': None,
            'original_name': drug_name,
            'error': 'Drug name is empty'
        }
    
    # Try PubChem first (usually faster and more reliable)
    smiles, source = get_smiles_from_pubchem(drug_name)
    if smiles:
        return {
            'success': True,
            'smiles': smiles,
            'source': source,
            'original_name': drug_name,
            'error': None
        }
    
    # If PubChem fails, try ChEMBL
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    smiles, source = get_smiles_from_chembl(drug_name)
    if smiles:
        return {
            'success': True,
            'smiles': smiles,
            'source': source,
            'original_name': drug_name,
            'error': None
        }
    
    # Both failed
    return {
        'success': False,
        'smiles': None,
        'source': None,
        'original_name': drug_name,
        'error': f'Drug "{drug_name}" not found in PubChem or ChEMBL databases'
    }


if __name__ == '__main__':
    # Test the function
    test_drugs = ['aspirin', 'ibuprofen', 'caffeine', 'paracetamol', 'metformin']
    
    print("Testing drug name to SMILES conversion:\n")
    for drug in test_drugs:
        result = drug_name_to_smiles(drug)
        if result['success']:
            print(f"✓ {drug.capitalize()}")
            print(f"  Source: {result['source']}")
            print(f"  SMILES: {result['smiles'][:50]}...")
        else:
            print(f"✗ {drug.capitalize()}: {result['error']}")
        print()
