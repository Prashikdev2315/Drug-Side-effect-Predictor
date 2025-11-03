# Setup script for Drug Side-Effect Prediction Project
# Run this script to install all dependencies

Write-Host "=" -NoNewline; Write-Host ("=" * 79)
Write-Host "Drug Side-Effect Prediction - Setup Script"
Write-Host "=" -NoNewline; Write-Host ("=" * 79)

# Check Python version
Write-Host "`nChecking Python version..."
$pythonVersion = python --version 2>&1
Write-Host "Found: $pythonVersion"

# Check if conda is available
Write-Host "`nChecking for conda..."
$condaExists = Get-Command conda -ErrorAction SilentlyContinue
if ($condaExists) {
    Write-Host "✓ Conda found!"
    Write-Host "`nRecommended: Use conda for RDKit installation"
    Write-Host "`nTo set up with conda, run these commands:"
    Write-Host "  conda create -n drug-adr python=3.10 -y"
    Write-Host "  conda activate drug-adr"
    Write-Host "  conda install -c conda-forge rdkit -y"
    Write-Host "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    Write-Host "  pip install -r requirements.txt"
} else {
    Write-Host "⚠ Conda not found (OK, will use pip)"
}

# Ask user if they want to install dependencies now
Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline; Write-Host ("=" * 79)
$response = Read-Host "Do you want to install dependencies with pip now? (y/n)"

if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host "`nInstalling dependencies..."
    
    # Upgrade pip
    Write-Host "`n[1/4] Upgrading pip..."
    python -m pip install --upgrade pip
    
    # Install PyTorch (CPU version)
    Write-Host "`n[2/4] Installing PyTorch (CPU)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Install requirements
    Write-Host "`n[3/4] Installing other dependencies..."
    pip install -r requirements.txt
    
    # Note about RDKit
    Write-Host "`n[4/4] Checking RDKit..."
    $rdkitTest = python -c "import rdkit" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠ RDKit installation failed with pip."
        Write-Host "Please install RDKit manually:"
        Write-Host "  Option 1 (Conda): conda install -c conda-forge rdkit"
        Write-Host "  Option 2 (Pip): pip install rdkit-pypi"
    } else {
        Write-Host "✓ RDKit installed successfully!"
    }
    
    Write-Host "`n" -NoNewline
    Write-Host "=" -NoNewline; Write-Host ("=" * 79)
    Write-Host "Installation complete!"
    Write-Host "=" -NoNewline; Write-Host ("=" * 79)
    Write-Host "`nNext steps:"
    Write-Host "  1. Run validation: python validate_setup.py"
    Write-Host "  2. Train model: python model.py"
    Write-Host "  3. Make predictions: python predict.py --smiles 'YOUR_SMILES_HERE'"
    
} else {
    Write-Host "`nSkipping installation."
    Write-Host "To install manually, see README.md"
}

Write-Host ""
