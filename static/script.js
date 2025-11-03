// Global state
let examples = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadExamples();
    loadStats();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    const predictBtn = document.getElementById('predictBtn');
    const smilesInput = document.getElementById('smilesInput');
    
    predictBtn.addEventListener('click', handlePredict);
    
    // Allow Enter key to trigger prediction
    smilesInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handlePredict();
        }
    });
    
    // Smooth scroll for nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            if (link.getAttribute('href').startsWith('#')) {
                e.preventDefault();
                const targetId = link.getAttribute('href').slice(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({ behavior: 'smooth' });
                }
                
                // Update active link
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            }
        });
    });
}

// Load example drugs
async function loadExamples() {
    try {
        const response = await fetch('/examples');
        const data = await response.json();
        
        if (data.success) {
            examples = data.examples;
            displayExamples(data.examples);
        }
    } catch (error) {
        console.error('Error loading examples:', error);
    }
}

// Display examples
function displayExamples(examples) {
    const examplesList = document.getElementById('examplesList');
    examplesList.innerHTML = examples.map(example => `
        <div class="example-card" onclick="selectExample('${example.name}')">
            <h4><i class="fas fa-prescription-bottle"></i> ${example.name}</h4>
            <p>${example.description}</p>
        </div>
    `).join('');
}

// Select example
function selectExample(drugName) {
    document.getElementById('smilesInput').value = drugName;
    // Scroll to results
    document.getElementById('resultsContainer').scrollIntoView({ behavior: 'smooth' });
}

// Load model statistics
async function loadStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        
        if (data.success) {
            // Update hero stats
            document.getElementById('totalSideEffects').textContent = data.model_info.total_side_effects;
            document.getElementById('modelAccuracy').textContent = (data.metrics.micro_f1 * 100).toFixed(1) + '%';
            document.getElementById('modelRecall').textContent = (data.metrics.recall * 100).toFixed(1) + '%';
            
            // Update metrics section
            document.getElementById('microF1').textContent = data.metrics.micro_f1.toFixed(4);
            document.getElementById('macroF1').textContent = data.metrics.macro_f1.toFixed(4);
            document.getElementById('precision').textContent = data.metrics.precision.toFixed(4);
            document.getElementById('recall').textContent = data.metrics.recall.toFixed(4);
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Handle prediction
async function handlePredict() {
    const smilesInput = document.getElementById('smilesInput');
    const topKSelect = document.getElementById('topKSelect');
    const resultsContainer = document.getElementById('resultsContainer');
    const loadingSpinner = document.getElementById('loadingSpinner');
    
    const input = smilesInput.value.trim();
    const topK = parseInt(topKSelect.value);
    
    if (!input) {
        showError('Please enter a drug name or SMILES string');
        return;
    }
    
    // Show loading
    loadingSpinner.style.display = 'block';
    resultsContainer.innerHTML = '';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ input: input, top_k: topK })
        });
        
        const data = await response.json();
        
        // Hide loading
        loadingSpinner.style.display = 'none';
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error);
        }
    } catch (error) {
        loadingSpinner.style.display = 'none';
        showError('An error occurred. Please try again.');
        console.error('Prediction error:', error);
    }
}

// Display prediction results
function displayResults(data) {
    const resultsContainer = document.getElementById('resultsContainer');
    
    const inputInfo = data.input_source ? `
        <div class="input-source-info">
            <i class="fas fa-info-circle"></i>
            <strong>Input:</strong> ${data.input} 
            <span class="badge badge-info">${data.input_source}</span>
        </div>
    ` : '';
    
    const molecularInfo = `
        <div class="molecular-info">
            <h4><i class="fas fa-atom"></i> Molecular Properties</h4>
            ${data.smiles !== data.input ? `<p class="smiles-display"><strong>SMILES:</strong> <code>${data.smiles}</code></p>` : ''}
            <div class="molecular-props">
                <div class="prop-item">
                    <span class="prop-label">Molecular Weight</span>
                    <span class="prop-value">${data.molecular_properties.molecular_weight} g/mol</span>
                </div>
                <div class="prop-item">
                    <span class="prop-label">LogP</span>
                    <span class="prop-value">${data.molecular_properties.logp}</span>
                </div>
                <div class="prop-item">
                    <span class="prop-label">Number of Atoms</span>
                    <span class="prop-value">${data.molecular_properties.num_atoms}</span>
                </div>
            </div>
        </div>
    `;
    
    const predictionsList = data.predictions.length > 0 ? `
        <div class="predictions-header" style="margin-bottom: 1.5rem;">
            <h4 style="color: var(--dark); margin-bottom: 0.5rem;">
                <i class="fas fa-exclamation-triangle"></i> 
                Predicted Side Effects (${data.total_predicted})
            </h4>
            <p style="color: var(--dark-light); font-size: 0.9rem;">
                Showing top ${data.predictions.length} predictions based on probability
            </p>
        </div>
        <div class="predictions-list">
            ${data.predictions.map((pred, index) => `
                <div class="prediction-item">
                    <div class="prediction-header">
                        <span class="prediction-name">
                            <span style="color: var(--dark-light); margin-right: 0.5rem;">${index + 1}.</span>
                            ${pred.name}
                        </span>
                        <span class="confidence-badge confidence-${pred.confidence.toLowerCase()}">
                            ${pred.confidence}
                        </span>
                    </div>
                    <div class="prediction-details">
                        <div class="detail-item">
                            <span class="detail-label">Probability</span>
                            <span class="detail-value">${(pred.probability * 100).toFixed(2)}%</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Threshold</span>
                            <span class="detail-value">${(pred.threshold * 100).toFixed(2)}%</span>
                        </div>
                    </div>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: ${pred.probability * 100}%"></div>
                    </div>
                </div>
            `).join('')}
        </div>
    ` : `
        <div class="empty-state">
            <i class="fas fa-check-circle"></i>
            <h4>No Significant Side Effects Predicted</h4>
            <p>The model did not predict any side effects above the confidence threshold for this compound.</p>
        </div>
    `;
    
    resultsContainer.innerHTML = inputInfo + molecularInfo + predictionsList;
}

// Show error message
function showError(message) {
    const resultsContainer = document.getElementById('resultsContainer');
    resultsContainer.innerHTML = `
        <div class="empty-state">
            <i class="fas fa-exclamation-circle" style="color: var(--danger-color);"></i>
            <h4 style="color: var(--danger-color);">Error</h4>
            <p>${message}</p>
        </div>
    `;
}

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
