// app/static/js/app.js
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const resultSection = document.getElementById('resultSection');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('errorMessage');

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        analyzeFile(files);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        analyzeFile(e.target.files);
    }
});

async function analyzeFile(file) {
    // Validation
    const validTypes = ['audio/mpeg', 'audio/wav', 'audio/flac', 'audio/ogg'];
    const validExtensions = ['.mp3', '.wav', '.flac', '.ogg'];
    
    const hasValidType = validTypes.some(type => file.type.startsWith(type));
    const hasValidExt = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    
    if (!hasValidType && !hasValidExt) {
        showError('Invalid file format. Please upload MP3, WAV, FLAC, or OGG.');
        return;
    }
    
    if (file.size > 16 * 1024 * 1024) {
        showError('File too large. Maximum size is 16MB.');
        return;
    }
    
    // Show loading
    resultSection.style.display = 'none';
    errorMessage.style.display = 'none';
    loading.style.display = 'block';
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        loading.style.display = 'none';
        
        if (!response.ok || !data.success) {
            showError(data.error || 'Analysis failed');
            return;
        }
        
        displayResults(data);
    } catch (error) {
        loading.style.display = 'none';
        showError('Network error: ' + error.message);
    }
}

function displayResults(data) {
    const card = document.getElementById('resultCard');
    const status = document.getElementById('resultStatus');
    const classification = document.getElementById('classification');
    const confidence = document.getElementById('confidence');
    const dnnScore = document.getElementById('dnnScore');
    const xgbScore = document.getElementById('xgbScore');
    const ensembleScore = document.getElementById('ensembleScore');
    
    const isPhishing = data.is_phishing;
    const className = isPhishing ? 'phishing' : 'legitimate';
    
    card.className = 'result-card ' + className;
    status.className = 'result-status ' + className;
    status.textContent = data.classification;
    
    classification.textContent = data.classification;
    confidence.textContent = data.confidence + '%';
    
    dnnScore.textContent = data.scores.dnn.toFixed(4);
    xgbScore.textContent = data.scores.xgboost.toFixed(4);
    ensembleScore.textContent = data.scores.ensemble.toFixed(4);
    
    resultSection.style.display = 'block';
}

function showError(message) {
    errorMessage.textContent = '❌ ' + message;
    errorMessage.style.display = 'block';
}

function reset() {
    fileInput.value = '';
    resultSection.style.display = 'none';
    errorMessage.style.display = 'none';
    loading.style.display = 'none';
}
