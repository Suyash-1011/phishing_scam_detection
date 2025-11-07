// app/static/js/app.js - FINAL BULLETPROOF VERSION

console.log('✅ JavaScript loaded');

// Get DOM elements
const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('audioFile');
const loading = document.getElementById('loading');
const errorMsg = document.getElementById('errorMsg');
const results = document.getElementById('results');
const statusEl = document.getElementById('status');

console.log('✅ DOM elements found:', {
    form: !!form,
    fileInput: !!fileInput,
    loading: !!loading,
    errorMsg: !!errorMsg,
    results: !!results,
    statusEl: !!statusEl
});

// On page load
window.addEventListener('load', () => {
    console.log('✅ Page fully loaded');
    checkHealth();
});

// Health check
async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        if (res.ok) {
            statusEl.textContent = '🟢 Ready';
            console.log('✅ Server ready');
        }
    } catch (e) {
        statusEl.textContent = '🔴 Error';
        console.error('Health check failed:', e);
    }
}

// ============================================================================
// FORM SUBMISSION - THE MAIN HANDLER
// ============================================================================

if (form) {
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        console.log('\n' + '='*70);
        console.log('FORM SUBMITTED');
        console.log('='*70);
        
        try {
            // Get the files from input
            const files = fileInput.files;
            console.log('Files in input:', files.length);
            
            if (!files || files.length === 0) {
                console.error('No files selected');
                showError('Please select a file');
                return;
            }
            
            // Get first file
            const file = files;
            console.log('✅ File object obtained');
            console.log('   Name:', file.name);
            console.log('   Size:', file.size, 'bytes');
            console.log('   Type:', file.type || 'undefined');
            
            // Validate file
            if (!validateFile(file)) {
                return;
            }
            
            // Upload file
            await uploadFile(file);
            
        } catch (error) {
            console.error('❌ Error in form submission:', error);
            showError('Error: ' + error.message);
        }
    });
} else {
    console.error('❌ Form not found!');
}

// ============================================================================
// FILE VALIDATION
// ============================================================================

function validateFile(file) {
    console.log('\nValidating file...');
    
    // Check if file exists
    if (!file) {
        console.error('File is null or undefined');
        showError('File is invalid');
        return false;
    }
    
    // Check file name exists
    if (!file.name) {
        console.error('File name is undefined');
        showError('File name is missing');
        return false;
    }
    
    // Get extension safely
    let ext = '';
    try {
        const lastDot = file.name.lastIndexOf('.');
        if (lastDot === -1) {
            console.error('No extension found');
            showError('File has no extension');
            return false;
        }
        ext = file.name.substring(lastDot).toLowerCase();
    } catch (e) {
        console.error('Error getting extension:', e);
        showError('Error validating file name');
        return false;
    }
    
    console.log('Extension:', ext);
    
    // Check if extension is valid
    const validExt = ['.mp3', '.wav', '.flac', '.ogg', '.m4a'];
    if (!validExt.includes(ext)) {
        console.error('Invalid extension:', ext);
        showError(`Invalid format: ${ext}. Supported: ${validExt.join(', ')}`);
        return false;
    }
    
    console.log('✅ Extension valid');
    
    // Check file size
    if (!file.size) {
        console.error('File size is 0 or undefined');
        showError('File is empty');
        return false;
    }
    
    const maxSize = 16 * 1024 * 1024; // 16MB
    if (file.size > maxSize) {
        console.error('File too large:', file.size);
        showError(`File too large: ${(file.size / 1024 / 1024).toFixed(2)}MB (max 16MB)`);
        return false;
    }
    
    console.log('✅ Size valid:', (file.size / 1024).toFixed(2), 'KB');
    console.log('✅ All validation passed');
    
    return true;
}

// ============================================================================
// UPLOAD TO SERVER
// ============================================================================

async function uploadFile(file) {
    console.log('\nUploading file to server...');
    
    try {
        // Hide results
        results.style.display = 'none';
        errorMsg.style.display = 'none';
        
        // Show loading
        loading.style.display = 'block';
        console.log('⏳ Loading shown');
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        console.log('📦 FormData created with file');
        
        // Send to Flask
        console.log('📤 Sending POST to /api/predict');
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData,
            // Don't set headers - let browser set Content-Type
        });
        
        console.log('📬 Response received:', response.status, response.statusText);
        
        // Parse response
        const data = await response.json();
        console.log('📋 Response parsed:', data);
        
        // Hide loading
        loading.style.display = 'none';
        console.log('✅ Loading hidden');
        
        // Check if successful
        if (!response.ok) {
            console.error('❌ Response not ok:', response.status);
            showError(data.error || `Server error: ${response.status}`);
            return;
        }
        
        if (!data.success) {
            console.error('❌ Data success false:', data);
            showError(data.error || 'Analysis failed');
            return;
        }
        
        console.log('\n✅ ✅ ✅ UPLOAD SUCCESSFUL ✅ ✅ ✅');
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        loading.style.display = 'none';
        console.error('❌ Upload error:', error);
        showError('Error: ' + error.message);
    }
}

// ============================================================================
// DISPLAY RESULTS
// ============================================================================

function displayResults(data) {
    console.log('\nDisplaying results...');
    
    try {
        // Get elements
        const classEl = document.getElementById('classification');
        const confEl = document.getElementById('confidence');
        const dnnEl = document.getElementById('dnnScore');
        const xgbEl = document.getElementById('xgbScore');
        const ensEl = document.getElementById('ensembleScore');
        
        if (!classEl || !confEl || !dnnEl || !xgbEl || !ensEl) {
            console.error('Result elements not found');
            return;
        }
        
        // Set values
        classEl.textContent = data.classification;
        confEl.textContent = data.confidence + '%';
        dnnEl.textContent = data.scores.dnn.toFixed(4);
        xgbEl.textContent = data.scores.xgboost.toFixed(4);
        ensEl.textContent = data.scores.ensemble.toFixed(4);
        
        console.log('✅ Results set');
        
        // Show results
        results.style.display = 'block';
        console.log('✅ Results displayed');
        
    } catch (error) {
        console.error('❌ Error displaying results:', error);
        showError('Error displaying results: ' + error.message);
    }
}

// ============================================================================
// UTILITIES
// ============================================================================

function showError(msg) {
    console.error('🚨 ERROR:', msg);
    errorMsg.textContent = msg;
    errorMsg.style.display = 'block';
}

function reset() {
    console.log('🔄 Resetting form');
    fileInput.value = '';
    results.style.display = 'none';
    errorMsg.style.display = 'none';
    loading.style.display = 'none';
}

console.log('\n✅ ✅ ✅ JavaScript fully loaded and ready ✅ ✅ ✅\n');
