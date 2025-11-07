# app/flask_app.py - BULLETPROOF VERSION

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import librosa
import pickle
import tensorflow as tf
from xgboost import XGBClassifier
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import traceback
from config import (SAMPLE_RATE, N_MFCC, UPLOAD_FOLDER, MAX_CONTENT_LENGTH,
                    DNN_MODEL_PATH, DNN_SCALER_PATH, XGBOOST_MODEL_PATH,
                    FLASK_DEBUG, FLASK_PORT)
from src.feature_extraction import AudioFeatureExtractor

# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# ============================================================================
# CORS & Headers
# ============================================================================

@app.after_request
def add_headers(response):
    """Add headers for cross-origin requests"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS, HEAD'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# ============================================================================
# GLOBAL MODELS
# ============================================================================

dnn_model = None
dnn_scaler = None
xgb_model = None
feature_extractor = None

def load_models():
    """Load all models at startup"""
    global dnn_model, dnn_scaler, xgb_model, feature_extractor
    
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*20 + "LOADING AI MODELS" + " "*32 + "║")
    print("╚" + "="*68 + "╝\n")
    
    try:
        # Load DNN
        print("[1/4] Loading DNN model...", end=" ", flush=True)
        dnn_model = tf.keras.models.load_model(str(DNN_MODEL_PATH))
        print("✅")
        
        # Load DNN Scaler
        print("[2/4] Loading DNN scaler...", end=" ", flush=True)
        with open(str(DNN_SCALER_PATH), 'rb') as f:
            dnn_scaler = pickle.load(f)
        print("✅")
        
        # Load XGBoost
        print("[3/4] Loading XGBoost model...", end=" ", flush=True)
        xgb_model = XGBClassifier()
        xgb_model.load_model(str(XGBOOST_MODEL_PATH))
        print("✅")
        
        # Initialize feature extractor
        print("[4/4] Initializing feature extractor...", end=" ", flush=True)
        feature_extractor = AudioFeatureExtractor(sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        print("✅")
        
        print("\n" + "="*70)
        print("  ✅ ALL MODELS LOADED SUCCESSFULLY!")
        print("="*70 + "\n")
        
        return True
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        print(traceback.format_exc())
        return False

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve homepage"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET', 'POST', 'OPTIONS'])
def health():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        models_ready = all([
            dnn_model is not None,
            dnn_scaler is not None,
            xgb_model is not None,
            feature_extractor is not None
        ])
        
        return jsonify({
            'status': 'healthy',
            'models_loaded': models_ready
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Main prediction endpoint"""
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 204
    
    print("\n" + "-"*70)
    print("  📍 PREDICTION REQUEST RECEIVED")
    print("-"*70)
    
    try:
        # =====================================================
        # REQUEST VALIDATION
        # =====================================================
        
        # Check if file in request
        if 'file' not in request.files:
            print("❌ No file in request.files")
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if not file or file.filename == '':
            print("❌ Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        print(f"📄 Filename: '{file.filename}'")
        print(f"📊 Content-Type: '{file.content_type}'")
        
        # =====================================================
        # FILE EXTENSION CHECK - BULLETPROOF
        # =====================================================
        
        allowed_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
        
        # Get filename safely
        filename = str(file.filename) if file.filename else ''
        print(f"📄 Filename string: '{filename}'")
        
        # Check if filename has extension
        if '.' not in filename:
            print("❌ No extension in filename")
            return jsonify({
                'success': False,
                'error': 'File has no extension'
            }), 400
        
        # Extract extension (with dot, lowercase)
        file_ext = '.' + filename.rsplit('.', 1).lower()
        print(f"🔍 Extension: '{file_ext}'")
        print(f"✓ Checking against: {allowed_extensions}")
        
        # Validate extension
        if file_ext not in allowed_extensions:
            print(f"❌ Invalid extension: '{file_ext}'")
            return jsonify({
                'success': False,
                'error': f'Invalid format: {file_ext}. Supported: {", ".join(sorted(allowed_extensions))}'
            }), 400
        
        print(f"✅ Extension valid: '{file_ext}'")
        
        # =====================================================
        # MODEL CHECK
        # =====================================================
        
        if dnn_model is None or xgb_model is None:
            print("❌ Models not loaded!")
            return jsonify({
                'success': False,
                'error': 'AI models not loaded. Server error.'
            }), 500
        
        # =====================================================
        # SAVE FILE
        # =====================================================
        
        filename_safe = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename_safe)
        
        print(f"💾 Saving to: {filepath}")
        file.save(filepath)
        print("✅ File saved")
        
        try:
            # =====================================================
            # EXTRACT FEATURES
            # =====================================================
            
            print("🎵 Extracting audio features...")
            # After this line:
            features_dict = feature_extractor.extract_features_from_file(filepath)
            if not features_dict:
                print("❌ Feature extraction failed")
                return jsonify({
                    'success': False,
                    'error': 'Failed to extract features from audio'
                }), 400

        # ✅ ADD THIS: Verify features are scalars
            print(f"📊 Checking {len(features_dict)} features...")
            for key, value in features_dict.items():
                if isinstance(value, (list, np.ndarray)):
                    print(f"❌ Feature '{key}' is not a scalar: {type(value)}, value: {value}")
                    return jsonify({
                        'success': False,
                        'error': f'Feature extraction error: {key} is not a scalar'
                    }), 500
                if not isinstance(value, (int, float, np.integer, np.floating)):
                    print(f"❌ Feature '{key}' is not numeric: {type(value)}")
                    return jsonify({
                        'success': False,
                        'error': f'Feature extraction error: {key} is not numeric'
                    }), 500

            print("✅ All features are valid scalars")

            # Then continue with:
            X = np.array(list(features_dict.values())).reshape(1, -1)

            print(f"📊 Feature matrix shape: {X.shape}")
            
            # =====================================================
            # DNN PREDICTION
            # =====================================================
            
            print("🧠 Running DNN model...")
            X_scaled = dnn_scaler.transform(X)
            dnn_pred_raw = dnn_model.predict(X_scaled, verbose=0)
            dnn_pred = float(dnn_pred_raw)
            print(f"   ✅ DNN score: {dnn_pred:.4f}")
            
            # =====================================================
            # XGBOOST PREDICTION
            # =====================================================
            
            print("🌲 Running XGBoost model...")
            xgb_pred_raw = xgb_model.predict_proba(X)
            xgb_pred = float(xgb_pred_raw)
            print(f"   ✅ XGBoost score: {xgb_pred:.4f}")
            
            # =====================================================
            # ENSEMBLE PREDICTION
            # =====================================================
            
            print("🤝 Ensemble voting...")
            ensemble_pred = (dnn_pred + xgb_pred) / 2
            is_phishing = ensemble_pred > 0.5
            confidence = max(ensemble_pred, 1 - ensemble_pred) * 100
            
            print(f"   ✅ Ensemble score: {ensemble_pred:.4f}")
            print(f"   ✅ Classification: {'🚨 PHISHING' if is_phishing else '✅ LEGITIMATE'}")
            print(f"   ✅ Confidence: {confidence:.2f}%")
            
            # =====================================================
            # BUILD RESPONSE
            # =====================================================
            
            response_data = {
                'success': True,
                'classification': 'PHISHING DETECTED ⚠️' if is_phishing else '✅ LEGITIMATE CALL',
                'is_phishing': bool(is_phishing),
                'confidence': round(confidence, 2),
                'scores': {
                    'dnn': round(dnn_pred, 4),
                    'xgboost': round(xgb_pred, 4),
                    'ensemble': round(ensemble_pred, 4)
                }
            }
            
            print("\n" + "✅"*35)
            print("PREDICTION SUCCESSFUL")
            print("✅"*35 + "\n")
            
            return jsonify(response_data), 200
            
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
            print(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Prediction error: {str(e)}'
            }), 500
        
        finally:
            # Cleanup
            if os.path.exists(filepath):
                os.remove(filepath)
                print("🗑️  Temporary file cleaned up")
    
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*8 + "🔐 VOICE PHISHING DETECTION SYSTEM - FLASK APP" + " "*14 + "║")
    print("╚" + "="*68 + "╝")
    
    # Create upload folder
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Load models
    if not load_models():
        print("\n⚠️  WARNING: Models failed to load!")
        print("Run training scripts:")
        print("  python src/04_train_dnn.py")
        print("  python src/05_train_xgboost.py")
    
    # Start Flask
    print(f"\n🚀 Starting Flask Server...")
    print(f"📍 URL: http://localhost:{FLASK_PORT}")
    print(f"🌐 Access: http://0.0.0.0:{FLASK_PORT}")
    print("\n⏹️  Press CTRL+C to stop\n")
    
    app.run(debug=FLASK_DEBUG, port=FLASK_PORT, host='0.0.0.0', use_reloader=False)
