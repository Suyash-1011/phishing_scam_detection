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
                    DNN_MODEL_PATH, DNN_SCALER_PATH, XGBOOST_MODEL_PATH, MODEL_DIR,
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
rf_model = None
lgbm_model = None
feature_extractor = None

def load_models():
    """Load all models at startup"""
    global dnn_model, dnn_scaler, xgb_model, rf_model, lgbm_model, feature_extractor
    
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*20 + "LOADING AI MODELS" + " "*32 + "║")
    print("╚" + "="*68 + "╝\n")
    
    try:
        # Load DNN
        print("[1/6] Loading DNN model...", end=" ", flush=True)
        dnn_model = tf.keras.models.load_model(str(DNN_MODEL_PATH))
        print("✅")
        
        # Load DNN Scaler
        print("[2/6] Loading DNN scaler...", end=" ", flush=True)
        with open(str(DNN_SCALER_PATH), 'rb') as f:
            dnn_scaler = pickle.load(f)
        print("✅")
        
        # Load XGBoost
        print("[3/6] Loading XGBoost model...", end=" ", flush=True)
        xgb_model = XGBClassifier()
        xgb_model.load_model(str(XGBOOST_MODEL_PATH))
        print("✅")
        
        # Load Random Forest
        print("[4/6] Loading Random Forest model...", end=" ", flush=True)
        rf_path = MODEL_DIR / "random_forest.pkl"
        if rf_path.exists():
            with open(rf_path, 'rb') as f:
                rf_model = pickle.load(f)
            print("✅")
        else:
            rf_model = None
            print("⚠️  (not found, skipping)")
        
        # Load LightGBM
        print("[5/6] Loading LightGBM model...", end=" ", flush=True)
        lgbm_path = MODEL_DIR / "lightgbm.pkl"
        if lgbm_path.exists():
            with open(lgbm_path, 'rb') as f:
                lgbm_model = pickle.load(f)
            print("✅")
        else:
            lgbm_model = None
            print("⚠️  (not found, skipping)")
        
        # Initialize feature extractor
        print("[6/6] Initializing feature extractor...", end=" ", flush=True)
        feature_extractor = AudioFeatureExtractor(sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        print("✅")
        
        # Count loaded models
        n_models = sum([
            dnn_model is not None,
            xgb_model is not None,
            rf_model is not None,
            lgbm_model is not None
        ])
        
        print("\n" + "="*70)
        print(f"  ✅ {n_models}/4 MODELS LOADED SUCCESSFULLY!")
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
        
        n_models = sum([
            dnn_model is not None,
            xgb_model is not None,
            rf_model is not None,
            lgbm_model is not None
        ])
        
        return jsonify({
            'status': 'healthy',
            'models_loaded': models_ready,
            'total_models': n_models
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
        # FILE EXTENSION CHECK
        # =====================================================
        
        allowed_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
        filename = str(file.filename) if file.filename else ''
        
        if '.' not in filename:
            print("❌ No extension in filename")
            return jsonify({
                'success': False,
                'error': 'File has no extension'
            }), 400
        
        file_ext = '.' + filename.rsplit('.', 1)[-1].lower()
        
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
            print("❌ Core models not loaded!")
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
            features_dict = feature_extractor.extract_features_from_file(filepath)
            
            if not features_dict:
                print("❌ Feature extraction failed")
                return jsonify({
                    'success': False,
                    'error': 'Failed to extract features from audio'
                }), 400
            
            # Verify features are scalars
            for key, value in features_dict.items():
                if isinstance(value, (list, np.ndarray)):
                    print(f"❌ Feature '{key}' is not a scalar")
                    return jsonify({
                        'success': False,
                        'error': f'Feature extraction error: {key} is not a scalar'
                    }), 500
            
            print("✅ All features are valid scalars")
            
            X = np.array(list(features_dict.values())).reshape(1, -1)
            print(f"📊 Feature matrix shape: {X.shape}")
            
            # =====================================================
            # PREDICTIONS FROM ALL MODELS
            # =====================================================
            
            predictions = {}
            
            # DNN
            print("🧠 Running DNN model...")
            X_scaled = dnn_scaler.transform(X)
            dnn_pred_raw = dnn_model.predict(X_scaled, verbose=0)
            dnn_pred = float(dnn_pred_raw.flatten()[0])
            predictions['dnn'] = dnn_pred
            print(f"   ✅ DNN score: {dnn_pred:.4f}")
            
            # XGBoost
            print("🌲 Running XGBoost model...")
            xgb_pred_raw = xgb_model.predict_proba(X)
            xgb_pred = float(xgb_pred_raw[0][1])
            predictions['xgb'] = xgb_pred
            print(f"   ✅ XGBoost score: {xgb_pred:.4f}")
            
            # Random Forest
            if rf_model is not None:
                print("🌲 Running Random Forest model...")
                rf_pred_raw = rf_model.predict_proba(X)
                rf_pred = float(rf_pred_raw[0][1])
                predictions['rf'] = rf_pred
                print(f"   ✅ Random Forest score: {rf_pred:.4f}")
            
            # LightGBM
            if lgbm_model is not None:
                print("💡 Running LightGBM model...")
                lgbm_pred_raw = lgbm_model.predict_proba(X)
                lgbm_pred = float(lgbm_pred_raw[0][1])
                predictions['lgbm'] = lgbm_pred
                print(f"   ✅ LightGBM score: {lgbm_pred:.4f}")
            
            # =====================================================
            # ENSEMBLE PREDICTION
            # =====================================================
            
            print("🤝 Ensemble voting...")
            ensemble_pred = np.mean(list(predictions.values()))
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
                    'dnn': round(predictions.get('dnn', 0), 4),
                    'xgboost': round(predictions.get('xgb', 0), 4),
                    'random_forest': round(predictions.get('rf', 0), 4) if rf_model else None,
                    'lightgbm': round(predictions.get('lgbm', 0), 4) if lgbm_model else None,
                    'ensemble': round(ensemble_pred, 4)
                }
            }
            
            # Remove None values
            response_data['scores'] = {k: v for k, v in response_data['scores'].items() if v is not None}
            
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
        print("\n⚠️  WARNING: Some models failed to load!")
        print("Run training scripts:")
        print("  python src/04_train_dnn.py")
        print("  python src/05_train_xgboost.py")
        print("  python src/07_train_random_forest.py")
        print("  python src/08_train_lightgbm.py")
    
    # Start Flask
    print(f"\n🚀 Starting Flask Server...")
    print(f"📍 URL: http://localhost:{FLASK_PORT}")
    print(f"🌐 Access: http://0.0.0.0:{FLASK_PORT}")
    print("\n⏹️  Press CTRL+C to stop\n")
    
    app.run(debug=FLASK_DEBUG, port=FLASK_PORT, host='0.0.0.0', use_reloader=False)