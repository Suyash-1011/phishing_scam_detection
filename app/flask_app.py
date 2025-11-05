# app/flask_app.py
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
from config import (SAMPLE_RATE, N_MFCC, UPLOAD_FOLDER, MAX_CONTENT_LENGTH,
                    DNN_MODEL_PATH, DNN_SCALER_PATH, XGBOOST_MODEL_PATH,
                    FLASK_DEBUG, FLASK_PORT)
from src.feature_extraction import AudioFeatureExtractor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Load models
print("Loading models...")
dnn_model = tf.keras.models.load_model(str(DNN_MODEL_PATH))
with open(str(DNN_SCALER_PATH), 'rb') as f:
    dnn_scaler = pickle.load(f)

xgb_model = XGBClassifier()
xgb_model.load_model(str(XGBOOST_MODEL_PATH))

feature_extractor = AudioFeatureExtractor(sr=SAMPLE_RATE, n_mfcc=N_MFCC)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
            return jsonify({'error': 'Invalid audio format. Supported: MP3, WAV, FLAC, OGG'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract features
            features_dict = feature_extractor.extract_features_from_file(filepath)
            if not features_dict:
                return jsonify({'error': 'Failed to extract features from audio'}), 400
            
            X = np.array(list(features_dict.values())).reshape(1, -1)
            
            # DNN prediction
            X_scaled = dnn_scaler.transform(X)
            dnn_pred = float(dnn_model.predict(X_scaled, verbose=0))
            
            # XGBoost prediction
            xgb_pred = float(xgb_model.predict_proba(X))
            
            # Ensemble prediction
            ensemble_pred = (dnn_pred + xgb_pred) / 2
            is_phishing = ensemble_pred > 0.5
            confidence = max(ensemble_pred, 1 - ensemble_pred) * 100
            
            return jsonify({
                'success': True,
                'classification': 'PHISHING DETECTED' if is_phishing else 'LEGITIMATE CALL',
                'is_phishing': bool(is_phishing),
                'confidence': round(confidence, 2),
                'scores': {
                    'dnn': round(dnn_pred, 4),
                    'xgboost': round(xgb_pred, 4),
                    'ensemble': round(ensemble_pred, 4)
                }
            })
        finally:
            # Cleanup
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch', methods=['POST'])
def batch_predict():
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files[]')
        results = []
        
        for file in files:
            if file and file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                try:
                    features_dict = feature_extractor.extract_features_from_file(filepath)
                    if not features_dict:
                        continue
                    
                    X = np.array(list(features_dict.values())).reshape(1, -1)
                    X_scaled = dnn_scaler.transform(X)
                    dnn_pred = float(dnn_model.predict(X_scaled, verbose=0))
                    xgb_pred = float(xgb_model.predict_proba(X))
                    ensemble_pred = (dnn_pred + xgb_pred) / 2
                    
                    results.append({
                        'filename': filename,
                        'classification': 'PHISHING' if ensemble_pred > 0.5 else 'LEGITIMATE',
                        'confidence': round(max(ensemble_pred, 1-ensemble_pred)*100, 2)
                    })
                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)
        
        return jsonify({'success': True, 'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'models': 'loaded'})

if __name__ == '__main__':
    app.run(debug=FLASK_DEBUG, port=FLASK_PORT, host='0.0.0.0')
