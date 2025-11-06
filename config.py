# config.py
import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
PHISHING_DIR = DATA_DIR / "phishing"
LEGITIMATE_DIR = DATA_DIR / "legitimate"
AUGMENTED_DIR = DATA_DIR / "augmented"
PROCESSED_DIR = DATA_DIR / "processed"
TEST_DIR = DATA_DIR / "test"

# Model Directories
MODEL_DIR = PROJECT_ROOT / "models"
DNN_MODEL_PATH = MODEL_DIR / "dnn_model.h5"
DNN_SCALER_PATH = MODEL_DIR / "dnn_scaler.pkl"
XGBOOST_MODEL_PATH = MODEL_DIR / "xgboost_model.json"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_PATH = RESULTS_DIR / "metrics.csv"
PLOTS_DIR = RESULTS_DIR

# Logs
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "training.log"

# Audio Processing
SAMPLE_RATE = 16000
N_MFCC = 13
AUDIO_DURATION = 10  # seconds

# Augmentation
AUGMENT_FACTOR = 5  # Generate 5 augmented versions per original file

# Model Parameters
DNN_EPOCHS = 100
DNN_BATCH_SIZE = 8
DNN_VALIDATION_SPLIT = 0.2

XGBOOST_N_ESTIMATORS = 100
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.1

# Ensemble
ENSEMBLE_THRESHOLD = 0.5
DNN_WEIGHT = 0.5
XGBOOST_WEIGHT = 0.5

# Flask
FLASK_DEBUG = True
FLASK_PORT = 5001
UPLOAD_FOLDER = PROJECT_ROOT / "app" / "temp_uploads"
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR, UPLOAD_FOLDER]:
    directory.mkdir(parents=True, exist_ok=True)
