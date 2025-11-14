import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, MODEL_DIR
from src.utils import print_section, print_subsection, get_logger

logger = get_logger()

class LightGBMPhishingDetector:
    """LightGBM model for phishing detection"""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1  # Suppress warnings
        )
        logger.info("LightGBM model initialized")
    
    def train(self, X, y, cv_folds=5):
        """Train LightGBM model with cross-validation"""
        print_subsection("Training LightGBM")
        
        self.model.fit(X, y)
        logger.info("LightGBM model trained")
        
        # Cross-validation
        print_subsection("Cross-Validation")
        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring='f1')
        
        print(f"CV Scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        logger.info(f"Cross-validation complete: {cv_scores.mean():.4f}")
        
        # Feature importance
        print_subsection("Top 10 Important Features")
        feature_importance = self.model.feature_importances_
        feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(importance_df.head(10).to_string(index=False))
        
        return cv_scores
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability estimates"""
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, model_path):
        """Save model"""
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"LightGBM model saved to {model_path}")
    
    def load(self, model_path):
        """Load model"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"LightGBM model loaded from {model_path}")

def train_lightgbm_model():
    """Train LightGBM on features"""
    print_section("LIGHTGBM MODEL TRAINING")
    
    # Load pre-split train and test data
    print_subsection("Loading Pre-Split Data")
    train_path = PROCESSED_DIR / "train_features.csv"
    test_path = PROCESSED_DIR / "test_features.csv"
    
    # Check if files exist
    if not train_path.exists() or not test_path.exists():
        logger.error("Train/test features not found. Please run feature extraction first.")
        print("❌ Error: train_features.csv or test_features.csv not found")
        print("Please run: python src/feature_extraction.py")
        return None, None
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(['label', 'filename'], axis=1).values
    y_train = train_df['label'].values
    
    X_test = test_df.drop(['label', 'filename'], axis=1).values
    y_test = test_df['label'].values
    
    print(f"Training set: {X_train.shape}")
    print(f"  - Phishing: {(y_train == 1).sum()}")
    print(f"  - Legitimate: {(y_train == 0).sum()}")
    print(f"Test set: {X_test.shape}")
    print(f"  - Phishing: {(y_test == 1).sum()}")
    print(f"  - Legitimate: {(y_test == 0).sum()}")
    
    # Train model
    detector = LightGBMPhishingDetector(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    cv_scores = detector.train(X_train, y_train, cv_folds=5)
    
    # Evaluate on test set
    print_subsection("Test Set Evaluation")
    test_predictions = detector.predict(X_test)
    test_accuracy = (test_predictions == y_test).mean()
    print(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"LightGBM Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    model_path = MODEL_DIR / "lightgbm.pkl"
    detector.save(model_path)
    
    return detector, cv_scores

if __name__ == "__main__":
    train_lightgbm_model()