# src/05_train_xgboost.py
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import classification_report
import json
from pathlib import Path
from config import (PROCESSED_DIR, XGBOOST_MODEL_PATH,
                    XGBOOST_N_ESTIMATORS, XGBOOST_MAX_DEPTH, 
                    XGBOOST_LEARNING_RATE)
from src.utils import print_section, print_subsection, get_logger

logger = get_logger()

class XGBoostPhishingDetector:
    """XGBoost model for phishing detection"""
    
    def __init__(self, n_estimators=XGBOOST_N_ESTIMATORS, 
                 max_depth=XGBOOST_MAX_DEPTH,
                 learning_rate=XGBOOST_LEARNING_RATE):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            verbosity=1
        )
        logger.info("XGBoost model initialized")
    
    def train(self, X, y, cv_folds=5):
        """Train XGBoost model with cross-validation"""
        print_subsection("Training XGBoost")
        
        self.model.fit(X, y, verbose=True)
        logger.info("XGBoost model trained")
        
        # Cross-validation
        print_subsection("Cross-Validation")
        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring='f1')
        
        print(f"CV Scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        logger.info(f"Cross-validation complete: {cv_scores.mean():.4f}")
        
        # Feature importance
        print_subsection("Top 10 Important Features")
        feature_importance = self.model.feature_importances_
        feature_names = self.get_feature_names()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(importance_df.head(10).to_string(index=False))
        
        return cv_scores
    
    def get_feature_names(self):
        """Get feature names from booster"""
        if hasattr(self.model, 'get_booster'):
            feature_names = self.model.get_booster().feature_names
            if feature_names:
                return feature_names
        return [f'feature_{i}' for i in range(self.model.n_features_in_)]
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability estimates"""
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, model_path=XGBOOST_MODEL_PATH):
        """Save model"""
        self.model.save_model(model_path)
        logger.info(f"XGBoost model saved to {model_path}")
    
    def load(self, model_path=XGBOOST_MODEL_PATH):
        """Load model"""
        self.model.load_model(model_path)
        logger.info(f"XGBoost model loaded from {model_path}")

def train_xgboost_model():
    """Train XGBoost on features"""
    print_section("XGBOOST MODEL TRAINING")
    
    # Load features
    print_subsection("Loading Features")
    features_path = PROCESSED_DIR / "features_augmented.csv"
    df = pd.read_csv(features_path)
    
    X = df.drop(['label', 'filename'], axis=1).values
    y = df['label'].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Phishing samples: {(y == 1).sum()}")
    print(f"Legitimate samples: {(y == 0).sum()}")
    
    # Train model
    detector = XGBoostPhishingDetector(
        n_estimators=XGBOOST_N_ESTIMATORS,
        max_depth=XGBOOST_MAX_DEPTH,
        learning_rate=XGBOOST_LEARNING_RATE
    )
    cv_scores = detector.train(X, y, cv_folds=5)
    
    # Save model
    detector.save()
    
    return detector, cv_scores

if __name__ == "__main__":
    train_xgboost_model()
