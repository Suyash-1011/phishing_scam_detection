# src/06_ensemble_eval.py - Part 1
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import (PROCESSED_DIR, MODEL_DIR, DNN_MODEL_PATH, DNN_SCALER_PATH,
                    XGBOOST_MODEL_PATH, RESULTS_DIR, ENSEMBLE_THRESHOLD,
                    DNN_WEIGHT, XGBOOST_WEIGHT)
from src.utils import print_section, print_subsection, save_dataframe, get_logger

logger = get_logger()

class EnsemblePhishingDetector:
    """Ensemble predictor combining DNN and XGBoost"""
    
    def __init__(self):
        print_subsection("Loading Models")
        
        # Load DNN
        self.dnn_model = tf.keras.models.load_model(DNN_MODEL_PATH)
        with open(DNN_SCALER_PATH, 'rb') as f:
            self.dnn_scaler = pickle.load(f)
        logger.info("DNN model loaded")
        
        # Load XGBoost
        self.xgb_model = XGBClassifier()
        self.xgb_model.load_model(XGBOOST_MODEL_PATH)
        logger.info("XGBoost model loaded")
    
    def predict_voting(self, X, threshold=ENSEMBLE_THRESHOLD):
        """Voting ensemble: Average predictions"""
        
        # DNN prediction
        X_scaled = self.dnn_scaler.transform(X)
        dnn_pred = self.dnn_model.predict(X_scaled, verbose=0).flatten()
        
        # XGBoost prediction
        xgb_pred = self.xgb_model.predict_proba(X)[:, 1]

        
        # Voting (weighted average)
        ensemble_pred = (DNN_WEIGHT * dnn_pred + XGBOOST_WEIGHT * xgb_pred)
        
        # Classification
        classifications = (ensemble_pred > threshold).astype(int)
        
        return ensemble_pred, classifications, dnn_pred, xgb_pred
    
    def get_detailed_predictions(self, X, threshold=ENSEMBLE_THRESHOLD):
        """Get detailed predictions with all scores"""
        
        X_scaled = self.dnn_scaler.transform(X)
        dnn_pred = self.dnn_model.predict(X_scaled, verbose=0).flatten()
        xgb_pred = self.xgb_model.predict_proba(X)[:, 1]

        ensemble_pred = (DNN_WEIGHT * dnn_pred + XGBOOST_WEIGHT * xgb_pred)
        
        results = []
        for i in range(len(X)):
            results.append({
                'dnn_score': dnn_pred[i],
                'xgb_score': xgb_pred[i],
                'ensemble_score': ensemble_pred[i],
                'prediction': 'PHISHING' if ensemble_pred[i] > threshold else 'LEGITIMATE',
                'confidence': max(ensemble_pred[i], 1 - ensemble_pred[i])
            })
        
        return results

def evaluate_models(X_train, X_test, y_train, y_test):
    """Comprehensive model evaluation"""
    print_section("MODEL EVALUATION")
    
    # Load models
    ensemble = EnsemblePhishingDetector()
    
    # Get predictions
    print_subsection("Generating Predictions")
    
    # Train predictions
    train_ensemble_pred, train_ensemble_class, train_dnn, train_xgb = ensemble.predict_voting(X_train)
    
    # Test predictions
    test_ensemble_pred, test_ensemble_class, test_dnn, test_xgb = ensemble.predict_voting(X_test)
    
    # Load DNN separately for train predictions
    dnn_model = tf.keras.models.load_model(DNN_MODEL_PATH)
    with open(DNN_SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    xgb_model = XGBClassifier()
    xgb_model.load_model(XGBOOST_MODEL_PATH)
    
    train_dnn_class = (train_dnn > 0.5).astype(int)
    train_xgb_class = xgb_model.predict(X_train)
    
    # Evaluation on test set
    print_subsection("Test Set Results")
    
    results = {}
    
    # DNN
    dnn_acc = accuracy_score(y_test, (test_dnn > 0.5).astype(int))
    dnn_prec = precision_score(y_test, (test_dnn > 0.5).astype(int))
    dnn_rec = recall_score(y_test, (test_dnn > 0.5).astype(int))
    dnn_f1 = f1_score(y_test, (test_dnn > 0.5).astype(int))
    dnn_auc = roc_auc_score(y_test, test_dnn)
    
    results['DNN'] = {
        'accuracy': dnn_acc,
        'precision': dnn_prec,
        'recall': dnn_rec,
        'f1': dnn_f1,
        'auc': dnn_auc
    }
    
    print("\nDNN Results:")
    print(f"  Accuracy:  {dnn_acc:.4f}")
    print(f"  Precision: {dnn_prec:.4f}")
    print(f"  Recall:    {dnn_rec:.4f}")
    print(f"  F1-Score:  {dnn_f1:.4f}")
    print(f"  ROC-AUC:   {dnn_auc:.4f}")
    
    # XGBoost
    # XGBoost
    xgb_class = (test_xgb > 0.5).astype(int)  # Convert probabilities to 0/1 for metrics
    xgb_acc = accuracy_score(y_test, xgb_class)
    xgb_prec = precision_score(y_test, xgb_class)
    xgb_rec = recall_score(y_test, xgb_class)
    xgb_f1 = f1_score(y_test, xgb_class)
    xgb_auc = roc_auc_score(y_test, test_xgb)  # Use probabilities only for AUC
    # keep probabilities for AUC only

    
    results['XGBoost'] = {
        'accuracy': xgb_acc,
        'precision': xgb_prec,
        'recall': xgb_rec,
        'f1': xgb_f1,
        'auc': xgb_auc
    }
    
    print("\nXGBoost Results:")
    print(f"  Accuracy:  {xgb_acc:.4f}")
    print(f"  Precision: {xgb_prec:.4f}")
    print(f"  Recall:    {xgb_rec:.4f}")
    print(f"  F1-Score:  {xgb_f1:.4f}")
    print(f"  ROC-AUC:   {xgb_auc:.4f}")
    
    # Ensemble
    ens_acc = accuracy_score(y_test, test_ensemble_class)
    ens_prec = precision_score(y_test, test_ensemble_class)
    ens_rec = recall_score(y_test, test_ensemble_class)
    ens_f1 = f1_score(y_test, test_ensemble_class)
    ens_auc = roc_auc_score(y_test, test_ensemble_pred)
    
    results['Ensemble'] = {
        'accuracy': ens_acc,
        'precision': ens_prec,
        'recall': ens_rec,
        'f1': ens_f1,
        'auc': ens_auc
    }
    
    print("\nEnsemble Voting Results:")
    print(f"  Accuracy:  {ens_acc:.4f}")
    print(f"  Precision: {ens_prec:.4f}")
    print(f"  Recall:    {ens_rec:.4f}")
    print(f"  F1-Score:  {ens_f1:.4f}")
    print(f"  ROC-AUC:   {ens_auc:.4f}")
    
    # Save metrics
    metrics_df = pd.DataFrame(results).T
    save_dataframe(metrics_df, RESULTS_DIR / "metrics.csv", "evaluation metrics")
    
    return results, (y_test, test_dnn, test_xgb, test_ensemble_pred)

def plot_results(y_test, test_dnn, test_xgb, test_ensemble):
    """Plot ROC curves and confusion matrices"""
    print_subsection("Generating Plots")
    
    # ROC Curves
    plt.figure(figsize=(10, 7))
    
    fpr_dnn, tpr_dnn, _ = roc_curve(y_test, test_dnn)
    auc_dnn = roc_auc_score(y_test, test_dnn)
    
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, test_xgb)
    auc_xgb = roc_auc_score(y_test, test_xgb)
    
    fpr_ens, tpr_ens, _ = roc_curve(y_test, test_ensemble)
    auc_ens = roc_auc_score(y_test, test_ensemble)
    
    plt.plot(fpr_dnn, tpr_dnn, label=f'DNN (AUC={auc_dnn:.3f})', linewidth=2)
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={auc_xgb:.3f})', linewidth=2)
    plt.plot(fpr_ens, tpr_ens, label=f'Ensemble (AUC={auc_ens:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'roc_curves.png', dpi=300)
    logger.info("ROC curves saved")
    
    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    models = [
        ('DNN', test_dnn),
        ('XGBoost', test_xgb),
        ('Ensemble', test_ensemble)
    ]
    
    for idx, (name, preds) in enumerate(models):
        class_preds = (preds > 0.5).astype(int)
        cm = confusion_matrix(y_test, class_preds)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'])
        axes[idx].set_title(f'{name} Confusion Matrix', fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'confusion_matrices.png', dpi=300)
    logger.info("Confusion matrices saved")

def run_evaluation():
    """Run complete evaluation"""
    print_section("ENSEMBLE EVALUATION & TESTING")
    
    # Load features
    print_subsection("Loading Features")
    features_path = PROCESSED_DIR / "features_augmented.csv"
    df = pd.read_csv(features_path)
    
    X = df.drop(['label', 'filename'], axis=1).values
    y = df['label'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape} samples")
    print(f"Test set: {X_test.shape} samples")
    
    # Evaluate
    results, plot_data = evaluate_models(X_train, X_test, y_train, y_test)
    
    # Plot results
    plot_results(*plot_data)
    
    print_subsection("Evaluation Complete")
    print(f"Results saved to: {RESULTS_DIR}")
    
    return results

if __name__ == "__main__":
    run_evaluation()
