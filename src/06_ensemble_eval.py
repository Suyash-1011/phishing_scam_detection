# src/06_ensemble_eval.py
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
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (PROCESSED_DIR, MODEL_DIR, DNN_MODEL_PATH, DNN_SCALER_PATH,
                    XGBOOST_MODEL_PATH, RESULTS_DIR, ENSEMBLE_THRESHOLD,
                    DNN_WEIGHT, XGBOOST_WEIGHT)
from src.utils import print_section, print_subsection, save_dataframe, get_logger

logger = get_logger()

class EnsemblePhishingDetector:
    """Ensemble predictor combining available models"""
    
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
        
        # Load Random Forest (if exists)
        rf_path = MODEL_DIR / "random_forest.pkl"
        if rf_path.exists():
            with open(rf_path, 'rb') as f:
                self.rf_model = pickle.load(f)
            logger.info("Random Forest model loaded")
        else:
            self.rf_model = None
            logger.warning("Random Forest model not found - skipping")
        
        # Load LightGBM (if exists)
        lgbm_path = MODEL_DIR / "lightgbm.pkl"
        if lgbm_path.exists():
            try:
                with open(lgbm_path, 'rb') as f:
                    self.lgbm_model = pickle.load(f)
                logger.info("LightGBM model loaded")
            except Exception as e:
                self.lgbm_model = None
                logger.warning(f"LightGBM model load failed: {e}")
        else:
            self.lgbm_model = None
            logger.warning("LightGBM model not found - skipping")
    
    def predict_all(self, X):
        """Get predictions from all available models"""
        predictions = {}
        
        # DNN prediction
        X_scaled = self.dnn_scaler.transform(X)
        predictions['dnn'] = self.dnn_model.predict(X_scaled, verbose=0).flatten()
        
        # XGBoost prediction
        predictions['xgb'] = self.xgb_model.predict_proba(X)[:, 1]
        
        # Random Forest prediction (if available)
        if self.rf_model is not None:
            predictions['rf'] = self.rf_model.predict_proba(X)[:, 1]
        
        # LightGBM prediction (if available)
        if self.lgbm_model is not None:
            predictions['lgbm'] = self.lgbm_model.predict_proba(X)[:, 1]
        
        return predictions
    
    def predict_voting(self, X, threshold=ENSEMBLE_THRESHOLD):
        """Voting ensemble: Average predictions from all available models"""
        predictions = self.predict_all(X)
        
        # Calculate average of all available models
        all_preds = []
        for model_pred in predictions.values():
            all_preds.append(model_pred)
        
        ensemble_pred = np.mean(all_preds, axis=0)
        classifications = (ensemble_pred > threshold).astype(int)
        
        return ensemble_pred, classifications, predictions

def evaluate_models(X_train, X_test, y_train, y_test):
    """Comprehensive model evaluation"""
    print_section("MODEL EVALUATION")
    
    # Load models
    ensemble = EnsemblePhishingDetector()
    
    # Get predictions
    print_subsection("Generating Predictions")
    test_ensemble_pred, test_ensemble_class, test_preds = ensemble.predict_voting(X_test)
    
    # Evaluation on test set
    print_subsection("Test Set Results")
    
    results = {}
    
    # DNN
    test_dnn = test_preds['dnn']
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
    test_xgb = test_preds['xgb']
    xgb_class = (test_xgb > 0.5).astype(int)
    xgb_acc = accuracy_score(y_test, xgb_class)
    xgb_prec = precision_score(y_test, xgb_class)
    xgb_rec = recall_score(y_test, xgb_class)
    xgb_f1 = f1_score(y_test, xgb_class)
    xgb_auc = roc_auc_score(y_test, test_xgb)
    
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
    
    # Random Forest (if available)
    if 'rf' in test_preds:
        test_rf = test_preds['rf']
        rf_class = (test_rf > 0.5).astype(int)
        rf_acc = accuracy_score(y_test, rf_class)
        rf_prec = precision_score(y_test, rf_class)
        rf_rec = recall_score(y_test, rf_class)
        rf_f1 = f1_score(y_test, rf_class)
        rf_auc = roc_auc_score(y_test, test_rf)
        
        results['RandomForest'] = {
            'accuracy': rf_acc,
            'precision': rf_prec,
            'recall': rf_rec,
            'f1': rf_f1,
            'auc': rf_auc
        }
        
        print("\nRandom Forest Results:")
        print(f"  Accuracy:  {rf_acc:.4f}")
        print(f"  Precision: {rf_prec:.4f}")
        print(f"  Recall:    {rf_rec:.4f}")
        print(f"  F1-Score:  {rf_f1:.4f}")
        print(f"  ROC-AUC:   {rf_auc:.4f}")
    
    # LightGBM (if available)
    if 'lgbm' in test_preds:
        test_lgbm = test_preds['lgbm']
        lgbm_class = (test_lgbm > 0.5).astype(int)
        lgbm_acc = accuracy_score(y_test, lgbm_class)
        lgbm_prec = precision_score(y_test, lgbm_class)
        lgbm_rec = recall_score(y_test, lgbm_class)
        lgbm_f1 = f1_score(y_test, lgbm_class)
        lgbm_auc = roc_auc_score(y_test, test_lgbm)
        
        results['LightGBM'] = {
            'accuracy': lgbm_acc,
            'precision': lgbm_prec,
            'recall': lgbm_rec,
            'f1': lgbm_f1,
            'auc': lgbm_auc
        }
        
        print("\nLightGBM Results:")
        print(f"  Accuracy:  {lgbm_acc:.4f}")
        print(f"  Precision: {lgbm_prec:.4f}")
        print(f"  Recall:    {lgbm_rec:.4f}")
        print(f"  F1-Score:  {lgbm_f1:.4f}")
        print(f"  ROC-AUC:   {lgbm_auc:.4f}")
    
    # Ensemble
    n_models = len(test_preds)
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
    
    print(f"\nEnsemble Voting Results ({n_models} Models):")
    print(f"  Accuracy:  {ens_acc:.4f}")
    print(f"  Precision: {ens_prec:.4f}")
    print(f"  Recall:    {ens_rec:.4f}")
    print(f"  F1-Score:  {ens_f1:.4f}")
    print(f"  ROC-AUC:   {ens_auc:.4f}")
    
    # Save metrics with model name as column
    metrics_df = pd.DataFrame(results).T
    metrics_df.insert(0, 'model', metrics_df.index)  # Add model name as first column
    metrics_df.reset_index(drop=True, inplace=True)  # Remove index
    save_dataframe(metrics_df, RESULTS_DIR / "metrics.csv", "evaluation metrics")
    
    # Print summary table
    print("\n" + "="*70)
    print("  📊 MODEL COMPARISON SUMMARY")
    print("="*70)
    print(metrics_df.to_string(index=False))
    print("="*70 + "\n")
    
    return results, (y_test, test_preds, test_ensemble_pred)

def plot_results(y_test, test_preds, test_ensemble):
    """Plot ROC curves and confusion matrices"""
    print_subsection("Generating Plots")
    
    # ROC Curves
    plt.figure(figsize=(12, 8))
    
    # Plot each model
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    model_names = {'dnn': 'DNN', 'xgb': 'XGBoost', 'rf': 'Random Forest', 'lgbm': 'LightGBM'}
    
    idx = 0
    for name, label in model_names.items():
        if name in test_preds:
            fpr, tpr, _ = roc_curve(y_test, test_preds[name])
            auc_score = roc_auc_score(y_test, test_preds[name])
            plt.plot(fpr, tpr, label=f'{label} (AUC={auc_score:.3f})', 
                    linewidth=2, color=colors[idx])
            idx += 1
    
    # Plot ensemble
    fpr_ens, tpr_ens, _ = roc_curve(y_test, test_ensemble)
    auc_ens = roc_auc_score(y_test, test_ensemble)
    plt.plot(fpr_ens, tpr_ens, label=f'Ensemble (AUC={auc_ens:.3f})', 
            linewidth=3, color=colors[idx], linestyle='--')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison - All Models', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'roc_curves.png', dpi=300)
    logger.info("ROC curves saved")
    plt.close()
    
    # Confusion matrices
    n_models = len(test_preds) + 1  # +1 for ensemble
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    idx = 0
    for name, label in model_names.items():
        if name in test_preds and idx < len(axes):
            class_preds = (test_preds[name] > 0.5).astype(int)
            cm = confusion_matrix(y_test, class_preds)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Legitimate', 'Phishing'],
                       yticklabels=['Legitimate', 'Phishing'])
            axes[idx].set_title(f'{label} Confusion Matrix', fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
            idx += 1
    
    # Ensemble confusion matrix
    if idx < len(axes):
        class_preds = (test_ensemble > 0.5).astype(int)
        cm = confusion_matrix(y_test, class_preds)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[idx],
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'])
        axes[idx].set_title('Ensemble Confusion Matrix', fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
        idx += 1
    
    # Hide unused subplots
    for i in range(idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'confusion_matrices.png', dpi=300)
    logger.info("Confusion matrices saved")
    plt.close()

def run_evaluation():
    """Run complete evaluation"""
    print_section("ENSEMBLE EVALUATION & TESTING")
    
    # Load pre-split train and test data
    print_subsection("Loading Pre-Split Data")
    train_path = PROCESSED_DIR / "train_features.csv"
    test_path = PROCESSED_DIR / "test_features.csv"
    
    # Check if files exist
    if not train_path.exists() or not test_path.exists():
        logger.error("Train/test features not found. Please run feature extraction first.")
        print("❌ Error: train_features.csv or test_features.csv not found")
        print("Please run: python src/feature_extraction.py")
        return None
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(['label', 'filename'], axis=1).values
    y_train = train_df['label'].values
    
    X_test = test_df.drop(['label', 'filename'], axis=1).values
    y_test = test_df['label'].values
    
    print(f"Training set: {X_train.shape} samples")
    print(f"  - Phishing: {(y_train == 1).sum()}")
    print(f"  - Legitimate: {(y_train == 0).sum()}")
    print(f"Test set: {X_test.shape} samples")
    print(f"  - Phishing: {(y_test == 1).sum()}")
    print(f"  - Legitimate: {(y_test == 0).sum()}")
    
    # Evaluate
    results, plot_data = evaluate_models(X_train, X_test, y_train, y_test)
    
    # Plot results
    plot_results(*plot_data)
    
    print_subsection("Evaluation Complete")
    print(f"Results saved to: {RESULTS_DIR}")
    
    return results

if __name__ == "__main__":
    run_evaluation()