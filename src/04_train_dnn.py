# src/04_train_dnn.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from config import (PROCESSED_DIR, MODEL_DIR, DNN_MODEL_PATH, DNN_SCALER_PATH,
                    DNN_EPOCHS, DNN_BATCH_SIZE, DNN_VALIDATION_SPLIT)
from src.utils import print_section, print_subsection, get_logger

logger = get_logger()

class DeepPhishingDetector:
    """Deep Neural Network for phishing detection"""
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.scaler = StandardScaler()
        self.model = None
        self.build_model()
    
    def build_model(self):
        """Build DNN architecture"""
        self.model = keras.Sequential([
            keras.layers.Input(shape=(self.input_dim,)),
            
            # First hidden block
            keras.layers.Dense(256, activation='relu', name='dense_1'),
            keras.layers.BatchNormalization(name='bn_1'),
            keras.layers.Dropout(0.4, name='dropout_1'),
            
            # Second hidden block
            keras.layers.Dense(128, activation='relu', name='dense_2'),
            keras.layers.BatchNormalization(name='bn_2'),
            keras.layers.Dropout(0.3, name='dropout_2'),
            
            # Third hidden block
            keras.layers.Dense(64, activation='relu', name='dense_3'),
            keras.layers.BatchNormalization(name='bn_3'),
            keras.layers.Dropout(0.2, name='dropout_3'),
            
            # Fourth hidden block
            keras.layers.Dense(32, activation='relu', name='dense_4'),
            keras.layers.BatchNormalization(name='bn_4'),
            keras.layers.Dropout(0.2, name='dropout_4'),
            
            # Output layer
            keras.layers.Dense(1, activation='sigmoid', name='output')
        ], name='PhishingDetectorDNN')
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                keras.metrics.AUC()
            ]
        )
        
        logger.info("DNN model created")
    
    def train(self, X, y, epochs=DNN_EPOCHS, batch_size=DNN_BATCH_SIZE, 
              validation_split=DNN_VALIDATION_SPLIT):
        """Train the DNN model"""
        
        print_subsection("Scaling Features")
        X_scaled = self.scaler.fit_transform(X)
        logger.info(f"Features scaled: mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")
        
        print_subsection("Training DNN")
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train
        history = self.model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        logger.info(f"Training complete. Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0)
    
    def save(self, model_path=DNN_MODEL_PATH, scaler_path=DNN_SCALER_PATH):
        """Save model and scaler"""
        self.model.save(model_path)
        logger.info(f"DNN model saved to {model_path}")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"DNN scaler saved to {scaler_path}")

def train_dnn_model():
    """Train DNN on features"""
    print_section("DEEP NEURAL NETWORK TRAINING")
    
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
    detector = DeepPhishingDetector(input_dim=X.shape[1])

    history = detector.train(X, y, epochs=DNN_EPOCHS, batch_size=DNN_BATCH_SIZE)
    
    # Save model
    detector.save()
    
    # Print summary
    print_subsection("Model Summary")
    detector.model.summary()
    
    return detector, history

if __name__ == "__main__":
    train_dnn_model()
