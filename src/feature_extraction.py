import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (AUGMENTED_DIR, PHISHING_DIR, LEGITIMATE_DIR, 
                    PROCESSED_DIR, SAMPLE_RATE, N_MFCC)
from src.utils import (print_section, print_subsection, get_audio_files, 
                   save_dataframe, ProgressBar, get_logger)

logger = get_logger()

class AudioFeatureExtractor:
    """Extract audio features from files"""
    
    def __init__(self, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
        self.sr = sr
        self.n_mfcc = n_mfcc
    
    def extract_mfcc(self, y):
        """Extract Mel-Frequency Cepstral Coefficients"""
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
        features = {}
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
            features[f'mfcc_{i}_std'] = np.std(mfcc[i])
        return features
    
    def extract_mel_spectrogram(self, y):
        """Extract Mel-scale spectrogram statistics"""
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return {
            'mel_spec_mean': np.mean(mel_spec_db),
            'mel_spec_std': np.std(mel_spec_db),
            'mel_spec_max': np.max(mel_spec_db),
            'mel_spec_min': np.min(mel_spec_db)
        }
    
    def extract_spectral_features(self, y):
        """Extract spectral contrast and centroid"""
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)
        
        return {
            'spectral_contrast_mean': np.mean(spectral_contrast),
            'spectral_contrast_std': np.std(spectral_contrast),
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_std': np.std(spectral_centroid),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_rolloff_std': np.std(spectral_rolloff)
        }
    
    def extract_energy_features(self, y):
        """Extract energy and zero-crossing rate"""
        zcr = librosa.feature.zero_crossing_rate(y)
        energy = np.sum(y**2)
        rms = np.sqrt(np.mean(y**2))
        
        return {
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'energy': energy,
            'energy_normalized': energy / len(y),
            'rms_energy': rms
        }
    
    def extract_chroma_features(self, y):
        """Extract chroma features"""
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        return {
            'chroma_mean': np.mean(chroma),
            'chroma_std': np.std(chroma)
        }
    
    def extract_tempogram(self, y):
        """Extract temporal features"""
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        
        return {
            'onset_mean': np.mean(onset_env),
            'onset_std': np.std(onset_env),
            'onset_max': np.max(onset_env)
        }
    
    def extract_features_from_file(self, filepath):
        
        try:
            # Load audio
            y, sr = librosa.load(filepath, sr=self.sr)
            
            features = {}
            
            # MFCC features - get MEAN of each coefficient
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            
            # ✅ IMPORTANT: Take mean across time axis to get scalars
            for i, mfcc_coef in enumerate(mfcc):
                features[f'mfcc_{i}'] = float(np.mean(mfcc_coef))  # ✅ Scalar value
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid'] = float(np.mean(spectral_centroids))  # ✅ Scalar
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff'] = float(np.mean(spectral_rolloff))  # ✅ Scalar
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zero_crossing_rate'] = float(np.mean(zcr))  # ✅ Scalar
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)  # ✅ Scalar
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            for i in range(12):
                features[f'chroma_{i}'] = float(np.mean(chroma[i]))  # ✅ Scalar
            
            # Mel spectrogram
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            for i in range(min(10, mel_db.shape[0])):
                features[f'mel_{i}'] = float(np.mean(mel_db[i]))  # ✅ Scalar
            
            # Verify all values are scalars
            for key, value in features.items():
                if isinstance(value, (list, np.ndarray)):
                    logger.error(f"Feature '{key}' is not a scalar: {type(value)}")
                    return None
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def extract_all_features(self, use_augmented=True):
        """
        Extract features from entire dataset with PROPER train-test split
        ✅ FIXED: Now splits ORIGINAL files first, then augments only training data
        """
        print_subsection("Extracting Features with Proper Train-Test Split")
        
        if not use_augmented:
            # Original behavior for non-augmented case
            return self._extract_without_augmentation()
        
        # ✅ FIXED: Get ORIGINAL files only
        phishing_files = get_audio_files(PHISHING_DIR)
        legitimate_files = get_audio_files(LEGITIMATE_DIR)
        
        logger.info(f"Found {len(phishing_files)} original phishing files")
        logger.info(f"Found {len(legitimate_files)} original legitimate files")
        
        # Create file-label pairs
        phishing_pairs = [(f, 1) for f in phishing_files]
        legitimate_pairs = [(f, 0) for f in legitimate_files]
        all_pairs = phishing_pairs + legitimate_pairs
        
        files, labels = zip(*all_pairs)
        
        # ✅ FIXED: Split ORIGINAL files FIRST (before augmentation)
        train_files, test_files, train_labels, test_labels = train_test_split(
            files, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        logger.info(f"Train split: {len(train_files)} files, Test split: {len(test_files)} files")
        
        # ✅ Extract features from TRAINING files (with augmentation)
        X_train = []
        y_train = []
        filenames_train = []
        
        print(f"\nExtracting training features (with augmentation)...")
        progress = ProgressBar(len(train_files), 'Training:')
        
        for file, label in zip(train_files, train_labels):
            # Get augmented versions for this training file
            base_name = Path(file).stem
            label_name = "phishing" if label == 1 else "legitimate"
            aug_pattern = f"{label_name}_{base_name}_*.wav"
            aug_files = list(AUGMENTED_DIR.glob(aug_pattern))
            
            if not aug_files:
                logger.warning(f"No augmented files found for {file}. Using original.")
                # Fall back to original if no augmented files
                features = self.extract_features_from_file(file)
                if features:
                    X_train.append(features)
                    y_train.append(label)
                    filenames_train.append(Path(file).name)
            else:
                # Use augmented versions
                for aug_file in aug_files:
                    features = self.extract_features_from_file(aug_file)
                    if features:
                        X_train.append(features)
                        y_train.append(label)
                        filenames_train.append(aug_file.name)
            
            progress.update()
        progress.finish()
        
        # ✅ Extract features from TEST files (NO augmentation, original only)
        X_test = []
        y_test = []
        filenames_test = []
        
        print(f"\nExtracting test features (original files only, no augmentation)...")
        progress = ProgressBar(len(test_files), 'Test:')
        
        for file, label in zip(test_files, test_labels):
            # Use ORIGINAL file only for test set
            features = self.extract_features_from_file(file)
            if features:
                X_test.append(features)
                y_test.append(label)
                filenames_test.append(Path(file).name)
            progress.update()
        progress.finish()
        
        # Convert to DataFrames
        train_df = pd.DataFrame(X_train)
        train_df['label'] = y_train
        train_df['filename'] = filenames_train
        
        test_df = pd.DataFrame(X_test)
        test_df['label'] = y_test
        test_df['filename'] = filenames_test
        
        logger.info(f"Training samples (with augmentation): {len(train_df)}")
        logger.info(f"Test samples (original only): {len(test_df)}")
        logger.info(f"Total features: {len(train_df.columns) - 2}")
        
        return train_df, test_df
    
    def _extract_without_augmentation(self):
        """Original extraction without augmentation"""
        X = []
        y = []
        filenames = []
        
        # Phishing samples
        phishing_files = get_audio_files(PHISHING_DIR)
        
        progress = ProgressBar(len(phishing_files), 'Extracting phishing:')
        for file in phishing_files:
            features = self.extract_features_from_file(file)
            if features:
                X.append(features)
                y.append(1)  # Phishing
                filenames.append(file.name)
            progress.update()
        progress.finish()
        
        # Legitimate samples
        legitimate_files = get_audio_files(LEGITIMATE_DIR)
        
        progress = ProgressBar(len(legitimate_files), 'Extracting legitimate:')
        for file in legitimate_files:
            features = self.extract_features_from_file(file)
            if features:
                X.append(features)
                y.append(0)  # Legitimate
                filenames.append(file.name)
            progress.update()
        progress.finish()
        
        # Convert to DataFrame
        df = pd.DataFrame(X)
        df['label'] = y
        df['filename'] = filenames
        
        logger.info(f"Extracted features from {len(df)} samples")
        
        return df, None  # Return None for test_df in non-augmented case

def run_feature_extraction():
    """Run complete feature extraction pipeline"""
    print_section("FEATURE EXTRACTION PIPELINE")
    extractor = AudioFeatureExtractor(sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    train_df, test_df = extractor.extract_all_features(use_augmented=True)
    
    print_subsection("Feature Extraction Summary")
    print(f"Training samples (with augmentation): {len(train_df)}")
    print(f"  - Phishing: {(train_df['label'] == 1).sum()}")
    print(f"  - Legitimate: {(train_df['label'] == 0).sum()}")
    print(f"Test samples (original only): {len(test_df)}")
    print(f"  - Phishing: {(test_df['label'] == 1).sum()}")
    print(f"  - Legitimate: {(test_df['label'] == 0).sum()}")
    print(f"Feature dimensions: {len(train_df.columns) - 2}")
    
    train_output_path = PROCESSED_DIR / "train_features.csv"
    test_output_path = PROCESSED_DIR / "test_features.csv"
    
    save_dataframe(train_df, train_output_path, "training features")
    save_dataframe(test_df, test_output_path, "test features")
    
    return train_df, test_df

if __name__ == "__main__":
    run_feature_extraction()