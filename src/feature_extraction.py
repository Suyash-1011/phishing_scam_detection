# src/03_feature_extraction.py
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
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
            for i in range(min(10, mel_db.shape)):
                features[f'mel_{i}'] = float(np.mean(mel_db[i]))  # ✅ Scalar
            
            print(f"✅ Extracted {len(features)} scalar features")
            
            # Verify all values are scalars
            for key, value in features.items():
                if isinstance(value, (list, np.ndarray)):
                    print(f"❌ ERROR: Feature '{key}' is not a scalar: {type(value)}")
                    return None
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            return None

    def extract_all_features(self, use_augmented=True):
        """Extract features from entire dataset"""
        print_subsection("Extracting Features")
        
        if use_augmented:
            source_dir = AUGMENTED_DIR
            label_name = "augmented_label"
        else:
            source_dir = None
        
        X = []
        y = []
        filenames = []
        
        # Phishing samples
        phishing_dirs = [AUGMENTED_DIR / "phishing*"] if use_augmented else [PHISHING_DIR]
        if use_augmented:
            phishing_files = [f for f in AUGMENTED_DIR.glob("phishing*") if f.is_file()]
        else:
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
        if use_augmented:
            legitimate_files = [f for f in AUGMENTED_DIR.glob("legitimate*") if f.is_file()]
        else:
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
        logger.info(f"Total features: {len(df.columns) - 2}")
        
        return df

def run_feature_extraction():
    """Run complete feature extraction pipeline"""
    print_section("FEATURE EXTRACTION PIPELINE")
    
    extractor = AudioFeatureExtractor(sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    features_df = extractor.extract_all_features(use_augmented=True)
    
    print_subsection("Feature Extraction Summary")
    print(f"Total samples: {len(features_df)}")
    print(f"Feature dimensions: {len(features_df.columns) - 2}")
    print(f"Phishing samples: {(features_df['label'] == 1).sum()}")
    print(f"Legitimate samples: {(features_df['label'] == 0).sum()}")
    
    # Save features
    output_path = PROCESSED_DIR / "features_augmented.csv"
    save_dataframe(features_df, output_path, "augmented features")
    
    return features_df

if __name__ == "__main__":
    run_feature_extraction()
