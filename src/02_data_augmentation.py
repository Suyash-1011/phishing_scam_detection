# src/02_data_augmentation.py
import librosa
import numpy as np
import soundfile as sf
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (PHISHING_DIR, LEGITIMATE_DIR, AUGMENTED_DIR, 
                    SAMPLE_RATE, AUGMENT_FACTOR)
from src.utils import print_section, print_subsection, get_audio_files, ProgressBar, get_logger

logger = get_logger()

class AudioAugmenter:
    """Audio augmentation techniques"""
    
    def __init__(self, sr=SAMPLE_RATE):
        self.sr = sr
    
    def time_stretch(self, y, rate=1.1):
        """Speed up/slow down without changing pitch"""
        return librosa.effects.time_stretch(y, rate=rate)
    
    def pitch_shift(self, y, n_steps=2):
        """Change pitch without changing speed"""
        return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
    
    def add_white_noise(self, y, noise_factor=0.005):
        """Add white noise to simulate poor call quality"""
        noise = np.random.randn(len(y))
        return y + noise_factor * noise
    
    def add_background_hum(self, y, noise_factor=0.003):
        """Add low-frequency background hum"""
        t = np.arange(len(y)) / self.sr
        hum = np.sin(2 * np.pi * 50 * t)  # 50 Hz hum
        return y + noise_factor * hum
    
    def amplitude_scale(self, y, scale=0.8):
        """Change amplitude"""
        return y * scale
    
    def augment_file(self, input_path, label, output_dir=AUGMENTED_DIR, augment_count=5):
        """Apply multiple augmentations to single file"""
        try:
            y, sr = librosa.load(input_path, sr=self.sr)
            filename = Path(input_path).stem
            
            augmentations = [
                ("original", y),
                ("time_stretch_slow", self.time_stretch(y, rate=0.9)),
                ("time_stretch_fast", self.time_stretch(y, rate=1.1)),
                ("pitch_up", self.pitch_shift(y, n_steps=2)),
                ("pitch_down", self.pitch_shift(y, n_steps=-2)),
                ("noise", self.add_white_noise(y, noise_factor=0.007)),
                ("hum", self.add_background_hum(y, noise_factor=0.005)),
                ("quiet", self.amplitude_scale(y, scale=0.7)),
            ]
            
            saved_count = 0
            for aug_name, augmented_y in augmentations[:augment_count]:
                try:
                    output_filename = f"{label}_{filename}_{aug_name}.wav"
                    output_path = output_dir / output_filename
                    sf.write(output_path, augmented_y, sr)
                    saved_count += 1
                except Exception as e:
                    logger.warning(f"Failed to save augmentation {aug_name}: {e}")
            
            return saved_count
        except Exception as e:
            logger.error(f"Error augmenting {input_path}: {e}")
            return 0
    
    def augment_dataset(self, phishing_dir, legitimate_dir, output_dir=AUGMENTED_DIR):
        """Augment all files in dataset"""
        total_originals = 0
        total_augmented = 0
        
        print_subsection(f"Augmenting Phishing Samples (factor={AUGMENT_FACTOR})")
        phishing_files = get_audio_files(phishing_dir)
        total_originals += len(phishing_files)
        
        progress = ProgressBar(len(phishing_files), 'Phishing:')
        for file in phishing_files:
            count = self.augment_file(file, "phishing", output_dir, AUGMENT_FACTOR)
            total_augmented += count
            progress.update()
        progress.finish()
        
        print_subsection(f"Augmenting Legitimate Samples (factor={AUGMENT_FACTOR})")
        legitimate_files = get_audio_files(legitimate_dir)
        total_originals += len(legitimate_files)
        
        progress = ProgressBar(len(legitimate_files), 'Legitimate:')
        for file in legitimate_files:
            count = self.augment_file(file, "legitimate", output_dir, AUGMENT_FACTOR)
            total_augmented += count
            progress.update()
        progress.finish()
        
        logger.info(f"Augmentation complete!")
        logger.info(f"Original files: {total_originals}")
        logger.info(f"Augmented files generated: {total_augmented}")
        logger.info(f"Output directory: {output_dir}")
        
        return total_originals, total_augmented

def run_augmentation():
    """Run complete augmentation pipeline"""
    print_section("DATA AUGMENTATION PIPELINE")
    
    augmenter = AudioAugmenter(sr=SAMPLE_RATE)
    orig, aug = augmenter.augment_dataset(PHISHING_DIR, LEGITIMATE_DIR, AUGMENTED_DIR)
    
    print_subsection("Augmentation Summary")
    print(f"Original samples: {orig}")
    print(f"Total augmented files: {aug}")
    print(f"Expansion factor: {aug/orig:.1f}x")

if __name__ == "__main__":
    run_augmentation()
