# src/01_data_exploration.py
import librosa
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PHISHING_DIR, LEGITIMATE_DIR, SAMPLE_RATE
from src.utils import print_section, print_subsection, get_audio_files, get_logger

logger = get_logger()

def analyze_single_file(filepath):
    """Analyze a single audio file"""
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
        duration = librosa.get_duration(y=y, sr=sr)
        
        return {
            'filename': Path(filepath).name,
            'duration_seconds': round(duration, 2),
            'sample_rate': sr,
            'total_samples': len(y),
            'status': 'OK'
        }
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return {
            'filename': Path(filepath).name,
            'status': f'ERROR: {str(e)}'
        }

def explore_dataset():
    """Explore and analyze the entire dataset"""
    print_section("DATASET EXPLORATION")
    
    # Phishing files
    print_subsection("Analyzing Phishing Audio Files")
    phishing_files = get_audio_files(PHISHING_DIR)
    phishing_stats = []
    
    for idx, file in enumerate(phishing_files, 1):
        stats = analyze_single_file(file)
        phishing_stats.append(stats)
        if idx <= 5:  # Show first 5
            print(f"File {idx}: {stats['filename']}")
            print(f"  Duration: {stats.get('duration_seconds', 'ERROR')}s | Sample Rate: {stats.get('sample_rate', 'N/A')} Hz")
    
    print(f"\nTotal phishing files: {len(phishing_files)}")
    
    # Legitimate files
    print_subsection("Analyzing Legitimate Audio Files")
    legitimate_files = get_audio_files(LEGITIMATE_DIR)
    legitimate_stats = []
    
    for idx, file in enumerate(legitimate_files, 1):
        stats = analyze_single_file(file)
        legitimate_stats.append(stats)
        if idx <= 5:  # Show first 5
            print(f"File {idx}: {stats['filename']}")
            print(f"  Duration: {stats.get('duration_seconds', 'ERROR')}s | Sample Rate: {stats.get('sample_rate', 'N/A')} Hz")
    
    print(f"\nTotal legitimate files: {len(legitimate_files)}")
    
    # Statistics
    print_subsection("Dataset Statistics")
    
    all_phishing_durations = [s['duration_seconds'] for s in phishing_stats if 'duration_seconds' in s]
    all_legit_durations = [s['duration_seconds'] for s in legitimate_stats if 'duration_seconds' in s]
    
    if all_phishing_durations:
        print(f"Phishing audio duration: {np.mean(all_phishing_durations):.2f}s ± {np.std(all_phishing_durations):.2f}s")
        print(f"  Min: {np.min(all_phishing_durations):.2f}s | Max: {np.max(all_phishing_durations):.2f}s")
    
    if all_legit_durations:
        print(f"Legitimate audio duration: {np.mean(all_legit_durations):.2f}s ± {np.std(all_legit_durations):.2f}s")
        print(f"  Min: {np.min(all_legit_durations):.2f}s | Max: {np.max(all_legit_durations):.2f}s")
    
    print(f"\nTotal samples before augmentation: {len(phishing_files) + len(legitimate_files)}")
    
    logger.info(f"Dataset exploration complete: {len(phishing_files)} phishing, {len(legitimate_files)} legitimate")

if __name__ == "__main__":
    explore_dataset()
