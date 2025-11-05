# src/utils.py
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from config import LOG_FILE, LOGS_DIR

# Configure logging
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_logger():
    """Get configured logger"""
    return logger

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def print_subsection(title):
    """Print formatted subsection header"""
    print(f"\n>>> {title}")
    print("-" * 70)

def get_audio_files(directory, extension=".mp3"):
    """Get all audio files from directory"""
    path = Path(directory)
    return sorted(list(path.glob(f"*{extension}")))

def check_dataset_completeness():
    """Verify dataset has minimum required files"""
    from config import PHISHING_DIR, LEGITIMATE_DIR
    
    phishing_files = get_audio_files(PHISHING_DIR)
    legitimate_files = get_audio_files(LEGITIMATE_DIR)
    
    logger.info(f"Found {len(phishing_files)} phishing files")
    logger.info(f"Found {len(legitimate_files)} legitimate files")
    
    if len(phishing_files) < 10:
        logger.warning(f"Only {len(phishing_files)} phishing files found (recommend 40+)")
    if len(legitimate_files) < 10:
        logger.warning(f"Only {len(legitimate_files)} legitimate files found (recommend 40+)")
    
    return len(phishing_files), len(legitimate_files)

def save_dataframe(df, filepath, description=""):
    """Save dataframe and log"""
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {description} to {filepath}")
    logger.info(f"Shape: {df.shape}")

def load_dataframe(filepath):
    """Load dataframe and log"""
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {filepath} with shape {df.shape}")
    return df

class ProgressBar:
    """Simple progress bar"""
    def __init__(self, total, prefix='Progress:'):
        self.total = total
        self.prefix = prefix
        self.current = 0
    
    def update(self, step=1):
        self.current += step
        progress = self.current / self.total
        filled = int(50 * progress)
        bar = '█' * filled + '░' * (50 - filled)
        print(f'\r{self.prefix} |{bar}| {progress*100:.1f}%', end='')
    
    def finish(self):
        print()
