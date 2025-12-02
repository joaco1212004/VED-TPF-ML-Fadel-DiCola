"""
Logging and configuration utilities for the ML pipeline.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(name, log_dir='logs', level=logging.INFO):
    """
    Setup logging for a module.
    
    Args:
        name: Logger name (usually __name__)
        log_dir: Directory to save log files
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        log_path / f'{name.replace(".", "_")}_{timestamp}.log'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


class Config:
    """Configuration class for ML pipeline."""
    
    # Paths
    DATA_DIR = Path('Data')
    MODELS_DIR = Path('models')
    RESULTS_DIR = Path('results')
    LOGS_DIR = Path('logs')
    
    # Data files
    METRICS_CSV = DATA_DIR / 'VED_DynamicData_Metrics.csv'
    FOURIER_CSV = DATA_DIR / 'VED_DynamicData_Fourier.csv'
    STATIC_ICE_HEV = DATA_DIR / 'VED_Static_Data_ICE&HEV.csv'
    STATIC_PHEV_EV = DATA_DIR / 'VED_Static_Data_PHEV&EV.csv'
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.16  # Of training data
    
    # Linear models
    LINEAR_RIDGE_ALPHA = 1.0
    LINEAR_LASSO_ALPHA = 0.01
    
    # Random Forest
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH = 20
    RF_MIN_SAMPLES_SPLIT = 5
    RF_MIN_SAMPLES_LEAF = 2
    RF_N_JOBS = -1
    
    # XGBoost
    XGB_N_ESTIMATORS = 300
    XGB_LEARNING_RATE = 0.05
    XGB_MAX_DEPTH = 7
    XGB_SUBSAMPLE = 0.8
    XGB_COLSAMPLE_BYTREE = 0.8
    XGB_EARLY_STOPPING_ROUNDS = 50
    
    # LightGBM
    LGBM_N_ESTIMATORS = 300
    LGBM_LEARNING_RATE = 0.05
    LGBM_MAX_DEPTH = 7
    LGBM_NUM_LEAVES = 31
    LGBM_EARLY_STOPPING_ROUNDS = 50
    
    # Neural Network
    MLP_HIDDEN_LAYERS = [128, 64, 32]
    MLP_DROPOUT_RATE = 0.2
    MLP_LEARNING_RATE = 0.001
    MLP_EPOCHS = 100
    MLP_BATCH_SIZE = 32
    MLP_EARLY_STOPPING_PATIENCE = 15
    MLP_REDUCE_LR_PATIENCE = 5
    MLP_REDUCE_LR_FACTOR = 0.5
    
    # Training
    VERBOSE = 1
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories."""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        (cls.MODELS_DIR / 'saved_models').mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary."""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        
        config_dict = cls.to_dict()
        for section_name in ['DATA', 'MODEL', 'TRAINING']:
            print(f"\n{section_name} PARAMETERS:")
            print("-" * 60)
            
            for key, value in sorted(config_dict.items()):
                if section_name in key:
                    display_key = key.replace(f'{section_name}_', '')
                    print(f"  {display_key:30s} = {value}")
        
        print("\n" + "="*60 + "\n")


# Create directories on import
Config.create_directories()
