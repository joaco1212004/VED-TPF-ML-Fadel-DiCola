"""
ML Model Training Package
Complete pipeline for training and comparing multiple regression models.
"""

from .preprocessing import DataPreprocessor, prepare_modeling_data
from .linear_models import LinearRegressionModel, train_linear_models
from .random_forest_model import RandomForestModel, train_random_forest
from .boosting_models import XGBoostModel, LightGBMModel, train_boosting_models
from .neural_network_models import MLPRegressor, train_mlp
from .evaluation import ModelComparator, calculate_prediction_errors, plot_predictions_vs_actual, plot_residuals, plot_error_distribution
from .feature_importance import FeatureImportanceAnalyzer, calculate_permutation_importance
from .config import Config, setup_logging
from .notebook_helpers import (
    load_all_data,
    prepare_features_target,
    create_results_summary,
    plot_model_performance_dashboard,
    save_all_results,
    print_data_summary
)

__all__ = [
    # Preprocessing
    'DataPreprocessor',
    'prepare_modeling_data',
    
    # Linear Models
    'LinearRegressionModel',
    'train_linear_models',
    
    # Random Forest
    'RandomForestModel',
    'train_random_forest',
    
    # Boosting
    'XGBoostModel',
    'LightGBMModel',
    'train_boosting_models',
    
    # Neural Networks
    'MLPRegressor',
    'train_mlp',
    
    # Evaluation
    'ModelComparator',
    'calculate_prediction_errors',
    'plot_predictions_vs_actual',
    'plot_residuals',
    'plot_error_distribution',
    
    # Feature Importance
    'FeatureImportanceAnalyzer',
    'calculate_permutation_importance',
    
    # Configuration
    'Config',
    'setup_logging',
    
    # Notebook Helpers
    'load_all_data',
    'prepare_features_target',
    'create_results_summary',
    'plot_model_performance_dashboard',
    'save_all_results',
    'print_data_summary',
]

__version__ = '1.0.0'
__author__ = 'ML Team'
