"""
Data preprocessing utilities for energy consumption prediction.
Handles scaling, standardization, and data preparation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path


class DataPreprocessor:
    """Handles data preprocessing and scaling."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
    
    def prepare_data(self, X, y, test_size=0.2, val_size=0.2):
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Features dataframe
            y: Target values
            test_size: Proportion for test set
            val_size: Proportion of remaining data for validation
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: separate test set (20%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state
        )
        
        # Second split: separate validation from training (20% of remaining)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def fit_scalers(self, X_train, y_train):
        """Fit scalers on training data."""
        self.scaler_X.fit(X_train)
        self.scaler_y.fit(y_train.values.reshape(-1, 1))
        self.is_fitted = True
    
    def scale_features(self, X, fit=False):
        """Scale features using fitted scaler."""
        if fit:
            self.scaler_X.fit(X)
            self.is_fitted = True
        return self.scaler_X.transform(X)
    
    def scale_target(self, y, fit=False):
        """Scale target variable."""
        if fit:
            self.scaler_y.fit(y.values.reshape(-1, 1))
        return self.scaler_y.transform(y.values.reshape(-1, 1)).ravel()
    
    def inverse_scale_target(self, y_scaled):
        """Convert scaled predictions back to original scale."""
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    
    def save_scalers(self, path):
        """Save scalers to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler_X, path / 'scaler_X.pkl')
        joblib.dump(self.scaler_y, path / 'scaler_y.pkl')
        print(f"✓ Scalers saved to {path}")
    
    def load_scalers(self, path):
        """Load scalers from disk."""
        path = Path(path)
        self.scaler_X = joblib.load(path / 'scaler_X.pkl')
        self.scaler_y = joblib.load(path / 'scaler_y.pkl')
        self.is_fitted = True
        print(f"✓ Scalers loaded from {path}")


def prepare_modeling_data(metrics_df, static_df, target_col='fuel_consumption'):
    """
    Prepare data for modeling by merging metrics and static data.
    
    Args:
        metrics_df: VED_DynamicData_Metrics.csv
        static_df: VED_Static_Data combined
        target_col: Column name for target variable
        
    Returns:
        X, y: Features and target
    """
    # Merge dynamic metrics with static data on vehicle ID
    data = metrics_df.merge(static_df, on='VehId', how='inner')
    
    # Separate features and target
    X = data.drop(columns=[target_col, 'filename'], errors='ignore')
    y = data[target_col]
    
    return X, y
