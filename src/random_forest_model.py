"""
Random Forest Regressor
Implements ensemble learning with decision trees.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import time


class RandomForestModel:
    """Random Forest Regressor wrapper."""
    
    def __init__(self, n_estimators=200, max_depth=20, min_samples_split=5, 
                 min_samples_leaf=2, random_state=42):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            random_state: Random seed
        """
        self.hyperparams = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
            'n_jobs': -1
        }
        
        self.model = RandomForestRegressor(**self.hyperparams)
        self.is_trained = False
        self.training_time = None
        self.metrics = {}
    
    def train(self, X_train, y_train):
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        print(f"\n{'='*60}")
        print(f"Training Random Forest")
        print(f"  n_estimators: {self.hyperparams['n_estimators']}")
        print(f"  max_depth: {self.hyperparams['max_depth']}")
        print(f"  min_samples_split: {self.hyperparams['min_samples_split']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        
        self.is_trained = True
        print(f"✓ Training completed in {self.training_time:.4f} seconds")
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, dataset_name='Test'):
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test target
            dataset_name: Name of dataset (for printing)
            
        Returns:
            dict: Metrics
        """
        y_pred = self.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        self.metrics[dataset_name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'MSE': mse
        }
        
        print(f"\n{dataset_name} Set Metrics (Random Forest):")
        print(f"  RMSE:  {rmse:.6f}")
        print(f"  MAE:   {mae:.6f}")
        print(f"  R²:    {r2:.6f}")
        print(f"  MAPE:  {mape:.2f}%")
        
        return self.metrics[dataset_name]
    
    def get_feature_importance(self, feature_names=None, top_n=15):
        """
        Get feature importance scores.
        
        Args:
            feature_names: Optional list of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame: Feature importance sorted by importance
        """
        importance = pd.DataFrame({
            'importance': self.model.feature_importances_
        })
        
        if feature_names:
            importance['feature'] = feature_names
        else:
            importance['feature'] = [f'feature_{i}' for i in range(len(self.model.feature_importances_))]
        
        importance = importance.sort_values('importance', ascending=False)
        
        if top_n:
            return importance.head(top_n)
        return importance
    
    def get_oob_score(self):
        """Get Out-of-Bag error score."""
        # Train with OOB enabled
        model_oob = RandomForestRegressor(
            n_estimators=self.hyperparams['n_estimators'],
            max_depth=self.hyperparams['max_depth'],
            oob_score=True,
            random_state=self.hyperparams['random_state'],
            n_jobs=-1
        )
        return model_oob.oob_score_
    
    def save_model(self, path):
        """Save trained model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / 'random_forest_model.pkl')
        print(f"✓ Model saved to {path / 'random_forest_model.pkl'}")
    
    def load_model(self, path):
        """Load trained model from disk."""
        path = Path(path)
        self.model = joblib.load(path / 'random_forest_model.pkl')
        self.is_trained = True
        print(f"✓ Model loaded from {path / 'random_forest_model.pkl'}")


def train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test, 
                       feature_names=None, **kwargs):
    """
    Train and evaluate Random Forest model.
    
    Args:
        X_train, X_val, X_test: Feature sets
        y_train, y_val, y_test: Target sets
        feature_names: Optional feature names
        **kwargs: Additional hyperparameters
        
    Returns:
        RandomForestModel: Trained model
    """
    model = RandomForestModel(**kwargs)
    model.train(X_train, y_train)
    model.evaluate(X_val, y_val, 'Validation')
    model.evaluate(X_test, y_test, 'Test')
    
    if feature_names:
        print(f"\nTop 15 Feature Importance:")
        print(model.get_feature_importance(feature_names, top_n=15))
    
    return model
