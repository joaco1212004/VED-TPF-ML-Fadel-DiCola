"""
XGBoost and LightGBM Models
State-of-the-art gradient boosting implementations.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import time


class XGBoostModel:
    """XGBoost Regressor wrapper."""
    
    def __init__(self, n_estimators=300, learning_rate=0.05, max_depth=7, 
                 subsample=0.8, colsample_bytree=0.8, random_state=42):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate (eta)
            max_depth: Maximum tree depth
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            random_state: Random seed
        """
        self.hyperparams = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'n_jobs': -1
        }
        
        self.model = xgb.XGBRegressor(**self.hyperparams)
        self.is_trained = False
        self.training_time = None
        self.metrics = {}
    
    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=50):
        """
        Train the XGBoost model with optional early stopping.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (for early stopping)
            y_val: Validation target (for early stopping)
            early_stopping_rounds: Rounds for early stopping
        """
        print(f"\n{'='*60}")
        print(f"Training XGBoost")
        print(f"  n_estimators: {self.hyperparams['n_estimators']}")
        print(f"  learning_rate: {self.hyperparams['learning_rate']}")
        print(f"  max_depth: {self.hyperparams['max_depth']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(X_train, y_train, eval_set=eval_set, 
                          early_stopping_rounds=early_stopping_rounds, verbose=False)
        else:
            self.model.fit(X_train, y_train, verbose=False)
        
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
        
        print(f"\n{dataset_name} Set Metrics (XGBoost):")
        print(f"  RMSE:  {rmse:.6f}")
        print(f"  MAE:   {mae:.6f}")
        print(f"  R²:    {r2:.6f}")
        print(f"  MAPE:  {mape:.2f}%")
        
        return self.metrics[dataset_name]
    
    def get_feature_importance(self, feature_names=None, top_n=15):
        """Get feature importance scores."""
        importance_dict = self.model.get_booster().get_score(importance_type='gain')
        importance = pd.DataFrame(list(importance_dict.items()), 
                                 columns=['feature', 'importance'])
        importance = importance.sort_values('importance', ascending=False)
        
        if top_n:
            return importance.head(top_n)
        return importance
    
    def save_model(self, path):
        """Save trained model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / 'xgboost_model.pkl')
        print(f"✓ Model saved to {path / 'xgboost_model.pkl'}")
    
    def load_model(self, path):
        """Load trained model from disk."""
        path = Path(path)
        self.model = joblib.load(path / 'xgboost_model.pkl')
        self.is_trained = True
        print(f"✓ Model loaded from {path / 'xgboost_model.pkl'}")


class LightGBMModel:
    """LightGBM Regressor wrapper."""
    
    def __init__(self, n_estimators=300, learning_rate=0.05, max_depth=7,
                 num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42):
        """
        Initialize LightGBM model.
        
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            num_leaves: Maximum number of leaves
            subsample: Subsample ratio
            colsample_bytree: Feature subsample ratio
            random_state: Random seed
        """
        self.hyperparams = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        
        self.model = lgb.LGBMRegressor(**self.hyperparams)
        self.is_trained = False
        self.training_time = None
        self.metrics = {}
    
    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=50):
        """
        Train the LightGBM model with optional early stopping.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (for early stopping)
            y_val: Validation target (for early stopping)
            early_stopping_rounds: Rounds for early stopping
        """
        print(f"\n{'='*60}")
        print(f"Training LightGBM")
        print(f"  n_estimators: {self.hyperparams['n_estimators']}")
        print(f"  learning_rate: {self.hyperparams['learning_rate']}")
        print(f"  max_depth: {self.hyperparams['max_depth']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(X_train, y_train, eval_set=eval_set,
                          early_stopping_rounds=early_stopping_rounds)
        else:
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
        
        print(f"\n{dataset_name} Set Metrics (LightGBM):")
        print(f"  RMSE:  {rmse:.6f}")
        print(f"  MAE:   {mae:.6f}")
        print(f"  R²:    {r2:.6f}")
        print(f"  MAPE:  {mape:.2f}%")
        
        return self.metrics[dataset_name]
    
    def get_feature_importance(self, feature_names=None, top_n=15):
        """Get feature importance scores."""
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
    
    def save_model(self, path):
        """Save trained model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / 'lightgbm_model.pkl')
        print(f"✓ Model saved to {path / 'lightgbm_model.pkl'}")
    
    def load_model(self, path):
        """Load trained model from disk."""
        path = Path(path)
        self.model = joblib.load(path / 'lightgbm_model.pkl')
        self.is_trained = True
        print(f"✓ Model loaded from {path / 'lightgbm_model.pkl'}")


def train_boosting_models(X_train, X_val, X_test, y_train, y_val, y_test,
                         feature_names=None):
    """
    Train and evaluate both XGBoost and LightGBM models.
    
    Args:
        X_train, X_val, X_test: Feature sets
        y_train, y_val, y_test: Target sets
        feature_names: Optional feature names
        
    Returns:
        dict: Trained models
    """
    results = {}
    
    # XGBoost
    model_xgb = XGBoostModel()
    model_xgb.train(X_train, y_train, X_val, y_val, early_stopping_rounds=50)
    model_xgb.evaluate(X_val, y_val, 'Validation')
    model_xgb.evaluate(X_test, y_test, 'Test')
    results['xgboost'] = model_xgb
    
    # LightGBM
    model_lgb = LightGBMModel()
    model_lgb.train(X_train, y_train, X_val, y_val, early_stopping_rounds=50)
    model_lgb.evaluate(X_val, y_val, 'Validation')
    model_lgb.evaluate(X_test, y_test, 'Test')
    results['lightgbm'] = model_lgb
    
    return results
