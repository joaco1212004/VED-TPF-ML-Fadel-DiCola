"""
Linear Regression Models (OLS, Ridge, Lasso)
Implements baseline linear models with regularization.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import time


class LinearRegressionModel:
    """Wrapper for Linear Regression models."""
    
    def __init__(self, model_type='ols', alpha=1.0):
        """
        Initialize linear regression model.
        
        Args:
            model_type: 'ols', 'ridge', or 'lasso'
            alpha: Regularization strength (for ridge/lasso)
        """
        self.model_type = model_type
        self.alpha = alpha
        
        if model_type == 'ols':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=alpha, random_state=42)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
        else:
            raise ValueError("model_type must be 'ols', 'ridge', or 'lasso'")
        
        self.is_trained = False
        self.training_time = None
        self.metrics = {}
    
    def train(self, X_train, y_train):
        """
        Train the linear model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_type.upper()} Regression")
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
        
        print(f"\n{dataset_name} Set Metrics ({self.model_type.upper()}):")
        print(f"  RMSE:  {rmse:.6f}")
        print(f"  MAE:   {mae:.6f}")
        print(f"  R²:    {r2:.6f}")
        print(f"  MAPE:  {mape:.2f}%")
        
        return self.metrics[dataset_name]
    
    def get_coefficients(self, feature_names=None):
        """
        Get model coefficients.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            DataFrame: Coefficients with feature names
        """
        coefs = pd.DataFrame({
            'coefficient': self.model.coef_
        })
        
        if feature_names:
            coefs['feature'] = feature_names
            coefs = coefs.sort_values('coefficient', key=abs, ascending=False)
        
        return coefs
    
    def save_model(self, path):
        """Save trained model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / f'{self.model_type}_model.pkl')
        print(f"✓ Model saved to {path / f'{self.model_type}_model.pkl'}")
    
    def load_model(self, path):
        """Load trained model from disk."""
        path = Path(path)
        self.model = joblib.load(path / f'{self.model_type}_model.pkl')
        self.is_trained = True
        print(f"✓ Model loaded from {path / f'{self.model_type}_model.pkl'}")


def train_linear_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_names=None):
    """
    Train and evaluate all linear regression models.
    
    Args:
        X_train, X_val, X_test: Feature sets
        y_train, y_val, y_test: Target sets
        feature_names: Optional feature names for coefficients
        
    Returns:
        dict: Trained models and results
    """
    results = {}
    
    # OLS
    model_ols = LinearRegressionModel(model_type='ols')
    model_ols.train(X_train, y_train)
    model_ols.evaluate(X_val, y_val, 'Validation')
    model_ols.evaluate(X_test, y_test, 'Test')
    results['ols'] = model_ols
    
    # Ridge (alpha=1.0)
    model_ridge = LinearRegressionModel(model_type='ridge', alpha=1.0)
    model_ridge.train(X_train, y_train)
    model_ridge.evaluate(X_val, y_val, 'Validation')
    model_ridge.evaluate(X_test, y_test, 'Test')
    results['ridge'] = model_ridge
    
    # Lasso (alpha=0.01)
    model_lasso = LinearRegressionModel(model_type='lasso', alpha=0.01)
    model_lasso.train(X_train, y_train)
    model_lasso.evaluate(X_val, y_val, 'Validation')
    model_lasso.evaluate(X_test, y_test, 'Test')
    results['lasso'] = model_lasso
    
    return results
