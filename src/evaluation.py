"""
Model Evaluation and Comparison Utilities
Handles metrics calculation, visualization, and comparison across models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path


class ModelComparator:
    """Compare multiple models across different metrics."""
    
    def __init__(self):
        self.results = {}
        self.models = {}
    
    def add_model_results(self, model_name, y_test, y_pred, training_time=None):
        """
        Add model results for comparison.
        
        Args:
            model_name: Name of the model
            y_test: True target values
            y_pred: Predicted values
            training_time: Optional training time in seconds
        """
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        self.results[model_name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'MSE': mse,
            'Training Time': training_time if training_time else 0
        }
    
    def get_comparison_dataframe(self):
        """Get comparison results as DataFrame."""
        df = pd.DataFrame(self.results).T
        return df.round(6)
    
    def plot_comparison(self, metrics=['RMSE', 'MAE', 'R2'], figsize=(15, 5)):
        """
        Plot comparison of models across metrics.
        
        Args:
            metrics: List of metrics to plot
            figsize: Figure size
        """
        df = self.get_comparison_dataframe()
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        
        colors = sns.color_palette("husl", len(df))
        
        for idx, metric in enumerate(metrics):
            if metric in df.columns:
                values = df[metric].sort_values(ascending=(metric in ['RMSE', 'MAE', 'MAPE']))
                
                bars = axes[idx].barh(range(len(values)), values.values, color=colors)
                axes[idx].set_yticks(range(len(values)))
                axes[idx].set_yticklabels(values.index)
                axes[idx].set_xlabel(metric)
                axes[idx].set_title(f'Model Comparison - {metric}')
                axes[idx].grid(True, alpha=0.3, axis='x')
                
                # Add value labels on bars
                for i, (bar, val) in enumerate(zip(bars, values.values)):
                    axes[idx].text(val, bar.get_y() + bar.get_height()/2,
                                  f'{val:.4f}', va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def get_best_model(self, metric='RMSE'):
        """
        Get best model for a given metric.
        
        Args:
            metric: Metric to rank by
            
        Returns:
            str: Best model name
            float: Best metric value
        """
        df = self.get_comparison_dataframe()
        
        if metric in df.columns:
            if metric in ['RMSE', 'MAE', 'MAPE']:
                best_idx = df[metric].idxmin()
                best_val = df[metric].min()
            else:  # R2
                best_idx = df[metric].idxmax()
                best_val = df[metric].max()
            
            return best_idx, best_val
        
        return None, None
    
    def print_summary(self):
        """Print summary of model comparison."""
        df = self.get_comparison_dataframe()
        
        print(f"\n{'='*80}")
        print(f"MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(df.to_string())
        print(f"\n{'='*80}")
        print(f"BEST MODELS:")
        print(f"  RMSE:  {self.get_best_model('RMSE')[0]} ({self.get_best_model('RMSE')[1]:.6f})")
        print(f"  MAE:   {self.get_best_model('MAE')[0]} ({self.get_best_model('MAE')[1]:.6f})")
        print(f"  R²:    {self.get_best_model('R2')[0]} ({self.get_best_model('R2')[1]:.6f})")
        print(f"  MAPE:  {self.get_best_model('MAPE')[0]} ({self.get_best_model('MAPE')[1]:.2f}%)")
        print(f"{'='*80}\n")


def calculate_prediction_errors(y_test, y_pred):
    """
    Calculate detailed error metrics.
    
    Args:
        y_test: True values
        y_pred: Predicted values
        
    Returns:
        dict: Detailed error statistics
    """
    errors = y_test - y_pred
    abs_errors = np.abs(errors)
    pct_errors = np.abs(errors / y_test) * 100
    
    return {
        'Mean Error': np.mean(errors),
        'Median Error': np.median(errors),
        'Std Error': np.std(errors),
        'Max Error': np.max(abs_errors),
        'Min Error': np.min(abs_errors),
        'Mean Abs Error': np.mean(abs_errors),
        'Median Abs Error': np.median(abs_errors),
        'Mean % Error': np.mean(pct_errors),
        'Median % Error': np.median(pct_errors),
        'Within 10% (%)': (pct_errors <= 10).sum() / len(pct_errors) * 100,
        'Within 20% (%)': (pct_errors <= 20).sum() / len(pct_errors) * 100
    }


def plot_predictions_vs_actual(y_test, y_pred_dict, figsize=(15, 5)):
    """
    Plot predictions vs actual for multiple models.
    
    Args:
        y_test: True values
        y_pred_dict: Dict of {model_name: predictions}
        figsize: Figure size
    """
    n_models = len(y_pred_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, y_pred) in enumerate(y_pred_dict.items()):
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        axes[idx].scatter(y_test, y_pred, alpha=0.5, s=30)
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        axes[idx].set_xlabel('Actual')
        axes[idx].set_ylabel('Predicted')
        axes[idx].set_title(f'{model_name}\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
    
    plt.tight_layout()
    return fig


def plot_residuals(y_test, y_pred_dict, figsize=(15, 5)):
    """
    Plot residuals for multiple models.
    
    Args:
        y_test: True values
        y_pred_dict: Dict of {model_name: predictions}
        figsize: Figure size
    """
    n_models = len(y_pred_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, y_pred) in enumerate(y_pred_dict.items()):
        residuals = y_test - y_pred
        
        axes[idx].scatter(y_pred, residuals, alpha=0.5, s=30)
        axes[idx].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Residuals')
        axes[idx].set_title(f'{model_name}\nMean Residual = {residuals.mean():.4f}')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_error_distribution(y_test, y_pred_dict, figsize=(15, 5)):
    """
    Plot error distribution for multiple models.
    
    Args:
        y_test: True values
        y_pred_dict: Dict of {model_name: predictions}
        figsize: Figure size
    """
    n_models = len(y_pred_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, y_pred) in enumerate(y_pred_dict.items()):
        errors = np.abs(y_test - y_pred)
        
        axes[idx].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[idx].axvline(errors.mean(), color='r', linestyle='--', lw=2, label=f'Mean: {errors.mean():.4f}')
        axes[idx].axvline(errors.median(), color='g', linestyle='--', lw=2, label=f'Median: {errors.median():.4f}')
        axes[idx].set_xlabel('Absolute Error')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{model_name}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig
