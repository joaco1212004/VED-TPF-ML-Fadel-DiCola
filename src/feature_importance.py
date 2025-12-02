"""
Feature Importance Analysis Utilities
Extract and visualize feature importance from different model types.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureImportanceAnalyzer:
    """Analyze and compare feature importance across models."""
    
    def __init__(self, feature_names):
        """
        Initialize analyzer.
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        self.importances = {}
    
    def add_importance(self, model_name, importance_values, importance_type='importance'):
        """
        Add feature importance from a model.
        
        Args:
            model_name: Name of the model
            importance_values: Array of importance values (must match feature count)
            importance_type: Type of importance ('importance', 'coefficient', etc.)
        """
        if len(importance_values) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {len(importance_values)}")
        
        self.importances[model_name] = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance_values,
            'Type': importance_type
        }).sort_values('Importance', ascending=False)
    
    def get_top_features(self, model_name, n=15):
        """
        Get top N features for a model.
        
        Args:
            model_name: Name of the model
            n: Number of top features
            
        Returns:
            DataFrame: Top features with importance scores
        """
        if model_name not in self.importances:
            raise ValueError(f"Model {model_name} not found")
        
        return self.importances[model_name].head(n)
    
    def plot_feature_importance(self, model_name, n=15, figsize=(10, 8)):
        """
        Plot top N features for a model.
        
        Args:
            model_name: Name of the model
            n: Number of top features
            figsize: Figure size
        """
        if model_name not in self.importances:
            raise ValueError(f"Model {model_name} not found")
        
        df = self.get_top_features(model_name, n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = sns.color_palette("viridis", len(df))
        bars = ax.barh(range(len(df)), df['Importance'].values, color=colors)
        
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['Feature'].values)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'{model_name} - Top {n} Features')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, df['Importance'].values)):
            ax.text(val, bar.get_y() + bar.get_height()/2,
                   f'{val:.6f}', va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_comparison(self, model_names=None, n=10, figsize=(14, 8)):
        """
        Compare top features across multiple models.
        
        Args:
            model_names: List of model names to compare (None = all)
            n: Number of top features per model
            figsize: Figure size
        """
        if model_names is None:
            model_names = list(self.importances.keys())
        
        fig, axes = plt.subplots(1, len(model_names), figsize=figsize)
        
        if len(model_names) == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(model_names):
            if model_name not in self.importances:
                continue
            
            df = self.get_top_features(model_name, n)
            
            colors = sns.color_palette("viridis", len(df))
            axes[idx].barh(range(len(df)), df['Importance'].values, color=colors)
            axes[idx].set_yticks(range(len(df)))
            axes[idx].set_yticklabels(df['Feature'].values, fontsize=9)
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{model_name}')
            axes[idx].invert_yaxis()
            axes[idx].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def get_common_important_features(self, model_names=None, n=10):
        """
        Get features that appear in top N of multiple models.
        
        Args:
            model_names: List of model names (None = all)
            n: Number of top features to consider
            
        Returns:
            dict: Features and their average importance
        """
        if model_names is None:
            model_names = list(self.importances.keys())
        
        common_features = {}
        
        for model_name in model_names:
            top_features = self.get_top_features(model_name, n)
            for _, row in top_features.iterrows():
                feature = row['Feature']
                importance = row['Importance']
                
                if feature not in common_features:
                    common_features[feature] = []
                common_features[feature].append(importance)
        
        # Calculate average importance across models
        avg_importance = {}
        for feature, importances in common_features.items():
            avg_importance[feature] = {
                'Mean': np.mean(importances),
                'Std': np.std(importances),
                'Count': len(importances),
                'Values': importances
            }
        
        # Sort by count (how many models it appeared in) then by mean importance
        sorted_features = sorted(
            avg_importance.items(),
            key=lambda x: (-x[1]['Count'], -x[1]['Mean'])
        )
        
        return dict(sorted_features)
    
    def print_summary(self, model_names=None, n=10):
        """
        Print summary of feature importance.
        
        Args:
            model_names: List of model names (None = all)
            n: Number of top features
        """
        if model_names is None:
            model_names = list(self.importances.keys())
        
        print(f"\n{'='*80}")
        print(f"FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*80}\n")
        
        for model_name in model_names:
            if model_name not in self.importances:
                continue
            
            print(f"\n{model_name} - Top {n} Features:")
            print("-" * 60)
            df = self.get_top_features(model_name, n)
            for idx, (_, row) in enumerate(df.iterrows(), 1):
                print(f"  {idx:2d}. {row['Feature']:40s} {row['Importance']:12.6f}")
        
        print(f"\n{'='*80}")
        print(f"COMMON IMPORTANT FEATURES (Top {n})")
        print(f"{'='*80}\n")
        
        common = self.get_common_important_features(model_names, n)
        
        for idx, (feature, stats) in enumerate(list(common.items())[:15], 1):
            print(f"{idx:2d}. {feature:40s}")
            print(f"    Mean: {stats['Mean']:10.6f}  Std: {stats['Std']:8.6f}  "
                  f"In {int(stats['Count'])}/{len(model_names)} models")


def calculate_permutation_importance(model, X_test, y_test, n_repeats=10):
    """
    Calculate permutation importance for a model.
    
    Args:
        model: Trained model with predict() method
        X_test: Test features
        y_test: Test target
        n_repeats: Number of permutation repeats
        
    Returns:
        dict: Feature importances
    """
    from sklearn.metrics import mean_squared_error
    
    baseline_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    importances = np.zeros(X_test.shape[1])
    
    for feature_idx in range(X_test.shape[1]):
        rmses = []
        for _ in range(n_repeats):
            X_permuted = X_test.copy()
            np.random.shuffle(X_permuted[:, feature_idx])
            permuted_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_permuted)))
            rmses.append(permuted_rmse - baseline_rmse)
        
        importances[feature_idx] = np.mean(rmses)
    
    return importances
