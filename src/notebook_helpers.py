"""
Helper functions for notebook usage.
Simplifies common operations in main.ipynb
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_all_data():
    """
    Load all required data files.
    
    Returns:
        dict: Dictionary with all dataframes
    """
    data = {}
    
    # Load CSVs
    data['metrics'] = pd.read_csv('Data/VED_DynamicData_Metrics.csv')
    data['fourier'] = pd.read_csv('Data/VED_DynamicData_Fourier.csv')
    data['static_ice'] = pd.read_csv('Data/VED_Static_Data_ICE&HEV.csv')
    data['static_ev'] = pd.read_csv('Data/VED_Static_Data_PHEV&EV.csv')
    
    print("✓ Data loaded successfully")
    print(f"  Metrics: {data['metrics'].shape}")
    print(f"  Fourier: {data['fourier'].shape}")
    print(f"  Static ICE/HEV: {data['static_ice'].shape}")
    print(f"  Static PHEV/EV: {data['static_ev'].shape}")
    
    return data


def prepare_features_target(metrics_df, static_ice, static_ev, 
                           metric_type='metrics', target_col=None):
    """
    Prepare feature matrix and target variable.
    
    Args:
        metrics_df: Metrics dataframe
        static_ice: ICE/HEV static data
        static_ev: PHEV/EV static data
        metric_type: 'metrics' or 'fourier'
        target_col: Column name for target (if None, uses first numeric)
        
    Returns:
        X, y: Feature matrix and target
        feature_cols: List of feature column names
    """
    # Combine static data
    static = pd.concat([static_ice, static_ev], ignore_index=True)
    
    # Merge
    data = metrics_df.merge(static, on='VehId', how='inner')
    
    # Define exclusion columns
    exclude = ['filename', 'VehId', 'Trip', 'Timestamp', 'Timestamp(ms)', 
               'DayNum', 'Latitude', 'Longitude']
    
    # Get feature columns
    feature_cols = [c for c in data.columns 
                   if c not in exclude 
                   and data[c].dtype in [np.float64, np.int64]]
    
    # Prepare X
    X = data[feature_cols].fillna(data[feature_cols].mean())
    
    # Prepare y
    if target_col is None:
        # Use first numeric column not in features
        for col in data.columns:
            if col not in feature_cols and col not in exclude:
                if data[col].dtype in [np.float64, np.int64]:
                    target_col = col
                    break
    
    if target_col is None:
        raise ValueError("No suitable target column found. Please specify target_col")
    
    y = data[target_col].values
    
    print(f"✓ Features prepared: {X.shape[1]} features, {len(y)} samples")
    print(f"  Target: {target_col}")
    print(f"  Features: {feature_cols[:5]}... (showing first 5)")
    
    return X, y, feature_cols


def create_results_summary(comparator, analyzer, best_model_name, best_model):
    """
    Create a comprehensive results summary.
    
    Args:
        comparator: ModelComparator instance
        analyzer: FeatureImportanceAnalyzer instance
        best_model_name: Name of best model
        best_model: Best model instance
        
    Returns:
        dict: Summary results
    """
    summary = {
        'Best Model': best_model_name,
        'Comparison Table': comparator.get_comparison_dataframe(),
    }
    
    # Add feature importance if available
    if hasattr(analyzer, 'get_common_important_features'):
        summary['Common Important Features'] = analyzer.get_common_important_features()
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    # Print best model
    comparison_df = summary['Comparison Table']
    print(f"\nBEST MODEL: {best_model_name}")
    print("-" * 80)
    print(comparison_df.loc[best_model_name].to_string())
    
    # Print ranking
    print(f"\nMODEL RANKING (by RMSE):")
    print("-" * 80)
    ranking = comparison_df.sort_values('RMSE')
    for i, (model, row) in enumerate(ranking.iterrows(), 1):
        print(f"{i}. {model:25s} RMSE={row['RMSE']:.6f}  MAE={row['MAE']:.6f}  R²={row['R2']:.6f}")
    
    print("\n" + "="*80 + "\n")
    
    return summary


def plot_model_performance_dashboard(comparator, y_test, predictions_dict, figsize=(18, 12)):
    """
    Create a comprehensive performance dashboard.
    
    Args:
        comparator: ModelComparator instance
        y_test: Test target values
        predictions_dict: Dict of {model_name: predictions}
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Model comparison - RMSE
    ax1 = fig.add_subplot(gs[0, 0])
    comparison_df = comparator.get_comparison_dataframe()
    rmse_sorted = comparison_df['RMSE'].sort_values()
    colors = ['green' if i == 0 else 'lightblue' for i in range(len(rmse_sorted))]
    rmse_sorted.plot(kind='barh', ax=ax1, color=colors)
    ax1.set_xlabel('RMSE (Lower is Better)')
    ax1.set_title('Model Comparison - RMSE')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Model comparison - R²
    ax2 = fig.add_subplot(gs[0, 1])
    r2_sorted = comparison_df['R2'].sort_values(ascending=False)
    colors = ['green' if i == 0 else 'lightblue' for i in range(len(r2_sorted))]
    r2_sorted.plot(kind='barh', ax=ax2, color=colors)
    ax2.set_xlabel('R² (Higher is Better)')
    ax2.set_title('Model Comparison - R²')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Training time comparison
    ax3 = fig.add_subplot(gs[0, 2])
    time_sorted = comparison_df['Training Time'].sort_values()
    time_sorted.plot(kind='barh', ax=ax3, color='lightcoral')
    ax3.set_xlabel('Training Time (seconds)')
    ax3.set_title('Training Time Comparison')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4-6. Predictions vs Actual for top 3 models
    for idx, (model_name, preds) in enumerate(list(predictions_dict.items())[:3]):
        ax = fig.add_subplot(gs[1, idx])
        ax.scatter(y_test, preds, alpha=0.5, s=20)
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        r2 = (comparison_df.loc[model_name, 'R2'] 
              if model_name in comparison_df.index else 0)
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{model_name}\nR² = {r2:.4f}')
        ax.grid(True, alpha=0.3)
    
    # 7-9. Residuals for top 3 models
    for idx, (model_name, preds) in enumerate(list(predictions_dict.items())[:3]):
        ax = fig.add_subplot(gs[2, idx])
        residuals = y_test - preds
        ax.scatter(preds, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        ax.set_title(f'{model_name}\nMean Res = {residuals.mean():.4f}')
        ax.grid(True, alpha=0.3)
    
    return fig


def save_all_results(comparator, analyzer, models_dict, preprocessor, output_dir='results'):
    """
    Save all results to files.
    
    Args:
        comparator: ModelComparator instance
        analyzer: FeatureImportanceAnalyzer instance
        models_dict: Dict of {model_name: model_instance}
        preprocessor: DataPreprocessor instance
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save comparison table
    comparison_df = comparator.get_comparison_dataframe()
    comparison_df.to_csv(output_path / 'model_comparison.csv')
    print(f"✓ Comparison table saved to {output_path / 'model_comparison.csv'}")
    
    # Save models
    models_path = Path('models/saved_models')
    models_path.mkdir(parents=True, exist_ok=True)
    
    for model_name, model_obj in models_dict.items():
        try:
            if hasattr(model_obj, 'save_model'):
                ext = '.h5' if 'MLP' in str(type(model_obj)) else '.pkl'
                model_obj.save_model(models_path / f'{model_name.replace(" ", "_")}{ext}')
                print(f"✓ Model saved: {model_name}")
        except Exception as e:
            print(f"⚠ Could not save {model_name}: {e}")
    
    # Save scalers
    try:
        preprocessor.save_scalers(models_path / 'scalers.pkl')
        print(f"✓ Scalers saved")
    except Exception as e:
        print(f"⚠ Could not save scalers: {e}")
    
    # Save feature importance
    if hasattr(analyzer, 'get_common_important_features'):
        common_features = analyzer.get_common_important_features()
        importance_data = []
        for feature, stats in common_features.items():
            importance_data.append({
                'Feature': feature,
                'Mean Importance': stats['Mean'],
                'Std Importance': stats['Std'],
                'Models Count': stats['Count']
            })
        
        importance_df = pd.DataFrame(importance_data)
        importance_df.to_csv(output_path / 'feature_importance.csv', index=False)
        print(f"✓ Feature importance saved to {output_path / 'feature_importance.csv'}")
    
    print(f"\n✓ All results saved to {output_path}")


def print_data_summary(X, y, feature_cols):
    """Print summary statistics of data."""
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    
    print(f"\nFeatures: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Target variable statistics:")
    print(f"  Mean: {y.mean():.6f}")
    print(f"  Std:  {y.std():.6f}")
    print(f"  Min:  {y.min():.6f}")
    print(f"  Max:  {y.max():.6f}")
    
    print(f"\nMissing values in features: {X.isnull().sum().sum()}")
    print(f"Missing values in target: {np.isnan(y).sum()}")
    
    print(f"\nFeature statistics:")
    print(f"  Mean of means: {X.mean().mean():.6f}")
    print(f"  Mean of stds:  {X.std().mean():.6f}")
    print(f"  Correlation with target (first 5):")
    
    for i, col in enumerate(feature_cols[:5]):
        corr = np.corrcoef(X[col], y)[0, 1]
        print(f"    {col}: {corr:.6f}")
    
    print("\n" + "="*80 + "\n")
