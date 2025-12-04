# Model Training Pipeline

Complete Python package for training and comparing multiple machine learning models for energy consumption prediction.

## Structure

### Core Modules

#### 1. **preprocessing.py**
Data preparation and feature scaling utilities.

**Key Classes:**
- `DataPreprocessor`: Handles data splitting, scaling, and serialization
  - `prepare_data()`: Split data into train (64%), validation (16%), test (20%)
  - `fit_scalers()`: Learn scaling parameters from training data
  - `scale_features()`: Scale feature data
  - `scale_target()`: Scale target variable
  - `inverse_scale_target()`: Reverse target scaling
  - `save_scalers()`: Serialize scalers to disk
  - `load_scalers()`: Load scalers from disk

**Key Functions:**
- `prepare_modeling_data()`: Merge metrics CSV with static data to create feature matrix

#### 2. **linear_models.py**
Linear regression models with regularization.

**Key Classes:**
- `LinearRegressionModel`: Wrapper for OLS, Ridge, and Lasso regression
  - Supports: `linear`, `ridge`, `lasso`
  - Methods: `train()`, `predict()`, `evaluate()`, `get_coefficients()`

**Key Functions:**
- `train_linear_models()`: Train all three variants (OLS, Ridge, Lasso)
  - Returns dict with trained models and metrics

#### 3. **random_forest_model.py**
Random Forest ensemble model.

**Key Classes:**
- `RandomForestModel`: Random Forest regressor wrapper
  - Hyperparameters: n_estimators=200, max_depth=20, min_samples_split=5
  - Methods: `train()`, `predict()`, `evaluate()`, `get_feature_importance()`, `get_oob_score()`

**Key Functions:**
- `train_random_forest()`: Train RF with feature importance analysis

#### 4. **boosting_models.py**
Gradient boosting models (XGBoost, LightGBM).

**Key Classes:**
- `XGBoostModel`: XGBoost regressor with early stopping
  - Hyperparameters: n_estimators=300, learning_rate=0.05, max_depth=7
  - Early stopping: after 50 rounds without improvement
  - Methods: `train()`, `predict()`, `evaluate()`, `get_feature_importance()`

- `LightGBMModel`: LightGBM regressor with early stopping
  - Similar architecture to XGBoost
  - Typically faster training

**Key Functions:**
- `train_boosting_models()`: Train both XGBoost and LightGBM

#### 5. **neural_network_models.py**
Deep learning models using Keras/TensorFlow.

**Key Classes:**
- `MLPRegressor`: Multi-layer perceptron with batch normalization and dropout
  - Architecture: Configurable hidden layers [default: 128, 64, 32]
  - Regularization: Batch normalization + Dropout (0.2)
  - Callbacks: EarlyStopping (patience=15), ReduceLROnPlateau
  - Methods: `train()`, `predict()`, `evaluate()`, `plot_training_history()`

**Key Functions:**
- `train_mlp()`: Train neural network with early stopping

#### 6. **evaluation.py**
Model evaluation and comparison utilities.

**Key Classes:**
- `ModelComparator`: Compare multiple models across metrics
  - Methods: `add_model_results()`, `get_comparison_dataframe()`, `plot_comparison()`, `get_best_model()`
  - Supports metrics: RMSE, MAE, R², MAPE, MSE, Training Time

**Key Functions:**
- `calculate_prediction_errors()`: Detailed error analysis
- `plot_predictions_vs_actual()`: Visualization of predictions
- `plot_residuals()`: Residual analysis plots
- `plot_error_distribution()`: Error distribution histograms

#### 7. **feature_importance.py**
Feature importance analysis and visualization.

**Key Classes:**
- `FeatureImportanceAnalyzer`: Analyze importance across models
  - Methods: `add_importance()`, `get_top_features()`, `plot_feature_importance()`, `plot_comparison()`
  - Cross-model analysis: Find features important in multiple models

**Key Functions:**
- `calculate_permutation_importance()`: Permutation-based importance scores

## Typical Workflow

```python
from src.preprocessing import DataPreprocessor, prepare_modeling_data
from src.linear_models import train_linear_models
from src.random_forest_model import train_random_forest
from src.boosting_models import train_boosting_models
from src.neural_network_models import train_mlp
from src.evaluation import ModelComparator
from src.feature_importance import FeatureImportanceAnalyzer

# 1. Prepare data
X, y = prepare_modeling_data('Data/VED_DynamicData_Metrics.csv', 
                             'Data/VED_Static_Data_ICE&HEV.csv',
                             'Data/VED_Static_Data_PHEV&EV.csv')

# 2. Preprocess and split
preprocessor = DataPreprocessor()
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data(X, y)

# 3. Scale features
preprocessor.fit_scalers(X_train)
X_train = preprocessor.scale_features(X_train)
X_val = preprocessor.scale_features(X_val)
X_test = preprocessor.scale_features(X_test)

# 4. Train models
linear_results = train_linear_models(X_train, X_val, X_test, y_train, y_val, y_test)
rf_model = train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test)
xgb_model, lgbm_model = train_boosting_models(X_train, X_val, X_test, y_train, y_val, y_test)
mlp_model = train_mlp(X_train, X_val, X_test, y_train, y_val, y_test, 
                      input_dim=X_train.shape[1])

# 5. Compare models
comparator = ModelComparator()
comparator.add_model_results('Linear OLS', y_test, linear_results['ols']['y_pred'])
comparator.add_model_results('Random Forest', y_test, rf_model.predict(X_test))
# ... add more models ...
comparator.print_summary()

# 6. Analyze feature importance
analyzer = FeatureImportanceAnalyzer(X.columns.tolist())
analyzer.add_importance('Random Forest', rf_model.get_feature_importance())
analyzer.add_importance('XGBoost', xgb_model.get_feature_importance())
# ... add more models ...
analyzer.print_summary()
```

## Metrics Used

All models report:
- **RMSE** (Root Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better
- **R²** (Coefficient of Determination): Higher is better (0-1 scale)
- **MAPE** (Mean Absolute Percentage Error): Lower is better (%)
- **MSE** (Mean Squared Error): Lower is better
- **Training Time**: Seconds

## Model Selection Tips

- **Linear Models**: Fast, interpretable, good for baseline
- **Random Forest**: Robust to outliers, handles non-linear patterns, good feature importance
- **XGBoost/LightGBM**: High performance, handles complex patterns, faster than deep learning
- **Neural Networks**: Best for very large datasets, can capture complex patterns, slower training

## Hyperparameter Tuning

All models have configurable hyperparameters. Examples:

```python
# Linear model
model = LinearRegressionModel(model_type='ridge', alpha=0.5)

# Random Forest
model = RandomForestModel(n_estimators=300, max_depth=15)

# XGBoost
model = XGBoostModel(learning_rate=0.01, max_depth=5)

# Neural Network
model = MLPRegressor(hidden_layers=[256, 128, 64], dropout_rate=0.3)
```

## Saving and Loading Models

All models support serialization:

```python
# Save
model.save_model('path/to/model.pkl')  # or .h5 for neural networks

# Load
model = SomeModel()
model.load_model('path/to/model.pkl')
```

## Troubleshooting

- **Import Errors**: Ensure all dependencies are installed (numpy, pandas, scikit-learn, xgboost, lightgbm, tensorflow)
- **Memory Issues**: For large datasets, reduce batch size in neural networks
- **Training Time**: LightGBM typically faster than XGBoost; linear models fastest
- **Overfitting**: Increase regularization (alpha in linear models, dropout in NN) or reduce model complexity
