"""
Neural Network Models (MLP)
Implements multilayer perceptron with TensorFlow/Keras.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import time
import matplotlib.pyplot as plt


class MLPRegressor:
    """Multilayer Perceptron Regressor wrapper."""
    
    def __init__(self, input_dim, hidden_layers=[128, 64, 32], 
                 dropout_rate=0.2, learning_rate=0.001, random_state=42):
        """
        Initialize MLP model.
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of neurons per hidden layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            random_state: Random seed
        """
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.model = self._build_model()
        self.is_trained = False
        self.training_time = None
        self.history = None
        self.metrics = {}
    
    def _build_model(self):
        """Build the neural network architecture."""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(self.input_dim,)))
        
        # Hidden layers with BatchNorm and Dropout
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1))
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
              patience=15):
        """
        Train the MLP model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            epochs: Number of training epochs
            batch_size: Batch size
            patience: Early stopping patience
        """
        print(f"\n{'='*60}")
        print(f"Training MLP Neural Network")
        print(f"  Hidden layers: {self.hidden_layers}")
        print(f"  Dropout rate: {self.dropout_rate}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Epochs: {epochs}, Batch size: {batch_size}")
        print(f"{'='*60}")
        
        # Early stopping callback
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=0
        )
        
        # Reduce learning rate callback
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0
        )
        
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        print(f"✓ Training completed in {self.training_time:.4f} seconds")
        print(f"  Stopped at epoch {len(self.history.history['loss'])}")
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict(X, verbose=0).ravel()
    
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
        
        print(f"\n{dataset_name} Set Metrics (MLP):")
        print(f"  RMSE:  {rmse:.6f}")
        print(f"  MAE:   {mae:.6f}")
        print(f"  R²:    {r2:.6f}")
        print(f"  MAPE:  {mape:.2f}%")
        
        return self.metrics[dataset_name]
    
    def plot_training_history(self):
        """Plot training history (loss curves)."""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('MLP Training History - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE plot
        axes[1].plot(self.history.history['mae'], label='Train MAE')
        axes[1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('MLP Training History - MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_model(self, path):
        """Save trained model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / 'mlp_model.h5')
        print(f"✓ Model saved to {path / 'mlp_model.h5'}")
    
    def load_model(self, path):
        """Load trained model from disk."""
        path = Path(path)
        self.model = keras.models.load_model(path / 'mlp_model.h5')
        self.is_trained = True
        print(f"✓ Model loaded from {path / 'mlp_model.h5'}")


def train_mlp(X_train, X_val, X_test, y_train, y_val, y_test,
              input_dim, hidden_layers=[128, 64, 32], dropout_rate=0.2,
              learning_rate=0.001, epochs=100, batch_size=32):
    """
    Train and evaluate MLP model.
    
    Args:
        X_train, X_val, X_test: Feature sets
        y_train, y_val, y_test: Target sets
        input_dim: Number of input features
        hidden_layers: Architecture
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        epochs: Number of epochs
        batch_size: Batch size
        
    Returns:
        MLPRegressor: Trained model
    """
    model = MLPRegressor(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    model.train(X_train, y_train, X_val, y_val, 
               epochs=epochs, batch_size=batch_size)
    model.evaluate(X_val, y_val, 'Validation')
    model.evaluate(X_test, y_test, 'Test')
    
    return model
