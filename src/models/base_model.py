"""
Base Model Class for PPNR Risk Models

Provides common functionality and interface for all PPNR model components.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yaml

class BaseModel(ABC):
    """
    Abstract base class for all PPNR models.
    
    Provides common functionality including:
    - Data validation
    - Model fitting interface
    - Prediction interface
    - Performance evaluation
    - Model persistence
    """
    
    def __init__(self, config: Dict[str, Any], model_name: str):
        """
        Initialize base model.
        
        Args:
            config: Model configuration dictionary
            model_name: Name of the specific model
        """
        self.config = config
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.feature_names = []
        self.target_name = ""
        
        # Set up logging
        self.logger = logging.getLogger(f"PPNR.{model_name}")
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate predictions using the fitted model.
        
        Args:
            X: Feature matrix for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of predictions
        """
        pass
    
    def validate_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        Validate input data for common issues.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
        """
        # Check for missing values
        if X.isnull().any().any():
            self.logger.warning("Missing values detected in features")
            
        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            raise ValueError("Infinite values detected in features")
            
        # Check target variable if provided
        if y is not None:
            if y.isnull().any():
                self.logger.warning("Missing values detected in target variable")
            if np.isinf(y).any():
                raise ValueError("Infinite values detected in target variable")
    
    def evaluate_performance(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance using standard metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'r_squared': r2_score(y_true, y_pred)
        }
        
        return metrics
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance if available from the model.
        
        Returns:
            Series of feature importances or None
        """
        if not self.is_fitted:
            self.logger.warning("Model not fitted. Cannot get feature importance.")
            return None
            
        # Try to get feature importance from common model types
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(
                self.model.feature_importances_,
                index=self.feature_names,
                name='importance'
            ).sort_values(ascending=False)
        elif hasattr(self.model, 'coef_'):
            return pd.Series(
                np.abs(self.model.coef_),
                index=self.feature_names,
                name='importance'
            ).sort_values(ascending=False)
        else:
            self.logger.info("Feature importance not available for this model type")
            return None
    
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted model to disk.
        
        Args:
            filepath: Path to save the model
        """
        import joblib
        
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
            
        model_data = {
            'model': self.model,
            'config': self.config,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a fitted model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.model_name = model_data['model_name']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
        self.is_fitted = model_data['is_fitted']
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def generate_model_report(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Generate comprehensive model performance report.
        
        Args:
            X_test: Test feature matrix
            y_test: Test target values
            
        Returns:
            Dictionary containing model report
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating report")
            
        # Generate predictions
        y_pred = self.predict(X_test)
        
        # Calculate performance metrics
        performance = self.evaluate_performance(y_test, y_pred)
        
        # Get feature importance
        feature_importance = self.get_feature_importance()
        
        report = {
            'model_name': self.model_name,
            'performance_metrics': performance,
            'feature_importance': feature_importance.to_dict() if feature_importance is not None else None,
            'prediction_summary': {
                'mean_prediction': float(np.mean(y_pred)),
                'std_prediction': float(np.std(y_pred)),
                'min_prediction': float(np.min(y_pred)),
                'max_prediction': float(np.max(y_pred))
            },
            'data_summary': {
                'n_features': len(self.feature_names),
                'n_observations': len(X_test),
                'feature_names': self.feature_names
            }
        }
        
        return report