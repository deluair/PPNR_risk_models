"""
Model Validator for PPNR Models

Comprehensive validation framework including:
- Backtesting
- Model performance metrics
- Regulatory compliance checks
- Model stability analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.base_model import BaseModel

class ModelValidator:
    """
    Comprehensive model validation for PPNR models.
    
    Features:
    - Backtesting framework
    - Performance metrics calculation
    - Regulatory compliance validation
    - Model stability analysis
    - Champion-challenger testing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.validation_config = config.get('model_validation', {})
        self.regulatory_config = config.get('regulatory', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.ModelValidator")
        
        # Validation results storage
        self.validation_results = {}
        self.backtesting_results = {}
        
    def run_comprehensive_validation(self, model: BaseModel, 
                                   historical_data: pd.DataFrame,
                                   actual_outcomes: pd.DataFrame,
                                   validation_period: int = 12) -> Dict[str, Any]:
        """
        Run comprehensive model validation.
        
        Args:
            model: Fitted PPNR model
            historical_data: Historical input data
            actual_outcomes: Actual outcomes for validation
            validation_period: Validation period in months
            
        Returns:
            Comprehensive validation results
        """
        self.logger.info("Starting comprehensive model validation...")
        
        validation_results = {
            'model_name': model.__class__.__name__,
            'validation_date': datetime.now().isoformat(),
            'validation_period_months': validation_period
        }
        
        # 1. Backtesting
        backtest_results = self.run_backtesting(model, historical_data, actual_outcomes, validation_period)
        validation_results['backtesting'] = backtest_results
        
        # 2. Performance metrics
        performance_metrics = self.calculate_performance_metrics(model, historical_data, actual_outcomes)
        validation_results['performance_metrics'] = performance_metrics
        
        # 3. Model stability analysis
        stability_analysis = self.analyze_model_stability(model, historical_data, validation_period)
        validation_results['stability_analysis'] = stability_analysis
        
        # 4. Regulatory compliance checks
        compliance_results = self.check_regulatory_compliance(model, validation_results)
        validation_results['regulatory_compliance'] = compliance_results
        
        # 5. Residual analysis
        residual_analysis = self.analyze_residuals(model, historical_data, actual_outcomes)
        validation_results['residual_analysis'] = residual_analysis
        
        # Store results
        self.validation_results[model.__class__.__name__] = validation_results
        
        self.logger.info("Comprehensive model validation completed")
        return validation_results
    
    def run_backtesting(self, model: BaseModel, 
                       historical_data: pd.DataFrame,
                       actual_outcomes: pd.DataFrame,
                       validation_period: int = 12) -> Dict[str, Any]:
        """
        Run backtesting analysis.
        
        Args:
            model: Fitted PPNR model
            historical_data: Historical input data
            actual_outcomes: Actual outcomes
            validation_period: Validation period in months
            
        Returns:
            Backtesting results
        """
        self.logger.info("Running backtesting analysis...")
        
        # Prepare data for backtesting
        backtest_data = historical_data.tail(validation_period).copy()
        backtest_actuals = actual_outcomes.tail(validation_period).copy()
        
        # Generate predictions
        predictions = model.predict(backtest_data)
        
        # Ensure predictions and actuals are aligned
        if isinstance(predictions, pd.DataFrame):
            pred_values = predictions.values.flatten()
        else:
            pred_values = predictions
        
        if isinstance(backtest_actuals, pd.DataFrame):
            actual_values = backtest_actuals.values.flatten()
        else:
            actual_values = backtest_actuals
        
        # Calculate backtesting metrics
        backtest_results = {
            'prediction_accuracy': self._calculate_prediction_accuracy(pred_values, actual_values),
            'directional_accuracy': self._calculate_directional_accuracy(pred_values, actual_values),
            'forecast_errors': self._analyze_forecast_errors(pred_values, actual_values),
            'time_series_analysis': self._analyze_time_series_performance(pred_values, actual_values, backtest_data.index)
        }
        
        # Statistical tests
        backtest_results['statistical_tests'] = self._run_statistical_tests(pred_values, actual_values)
        
        return backtest_results
    
    def _calculate_prediction_accuracy(self, predictions: np.ndarray, 
                                     actuals: np.ndarray) -> Dict[str, float]:
        """Calculate prediction accuracy metrics."""
        # Remove any NaN values
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        pred_clean = predictions[mask]
        actual_clean = actuals[mask]
        
        if len(pred_clean) == 0:
            return {'error': 'No valid data points for accuracy calculation'}
        
        accuracy_metrics = {
            'mse': mean_squared_error(actual_clean, pred_clean),
            'rmse': np.sqrt(mean_squared_error(actual_clean, pred_clean)),
            'mae': mean_absolute_error(actual_clean, pred_clean),
            'mape': np.mean(np.abs((actual_clean - pred_clean) / actual_clean)) * 100,
            'r2_score': r2_score(actual_clean, pred_clean),
            'correlation': np.corrcoef(pred_clean, actual_clean)[0, 1]
        }
        
        return accuracy_metrics
    
    def _calculate_directional_accuracy(self, predictions: np.ndarray, 
                                      actuals: np.ndarray) -> Dict[str, float]:
        """Calculate directional accuracy metrics."""
        # Calculate period-over-period changes
        pred_changes = np.diff(predictions)
        actual_changes = np.diff(actuals)
        
        # Remove NaN values
        mask = ~(np.isnan(pred_changes) | np.isnan(actual_changes))
        pred_changes_clean = pred_changes[mask]
        actual_changes_clean = actual_changes[mask]
        
        if len(pred_changes_clean) == 0:
            return {'error': 'No valid data points for directional accuracy'}
        
        # Calculate directional accuracy
        correct_direction = np.sign(pred_changes_clean) == np.sign(actual_changes_clean)
        directional_accuracy = np.mean(correct_direction) * 100
        
        # Calculate turning point accuracy
        pred_turning_points = self._identify_turning_points(predictions)
        actual_turning_points = self._identify_turning_points(actuals)
        
        turning_point_accuracy = self._calculate_turning_point_accuracy(
            pred_turning_points, actual_turning_points
        )
        
        return {
            'directional_accuracy_pct': directional_accuracy,
            'turning_point_accuracy_pct': turning_point_accuracy,
            'correct_predictions': int(np.sum(correct_direction)),
            'total_predictions': len(correct_direction)
        }
    
    def _identify_turning_points(self, series: np.ndarray, window: int = 3) -> List[int]:
        """Identify turning points in a time series."""
        turning_points = []
        
        for i in range(window, len(series) - window):
            # Check for local maximum
            if all(series[i] > series[i-j] for j in range(1, window+1)) and \
               all(series[i] > series[i+j] for j in range(1, window+1)):
                turning_points.append(i)
            # Check for local minimum
            elif all(series[i] < series[i-j] for j in range(1, window+1)) and \
                 all(series[i] < series[i+j] for j in range(1, window+1)):
                turning_points.append(i)
        
        return turning_points
    
    def _calculate_turning_point_accuracy(self, pred_tp: List[int], 
                                        actual_tp: List[int], 
                                        tolerance: int = 2) -> float:
        """Calculate turning point accuracy with tolerance."""
        if not actual_tp:
            return 100.0 if not pred_tp else 0.0
        
        correct_tp = 0
        for actual_point in actual_tp:
            # Check if any predicted turning point is within tolerance
            if any(abs(pred_point - actual_point) <= tolerance for pred_point in pred_tp):
                correct_tp += 1
        
        return (correct_tp / len(actual_tp)) * 100
    
    def _analyze_forecast_errors(self, predictions: np.ndarray, 
                               actuals: np.ndarray) -> Dict[str, Any]:
        """Analyze forecast error patterns."""
        errors = predictions - actuals
        
        # Remove NaN values
        errors_clean = errors[~np.isnan(errors)]
        
        if len(errors_clean) == 0:
            return {'error': 'No valid errors for analysis'}
        
        error_analysis = {
            'mean_error': np.mean(errors_clean),
            'std_error': np.std(errors_clean),
            'min_error': np.min(errors_clean),
            'max_error': np.max(errors_clean),
            'error_percentiles': {
                '5th': np.percentile(errors_clean, 5),
                '25th': np.percentile(errors_clean, 25),
                '50th': np.percentile(errors_clean, 50),
                '75th': np.percentile(errors_clean, 75),
                '95th': np.percentile(errors_clean, 95)
            },
            'bias_test': {
                'mean_error': np.mean(errors_clean),
                'is_biased': abs(np.mean(errors_clean)) > 2 * (np.std(errors_clean) / np.sqrt(len(errors_clean)))
            }
        }
        
        return error_analysis
    
    def _analyze_time_series_performance(self, predictions: np.ndarray, 
                                       actuals: np.ndarray,
                                       dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """Analyze time series performance patterns."""
        # Create DataFrame for analysis
        ts_data = pd.DataFrame({
            'predictions': predictions,
            'actuals': actuals,
            'errors': predictions - actuals
        }, index=dates)
        
        # Remove NaN values
        ts_data = ts_data.dropna()
        
        if len(ts_data) == 0:
            return {'error': 'No valid data for time series analysis'}
        
        # Calculate rolling metrics
        window = min(6, len(ts_data) // 2)  # 6-month rolling window or half the data
        
        rolling_metrics = {
            'rolling_mse': ts_data['errors'].rolling(window=window).apply(lambda x: np.mean(x**2)).tolist(),
            'rolling_correlation': ts_data[['predictions', 'actuals']].rolling(window=window).corr().iloc[0::2, -1].tolist()
        }
        
        # Seasonal analysis if enough data
        seasonal_analysis = {}
        if len(ts_data) >= 12:
            ts_data['month'] = ts_data.index.month
            monthly_performance = ts_data.groupby('month').agg({
                'errors': ['mean', 'std'],
                'predictions': 'mean',
                'actuals': 'mean'
            }).round(4)
            
            seasonal_analysis = {
                'monthly_performance': monthly_performance.to_dict(),
                'seasonal_bias_detected': self._test_seasonal_bias(ts_data)
            }
        
        return {
            'rolling_metrics': rolling_metrics,
            'seasonal_analysis': seasonal_analysis,
            'trend_analysis': self._analyze_trend_performance(ts_data)
        }
    
    def _test_seasonal_bias(self, ts_data: pd.DataFrame) -> bool:
        """Test for seasonal bias in errors."""
        try:
            from scipy.stats import kruskal
            monthly_errors = [group['errors'].values for name, group in ts_data.groupby('month')]
            # Filter out empty groups
            monthly_errors = [errors for errors in monthly_errors if len(errors) > 0]
            
            if len(monthly_errors) < 3:
                return False
            
            statistic, p_value = kruskal(*monthly_errors)
            return p_value < 0.05
        except:
            return False
    
    def _analyze_trend_performance(self, ts_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance during different trend periods."""
        # Calculate trends
        ts_data['actual_trend'] = ts_data['actuals'].diff().rolling(3).mean()
        ts_data['pred_trend'] = ts_data['predictions'].diff().rolling(3).mean()
        
        # Classify trend periods
        ts_data['trend_period'] = 'stable'
        ts_data.loc[ts_data['actual_trend'] > 0.1, 'trend_period'] = 'uptrend'
        ts_data.loc[ts_data['actual_trend'] < -0.1, 'trend_period'] = 'downtrend'
        
        # Calculate performance by trend period
        trend_performance = {}
        for trend in ['uptrend', 'downtrend', 'stable']:
            trend_data = ts_data[ts_data['trend_period'] == trend]
            if len(trend_data) > 0:
                trend_performance[trend] = {
                    'count': len(trend_data),
                    'mse': np.mean(trend_data['errors']**2),
                    'mae': np.mean(np.abs(trend_data['errors'])),
                    'correlation': np.corrcoef(trend_data['predictions'], trend_data['actuals'])[0, 1]
                }
        
        return trend_performance
    
    def _run_statistical_tests(self, predictions: np.ndarray, 
                             actuals: np.ndarray) -> Dict[str, Any]:
        """Run statistical tests on predictions vs actuals."""
        # Remove NaN values
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        pred_clean = predictions[mask]
        actual_clean = actuals[mask]
        
        if len(pred_clean) < 10:
            return {'error': 'Insufficient data for statistical tests'}
        
        errors = pred_clean - actual_clean
        
        statistical_tests = {}
        
        # Normality test for errors
        try:
            shapiro_stat, shapiro_p = stats.shapiro(errors)
            statistical_tests['normality_test'] = {
                'test': 'Shapiro-Wilk',
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'is_normal': shapiro_p > 0.05
            }
        except:
            statistical_tests['normality_test'] = {'error': 'Could not perform normality test'}
        
        # Bias test (t-test for zero mean)
        try:
            t_stat, t_p = stats.ttest_1samp(errors, 0)
            statistical_tests['bias_test'] = {
                'test': 'One-sample t-test',
                'statistic': float(t_stat),
                'p_value': float(t_p),
                'is_unbiased': t_p > 0.05
            }
        except:
            statistical_tests['bias_test'] = {'error': 'Could not perform bias test'}
        
        # Autocorrelation test for errors
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_stat, lb_p = acorr_ljungbox(errors, lags=min(10, len(errors)//4), return_df=False)
            statistical_tests['autocorrelation_test'] = {
                'test': 'Ljung-Box',
                'statistic': float(lb_stat.iloc[-1]) if hasattr(lb_stat, 'iloc') else float(lb_stat),
                'p_value': float(lb_p.iloc[-1]) if hasattr(lb_p, 'iloc') else float(lb_p),
                'no_autocorrelation': (lb_p.iloc[-1] if hasattr(lb_p, 'iloc') else lb_p) > 0.05
            }
        except:
            statistical_tests['autocorrelation_test'] = {'error': 'Could not perform autocorrelation test'}
        
        return statistical_tests
    
    def calculate_performance_metrics(self, model: BaseModel,
                                    historical_data: pd.DataFrame,
                                    actual_outcomes: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        self.logger.info("Calculating performance metrics...")
        
        # Generate predictions
        predictions = model.predict(historical_data)
        
        # Basic performance metrics
        basic_metrics = self._calculate_prediction_accuracy(
            predictions.values.flatten() if isinstance(predictions, pd.DataFrame) else predictions,
            actual_outcomes.values.flatten() if isinstance(actual_outcomes, pd.DataFrame) else actual_outcomes
        )
        
        # Model-specific metrics
        model_specific_metrics = {}
        
        # Feature importance if available
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            if importance is not None:
                model_specific_metrics['feature_importance'] = importance.head(10).to_dict()
        
        # Model complexity metrics
        complexity_metrics = self._calculate_model_complexity(model)
        
        return {
            'basic_metrics': basic_metrics,
            'model_specific_metrics': model_specific_metrics,
            'complexity_metrics': complexity_metrics
        }
    
    def _calculate_model_complexity(self, model: BaseModel) -> Dict[str, Any]:
        """Calculate model complexity metrics."""
        complexity = {}
        
        # Number of features
        if hasattr(model, 'feature_names_'):
            complexity['n_features'] = len(model.feature_names_)
        
        # Model parameters (if available)
        if hasattr(model, 'get_params'):
            params = model.get_params()
            complexity['n_parameters'] = len(params)
        
        # Model type
        complexity['model_type'] = model.__class__.__name__
        
        return complexity
    
    def analyze_model_stability(self, model: BaseModel,
                              historical_data: pd.DataFrame,
                              validation_period: int = 12) -> Dict[str, Any]:
        """Analyze model stability over time."""
        self.logger.info("Analyzing model stability...")
        
        stability_results = {}
        
        # Parameter stability (if model supports it)
        if hasattr(model, 'get_params'):
            stability_results['parameter_stability'] = self._analyze_parameter_stability(model, historical_data)
        
        # Prediction stability
        stability_results['prediction_stability'] = self._analyze_prediction_stability(model, historical_data, validation_period)
        
        # Feature importance stability
        if hasattr(model, 'get_feature_importance'):
            stability_results['feature_importance_stability'] = self._analyze_feature_importance_stability(model, historical_data)
        
        return stability_results
    
    def _analyze_parameter_stability(self, model: BaseModel, 
                                   historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze parameter stability over time."""
        # This would require retraining the model on different time windows
        # For now, return basic information
        return {
            'analysis': 'Parameter stability analysis requires model retraining',
            'current_parameters': model.get_params() if hasattr(model, 'get_params') else {}
        }
    
    def _analyze_prediction_stability(self, model: BaseModel,
                                    historical_data: pd.DataFrame,
                                    validation_period: int) -> Dict[str, Any]:
        """Analyze prediction stability."""
        # Generate predictions for different time windows
        window_size = len(historical_data) // 4  # Use quarter of data as window
        
        if window_size < validation_period:
            return {'error': 'Insufficient data for stability analysis'}
        
        predictions_by_window = []
        
        for i in range(0, len(historical_data) - window_size + 1, window_size // 2):
            window_data = historical_data.iloc[i:i + window_size]
            if len(window_data) >= validation_period:
                window_predictions = model.predict(window_data.tail(validation_period))
                predictions_by_window.append(window_predictions)
        
        if len(predictions_by_window) < 2:
            return {'error': 'Insufficient windows for stability analysis'}
        
        # Calculate stability metrics
        pred_correlations = []
        for i in range(len(predictions_by_window) - 1):
            pred1 = predictions_by_window[i].values.flatten() if isinstance(predictions_by_window[i], pd.DataFrame) else predictions_by_window[i]
            pred2 = predictions_by_window[i + 1].values.flatten() if isinstance(predictions_by_window[i + 1], pd.DataFrame) else predictions_by_window[i + 1]
            
            # Ensure same length
            min_len = min(len(pred1), len(pred2))
            corr = np.corrcoef(pred1[:min_len], pred2[:min_len])[0, 1]
            pred_correlations.append(corr)
        
        return {
            'prediction_correlations': pred_correlations,
            'mean_correlation': np.mean(pred_correlations),
            'stability_score': np.mean(pred_correlations) * 100,  # Convert to percentage
            'n_windows_analyzed': len(predictions_by_window)
        }
    
    def _analyze_feature_importance_stability(self, model: BaseModel,
                                            historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature importance stability."""
        # Get current feature importance
        importance = model.get_feature_importance()
        
        if importance is None:
            return {'error': 'Model does not support feature importance'}
        
        # For now, return current importance
        # Full analysis would require retraining on different periods
        return {
            'current_importance': importance.head(10).to_dict(),
            'analysis': 'Full feature importance stability requires model retraining on different periods'
        }
    
    def check_regulatory_compliance(self, model: BaseModel,
                                  validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check regulatory compliance requirements."""
        self.logger.info("Checking regulatory compliance...")
        
        compliance_results = {
            'overall_compliance': True,
            'compliance_checks': {}
        }
        
        # Get regulatory requirements
        sr_11_7_requirements = self.regulatory_config.get('sr_11_7', {})
        
        # Check model performance requirements
        performance_checks = self._check_performance_requirements(validation_results, sr_11_7_requirements)
        compliance_results['compliance_checks']['performance'] = performance_checks
        
        # Check documentation requirements
        documentation_checks = self._check_documentation_requirements(model)
        compliance_results['compliance_checks']['documentation'] = documentation_checks
        
        # Check validation requirements
        validation_checks = self._check_validation_requirements(validation_results)
        compliance_results['compliance_checks']['validation'] = validation_checks
        
        # Update overall compliance
        all_checks = [performance_checks, documentation_checks, validation_checks]
        compliance_results['overall_compliance'] = all(
            check.get('compliant', False) for check in all_checks
        )
        
        return compliance_results
    
    def _check_performance_requirements(self, validation_results: Dict[str, Any],
                                      requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Check performance requirements."""
        performance_check = {'compliant': True, 'issues': []}
        
        # Check R-squared threshold
        r2_threshold = requirements.get('min_r2', 0.7)
        if 'performance_metrics' in validation_results:
            basic_metrics = validation_results['performance_metrics'].get('basic_metrics', {})
            r2_score = basic_metrics.get('r2_score', 0)
            
            if r2_score < r2_threshold:
                performance_check['compliant'] = False
                performance_check['issues'].append(f"R-squared ({r2_score:.3f}) below threshold ({r2_threshold})")
        
        # Check bias test
        if 'backtesting' in validation_results:
            statistical_tests = validation_results['backtesting'].get('statistical_tests', {})
            bias_test = statistical_tests.get('bias_test', {})
            
            if not bias_test.get('is_unbiased', True):
                performance_check['compliant'] = False
                performance_check['issues'].append("Model shows significant bias")
        
        return performance_check
    
    def _check_documentation_requirements(self, model: BaseModel) -> Dict[str, Any]:
        """Check documentation requirements."""
        doc_check = {'compliant': True, 'issues': []}
        
        # Check if model has documentation
        if not hasattr(model, '__doc__') or not model.__doc__:
            doc_check['compliant'] = False
            doc_check['issues'].append("Model lacks proper documentation")
        
        # Check if model has required methods
        required_methods = ['fit', 'predict', 'get_performance_metrics']
        for method in required_methods:
            if not hasattr(model, method):
                doc_check['compliant'] = False
                doc_check['issues'].append(f"Model missing required method: {method}")
        
        return doc_check
    
    def _check_validation_requirements(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check validation requirements."""
        validation_check = {'compliant': True, 'issues': []}
        
        # Check if backtesting was performed
        if 'backtesting' not in validation_results:
            validation_check['compliant'] = False
            validation_check['issues'].append("Backtesting not performed")
        
        # Check if performance metrics were calculated
        if 'performance_metrics' not in validation_results:
            validation_check['compliant'] = False
            validation_check['issues'].append("Performance metrics not calculated")
        
        # Check validation period length
        validation_period = validation_results.get('validation_period_months', 0)
        min_validation_period = self.regulatory_config.get('min_validation_period_months', 12)
        
        if validation_period < min_validation_period:
            validation_check['compliant'] = False
            validation_check['issues'].append(f"Validation period ({validation_period} months) below minimum ({min_validation_period} months)")
        
        return validation_check
    
    def analyze_residuals(self, model: BaseModel,
                         historical_data: pd.DataFrame,
                         actual_outcomes: pd.DataFrame) -> Dict[str, Any]:
        """Analyze model residuals."""
        self.logger.info("Analyzing residuals...")
        
        # Generate predictions
        predictions = model.predict(historical_data)
        
        # Calculate residuals
        pred_values = predictions.values.flatten() if isinstance(predictions, pd.DataFrame) else predictions
        actual_values = actual_outcomes.values.flatten() if isinstance(actual_outcomes, pd.DataFrame) else actual_outcomes
        
        # Ensure same length
        min_len = min(len(pred_values), len(actual_values))
        residuals = actual_values[:min_len] - pred_values[:min_len]
        
        # Remove NaN values
        residuals_clean = residuals[~np.isnan(residuals)]
        
        if len(residuals_clean) == 0:
            return {'error': 'No valid residuals for analysis'}
        
        residual_analysis = {
            'basic_statistics': {
                'mean': float(np.mean(residuals_clean)),
                'std': float(np.std(residuals_clean)),
                'min': float(np.min(residuals_clean)),
                'max': float(np.max(residuals_clean)),
                'skewness': float(stats.skew(residuals_clean)),
                'kurtosis': float(stats.kurtosis(residuals_clean))
            },
            'distribution_tests': self._test_residual_distribution(residuals_clean),
            'autocorrelation_analysis': self._analyze_residual_autocorrelation(residuals_clean),
            'heteroscedasticity_test': self._test_heteroscedasticity(residuals_clean, pred_values[:min_len])
        }
        
        return residual_analysis
    
    def _test_residual_distribution(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test residual distribution properties."""
        distribution_tests = {}
        
        # Normality test
        try:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            distribution_tests['normality'] = {
                'test': 'Shapiro-Wilk',
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'is_normal': shapiro_p > 0.05
            }
        except:
            distribution_tests['normality'] = {'error': 'Could not perform normality test'}
        
        # Zero mean test
        try:
            t_stat, t_p = stats.ttest_1samp(residuals, 0)
            distribution_tests['zero_mean'] = {
                'test': 'One-sample t-test',
                'statistic': float(t_stat),
                'p_value': float(t_p),
                'has_zero_mean': t_p > 0.05
            }
        except:
            distribution_tests['zero_mean'] = {'error': 'Could not perform zero mean test'}
        
        return distribution_tests
    
    def _analyze_residual_autocorrelation(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Analyze residual autocorrelation."""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            # Ljung-Box test for autocorrelation
            lags = min(10, len(residuals) // 4)
            lb_stat, lb_p = acorr_ljungbox(residuals, lags=lags, return_df=False)
            
            return {
                'ljung_box_test': {
                    'statistic': float(lb_stat.iloc[-1]) if hasattr(lb_stat, 'iloc') else float(lb_stat),
                    'p_value': float(lb_p.iloc[-1]) if hasattr(lb_p, 'iloc') else float(lb_p),
                    'no_autocorrelation': (lb_p.iloc[-1] if hasattr(lb_p, 'iloc') else lb_p) > 0.05
                }
            }
        except:
            return {'error': 'Could not perform autocorrelation analysis'}
    
    def _test_heteroscedasticity(self, residuals: np.ndarray, 
                               predictions: np.ndarray) -> Dict[str, Any]:
        """Test for heteroscedasticity in residuals."""
        try:
            # Breusch-Pagan test approximation
            # Regress squared residuals on predictions
            squared_residuals = residuals ** 2
            
            # Simple correlation test
            correlation = np.corrcoef(squared_residuals, predictions)[0, 1]
            
            # Simple test: if correlation is significant, heteroscedasticity is present
            n = len(residuals)
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            return {
                'heteroscedasticity_test': {
                    'correlation': float(correlation),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'homoscedastic': p_value > 0.05
                }
            }
        except:
            return {'error': 'Could not perform heteroscedasticity test'}
    
    def generate_validation_report(self, validation_results: Dict[str, Any],
                                 output_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        self.logger.info("Generating validation report...")
        
        report = {
            'metadata': {
                'report_date': datetime.now().isoformat(),
                'model_name': validation_results.get('model_name', 'Unknown'),
                'validation_period': validation_results.get('validation_period_months', 'Unknown')
            },
            'executive_summary': self._create_executive_summary(validation_results),
            'detailed_results': validation_results,
            'recommendations': self._generate_recommendations(validation_results)
        }
        
        # Save report if path provided
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Validation report saved to {output_path}")
        
        return report
    
    def _create_executive_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of validation results."""
        summary = {
            'overall_assessment': 'PASS',
            'key_findings': [],
            'critical_issues': [],
            'performance_summary': {}
        }
        
        # Check performance metrics
        if 'performance_metrics' in validation_results:
            basic_metrics = validation_results['performance_metrics'].get('basic_metrics', {})
            r2_score = basic_metrics.get('r2_score', 0)
            
            summary['performance_summary']['r2_score'] = r2_score
            
            if r2_score < 0.7:
                summary['overall_assessment'] = 'CONDITIONAL'
                summary['critical_issues'].append(f"Low R-squared score: {r2_score:.3f}")
            else:
                summary['key_findings'].append(f"Good model fit with R-squared: {r2_score:.3f}")
        
        # Check regulatory compliance
        if 'regulatory_compliance' in validation_results:
            compliance = validation_results['regulatory_compliance']
            if not compliance.get('overall_compliance', False):
                summary['overall_assessment'] = 'FAIL'
                summary['critical_issues'].append("Regulatory compliance issues identified")
            else:
                summary['key_findings'].append("Model meets regulatory compliance requirements")
        
        # Check backtesting results
        if 'backtesting' in validation_results:
            backtest = validation_results['backtesting']
            directional_acc = backtest.get('directional_accuracy', {}).get('directional_accuracy_pct', 0)
            
            if directional_acc < 60:
                summary['critical_issues'].append(f"Low directional accuracy: {directional_acc:.1f}%")
            else:
                summary['key_findings'].append(f"Good directional accuracy: {directional_acc:.1f}%")
        
        return summary
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Performance-based recommendations
        if 'performance_metrics' in validation_results:
            basic_metrics = validation_results['performance_metrics'].get('basic_metrics', {})
            r2_score = basic_metrics.get('r2_score', 0)
            
            if r2_score < 0.7:
                recommendations.append("Consider model re-specification or additional features to improve fit")
        
        # Backtesting recommendations
        if 'backtesting' in validation_results:
            backtest = validation_results['backtesting']
            
            # Check for bias
            statistical_tests = backtest.get('statistical_tests', {})
            bias_test = statistical_tests.get('bias_test', {})
            
            if not bias_test.get('is_unbiased', True):
                recommendations.append("Address model bias through recalibration or methodology adjustment")
            
            # Check directional accuracy
            directional_acc = backtest.get('directional_accuracy', {}).get('directional_accuracy_pct', 0)
            if directional_acc < 60:
                recommendations.append("Improve directional accuracy through better feature engineering or model selection")
        
        # Residual analysis recommendations
        if 'residual_analysis' in validation_results:
            residual_analysis = validation_results['residual_analysis']
            
            # Check normality
            distribution_tests = residual_analysis.get('distribution_tests', {})
            normality = distribution_tests.get('normality', {})
            
            if not normality.get('is_normal', True):
                recommendations.append("Consider transformation of target variable or alternative modeling approach for non-normal residuals")
            
            # Check heteroscedasticity
            hetero_test = residual_analysis.get('heteroscedasticity_test', {}).get('heteroscedasticity_test', {})
            if not hetero_test.get('homoscedastic', True):
                recommendations.append("Address heteroscedasticity through robust standard errors or model transformation")
        
        # Compliance recommendations
        if 'regulatory_compliance' in validation_results:
            compliance = validation_results['regulatory_compliance']
            if not compliance.get('overall_compliance', False):
                recommendations.append("Address regulatory compliance issues before model deployment")
        
        # Default recommendation if no issues found
        if not recommendations:
            recommendations.append("Model validation successful - proceed with deployment and ongoing monitoring")
        
        return recommendations