"""
Net Interest Income (NII) Model

Forecasts net interest income based on:
- Interest rate scenarios
- Balance sheet composition
- Asset-liability duration matching
- Credit risk adjustments
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings

from .base_model import BaseModel

class NIIModel(BaseModel):
    """
    Net Interest Income forecasting model.
    
    Features:
    - Interest rate sensitivity analysis
    - Balance sheet component modeling
    - Duration risk assessment
    - Scenario-based forecasting
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NII model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config, "NII_Model")
        
        # NII-specific configuration
        self.nii_config = config.get('models', {}).get('net_interest_income', {})
        self.lookback_period = self.nii_config.get('lookback_period', 60)
        self.forecast_horizon = self.nii_config.get('forecast_horizon', 36)
        self.balance_sheet_components = self.nii_config.get('balance_sheet_components', [])
        
        # Model components
        self.asset_yield_model = None
        self.liability_cost_model = None
        self.balance_model = None
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for NII modeling.
        
        Args:
            data: Raw data containing rates, balances, and economic indicators
            
        Returns:
            Feature matrix for modeling
        """
        features = pd.DataFrame(index=data.index)
        
        # Interest rate features
        if 'fed_funds_rate' in data.columns:
            features['fed_funds_rate'] = data['fed_funds_rate']
            features['fed_funds_rate_lag1'] = data['fed_funds_rate'].shift(1)
            features['fed_funds_rate_change'] = data['fed_funds_rate'].diff()
            features['fed_funds_rate_ma3'] = data['fed_funds_rate'].rolling(3).mean()
        
        # Yield curve features
        rate_columns = [col for col in data.columns if 'rate' in col.lower() or 'yield' in col.lower()]
        for col in rate_columns:
            if col != 'fed_funds_rate':
                features[f'{col}'] = data[col]
                features[f'{col}_change'] = data[col].diff()
        
        # Calculate yield curve slope and curvature
        if 'treasury_10y' in data.columns and 'treasury_2y' in data.columns:
            features['yield_curve_slope'] = data['treasury_10y'] - data['treasury_2y']
            
        if 'treasury_10y' in data.columns and 'treasury_2y' in data.columns and 'treasury_5y' in data.columns:
            features['yield_curve_curvature'] = 2 * data['treasury_5y'] - data['treasury_2y'] - data['treasury_10y']
        
        # Balance sheet features
        for component in self.balance_sheet_components:
            if component in data.columns:
                features[f'{component}_balance'] = data[component]
                features[f'{component}_growth'] = data[component].pct_change()
                features[f'{component}_ma6'] = data[component].rolling(6).mean()
        
        # Economic indicators
        econ_indicators = ['gdp_growth', 'unemployment_rate', 'inflation_rate', 'vix']
        for indicator in econ_indicators:
            if indicator in data.columns:
                features[indicator] = data[indicator]
                features[f'{indicator}_lag1'] = data[indicator].shift(1)
        
        # Seasonal features
        features['month'] = data.index.month
        features['quarter'] = data.index.quarter
        
        # Interaction terms
        if 'fed_funds_rate' in features.columns and 'loans_commercial_balance' in features.columns:
            features['rate_loan_interaction'] = features['fed_funds_rate'] * features['loans_commercial_balance']
        
        return features.fillna(method='ffill').fillna(0)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'NIIModel':
        """
        Fit the NII model.
        
        Args:
            X: Feature matrix containing rates, balances, and economic data
            y: Target NII values
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Fitting NII model...")
        
        # Validate data
        self.validate_data(X, y)
        
        # Prepare features
        features = self.prepare_features(X)
        
        # Store feature names and target
        self.feature_names = features.columns.tolist()
        self.target_name = y.name if y.name else "nii"
        
        # Fit ensemble model (Random Forest + Linear Regression)
        self.model = {
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'lr': LinearRegression()
        }
        
        # Fit models
        self.model['rf'].fit(features, y)
        self.model['lr'].fit(features, y)
        
        self.is_fitted = True
        self.logger.info("NII model fitted successfully")
        
        return self
    
    def predict(self, X: pd.DataFrame, scenario: str = 'base', **kwargs) -> np.ndarray:
        """
        Generate NII predictions.
        
        Args:
            X: Feature matrix for prediction
            scenario: Interest rate scenario ('base', 'rising', 'falling', 'shock')
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of NII predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare features
        features = self.prepare_features(X)
        
        # Apply scenario adjustments
        features_scenario = self._apply_scenario(features, scenario)
        
        # Generate predictions from both models
        rf_pred = self.model['rf'].predict(features_scenario)
        lr_pred = self.model['lr'].predict(features_scenario)
        
        # Ensemble prediction (weighted average)
        ensemble_pred = 0.7 * rf_pred + 0.3 * lr_pred
        
        return ensemble_pred
    
    def _apply_scenario(self, features: pd.DataFrame, scenario: str) -> pd.DataFrame:
        """
        Apply interest rate scenario adjustments to features.
        
        Args:
            features: Base feature matrix
            scenario: Scenario name
            
        Returns:
            Adjusted feature matrix
        """
        features_adj = features.copy()
        
        # Scenario adjustments
        scenario_adjustments = {
            'base': {},  # No adjustments
            'rising': {
                'fed_funds_rate': 2.0,  # +200 bps
                'treasury_2y': 1.8,
                'treasury_10y': 1.5
            },
            'falling': {
                'fed_funds_rate': -1.5,  # -150 bps
                'treasury_2y': -1.3,
                'treasury_10y': -1.0
            },
            'shock': {
                'fed_funds_rate': 3.0,  # +300 bps shock
                'treasury_2y': 2.5,
                'treasury_10y': 2.0
            }
        }
        
        adjustments = scenario_adjustments.get(scenario, {})
        
        for rate_var, adjustment in adjustments.items():
            if rate_var in features_adj.columns:
                features_adj[rate_var] += adjustment
                
                # Update derived features
                if f'{rate_var}_change' in features_adj.columns:
                    features_adj[f'{rate_var}_change'] = features_adj[rate_var].diff()
        
        # Recalculate yield curve features
        if 'treasury_10y' in features_adj.columns and 'treasury_2y' in features_adj.columns:
            features_adj['yield_curve_slope'] = features_adj['treasury_10y'] - features_adj['treasury_2y']
        
        return features_adj
    
    def forecast_nii(self, data: pd.DataFrame, scenarios: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate multi-scenario NII forecasts.
        
        Args:
            data: Historical data for forecasting
            scenarios: List of scenarios to forecast
            
        Returns:
            Dictionary of scenario forecasts
        """
        if scenarios is None:
            scenarios = ['base', 'rising', 'falling', 'shock']
        
        forecasts = {}
        
        for scenario in scenarios:
            self.logger.info(f"Generating forecast for scenario: {scenario}")
            
            # Generate forecast
            forecast = self.predict(data, scenario=scenario)
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'nii_forecast': forecast,
                'scenario': scenario
            }, index=data.index)
            
            forecasts[scenario] = forecast_df
        
        return forecasts
    
    def calculate_asset_sensitivity(self, data: pd.DataFrame, rate_shock: float = 1.0) -> Dict[str, float]:
        """
        Calculate asset sensitivity to interest rate changes.
        
        Args:
            data: Historical data
            rate_shock: Interest rate shock in percentage points
            
        Returns:
            Dictionary of sensitivity metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating sensitivity")
        
        # Base case prediction
        base_pred = self.predict(data, scenario='base')
        
        # Create shocked scenario
        shocked_data = data.copy()
        rate_columns = ['fed_funds_rate', 'treasury_2y', 'treasury_10y']
        
        for col in rate_columns:
            if col in shocked_data.columns:
                shocked_data[col] += rate_shock
        
        shocked_pred = self.predict(shocked_data, scenario='base')
        
        # Calculate sensitivity metrics
        nii_change = np.mean(shocked_pred - base_pred)
        nii_change_pct = (nii_change / np.mean(base_pred)) * 100
        
        sensitivity = {
            'rate_shock_bp': rate_shock * 100,  # basis points
            'nii_change_absolute': nii_change,
            'nii_change_percent': nii_change_pct,
            'asset_sensitive': nii_change > 0  # True if asset sensitive
        }
        
        return sensitivity