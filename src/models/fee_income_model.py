"""
Fee Income Model

Forecasts non-interest income components including:
- Service charges and fees
- Investment banking revenue
- Trading fees and commissions
- Card and payment processing
- Trust and wealth management fees
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import warnings

from .base_model import BaseModel

class FeeIncomeModel(BaseModel):
    """
    Fee income forecasting model for multiple revenue streams.
    
    Features:
    - Multi-output modeling for different fee categories
    - Seasonality adjustment
    - Economic cycle sensitivity
    - Market volatility impact modeling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Fee Income model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config, "Fee_Income_Model")
        
        # Fee income specific configuration
        self.fee_config = config.get('models', {}).get('fee_income', {})
        self.components = self.fee_config.get('components', [])
        self.seasonality_adjustment = self.fee_config.get('seasonality_adjustment', True)
        
        # Model components
        self.scaler = StandardScaler()
        self.component_models = {}
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for fee income modeling.
        
        Args:
            data: Raw data containing market indicators and economic data
            
        Returns:
            Feature matrix for modeling
        """
        features = pd.DataFrame(index=data.index)
        
        # Market indicators
        market_indicators = ['vix', 'sp500_return', 'nasdaq_return', 'bond_volatility']
        for indicator in market_indicators:
            if indicator in data.columns:
                features[indicator] = data[indicator]
                features[f'{indicator}_lag1'] = data[indicator].shift(1)
                features[f'{indicator}_ma3'] = data[indicator].rolling(3).mean()
                features[f'{indicator}_volatility'] = data[indicator].rolling(12).std()
        
        # Economic indicators
        econ_indicators = ['gdp_growth', 'unemployment_rate', 'consumer_confidence', 'ism_manufacturing']
        for indicator in econ_indicators:
            if indicator in data.columns:
                features[indicator] = data[indicator]
                features[f'{indicator}_lag1'] = data[indicator].shift(1)
                features[f'{indicator}_yoy'] = data[indicator].pct_change(12)
        
        # Interest rate environment
        if 'fed_funds_rate' in data.columns:
            features['fed_funds_rate'] = data['fed_funds_rate']
            features['rate_environment'] = self._classify_rate_environment(data['fed_funds_rate'])
        
        # Market activity proxies
        if 'trading_volume' in data.columns:
            features['trading_volume'] = data['trading_volume']
            features['trading_volume_ma6'] = data['trading_volume'].rolling(6).mean()
            features['volume_trend'] = data['trading_volume'].pct_change(3)
        
        # Credit market conditions
        if 'credit_spreads' in data.columns:
            features['credit_spreads'] = data['credit_spreads']
            features['credit_spreads_change'] = data['credit_spreads'].diff()
            features['credit_stress'] = (data['credit_spreads'] > data['credit_spreads'].rolling(12).quantile(0.75)).astype(int)
        
        # Seasonal features
        if self.seasonality_adjustment:
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            features['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
            features['quarter_sin'] = np.sin(2 * np.pi * data.index.quarter / 4)
            features['quarter_cos'] = np.cos(2 * np.pi * data.index.quarter / 4)
        
        # Interaction terms
        if 'vix' in features.columns and 'sp500_return' in features.columns:
            features['vix_return_interaction'] = features['vix'] * features['sp500_return']
        
        if 'gdp_growth' in features.columns and 'consumer_confidence' in features.columns:
            features['growth_confidence_interaction'] = features['gdp_growth'] * features['consumer_confidence']
        
        return features.fillna(method='ffill').fillna(0)
    
    def _classify_rate_environment(self, rates: pd.Series) -> pd.Series:
        """
        Classify interest rate environment.
        
        Args:
            rates: Interest rate series
            
        Returns:
            Classified rate environment (0=low, 1=normal, 2=high)
        """
        rate_percentiles = rates.rolling(60).quantile([0.33, 0.67])
        
        environment = pd.Series(1, index=rates.index)  # Default to normal
        environment[rates <= rate_percentiles.iloc[:, 0]] = 0  # Low rates
        environment[rates >= rate_percentiles.iloc[:, 1]] = 2  # High rates
        
        return environment
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'FeeIncomeModel':
        """
        Fit the fee income model.
        
        Args:
            X: Feature matrix containing market and economic data
            y: Target fee income components (DataFrame with multiple columns)
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Fitting Fee Income model...")
        
        # Validate data
        self.validate_data(X)
        
        # Prepare features
        features = self.prepare_features(X)
        
        # Scale features
        features_scaled = pd.DataFrame(
            self.scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Fit individual models for each fee component
        for component in y.columns:
            self.logger.info(f"Fitting model for {component}")
            
            # Create component-specific model
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            # Fit model
            valid_idx = ~y[component].isna()
            model.fit(features_scaled[valid_idx], y[component][valid_idx])
            
            self.component_models[component] = model
        
        self.is_fitted = True
        self.logger.info("Fee Income model fitted successfully")
        
        return self
    
    def predict(self, X: pd.DataFrame, components: List[str] = None, **kwargs) -> pd.DataFrame:
        """
        Generate fee income predictions.
        
        Args:
            X: Feature matrix for prediction
            components: List of components to predict (None for all)
            **kwargs: Additional prediction parameters
            
        Returns:
            DataFrame of fee income predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare features
        features = self.prepare_features(X)
        features_scaled = pd.DataFrame(
            self.scaler.transform(features),
            columns=features.columns,
            index=features.index
        )
        
        # Determine components to predict
        if components is None:
            components = list(self.component_models.keys())
        
        # Generate predictions
        predictions = pd.DataFrame(index=X.index)
        
        for component in components:
            if component in self.component_models:
                pred = self.component_models[component].predict(features_scaled)
                predictions[f'{component}_forecast'] = pred
            else:
                self.logger.warning(f"No model found for component: {component}")
        
        return predictions
    
    def forecast_stress_scenario(self, X: pd.DataFrame, stress_factors: Dict[str, float]) -> pd.DataFrame:
        """
        Generate fee income forecasts under stress scenarios.
        
        Args:
            X: Base feature matrix
            stress_factors: Dictionary of stress adjustments
            
        Returns:
            DataFrame of stressed fee income predictions
        """
        # Create stressed features
        stressed_features = self.prepare_features(X).copy()
        
        # Apply stress factors
        for factor, adjustment in stress_factors.items():
            if factor in stressed_features.columns:
                if factor == 'vix':
                    # VIX stress - multiplicative
                    stressed_features[factor] *= (1 + adjustment)
                elif 'return' in factor:
                    # Return stress - additive
                    stressed_features[factor] += adjustment
                elif factor == 'gdp_growth':
                    # GDP stress - additive
                    stressed_features[factor] += adjustment
                elif factor == 'unemployment_rate':
                    # Unemployment stress - additive
                    stressed_features[factor] += adjustment
        
        # Scale stressed features
        stressed_features_scaled = pd.DataFrame(
            self.scaler.transform(stressed_features),
            columns=stressed_features.columns,
            index=stressed_features.index
        )
        
        # Generate stressed predictions
        stressed_predictions = pd.DataFrame(index=X.index)
        
        for component, model in self.component_models.items():
            pred = model.predict(stressed_features_scaled)
            stressed_predictions[f'{component}_stressed'] = pred
        
        return stressed_predictions
    
    def analyze_component_sensitivity(self, X: pd.DataFrame, component: str) -> Dict[str, float]:
        """
        Analyze sensitivity of a fee component to various factors.
        
        Args:
            X: Feature matrix
            component: Fee component to analyze
            
        Returns:
            Dictionary of sensitivity metrics
        """
        if component not in self.component_models:
            raise ValueError(f"No model found for component: {component}")
        
        # Base prediction
        base_features = self.prepare_features(X)
        base_features_scaled = self.scaler.transform(base_features)
        base_pred = self.component_models[component].predict(base_features_scaled)
        
        sensitivities = {}
        
        # Test sensitivity to key factors
        test_factors = {
            'vix': 0.2,  # 20% increase in VIX
            'sp500_return': -0.1,  # -10% market return
            'gdp_growth': -0.02,  # -2% GDP growth
            'unemployment_rate': 0.02,  # +2% unemployment
            'credit_spreads': 0.5  # +50bps credit spreads
        }
        
        for factor, shock in test_factors.items():
            if factor in base_features.columns:
                # Create shocked features
                shocked_features = base_features.copy()
                shocked_features[factor] += shock
                
                # Scale and predict
                shocked_features_scaled = self.scaler.transform(shocked_features)
                shocked_pred = self.component_models[component].predict(shocked_features_scaled)
                
                # Calculate sensitivity
                sensitivity = np.mean(shocked_pred - base_pred) / np.mean(base_pred) * 100
                sensitivities[f'{factor}_sensitivity_pct'] = sensitivity
        
        return sensitivities
    
    def get_component_importance(self, component: str) -> Optional[pd.Series]:
        """
        Get feature importance for a specific fee component.
        
        Args:
            component: Fee component name
            
        Returns:
            Series of feature importances
        """
        if component not in self.component_models:
            self.logger.warning(f"No model found for component: {component}")
            return None
        
        model = self.component_models[component]
        
        if hasattr(model, 'feature_importances_'):
            return pd.Series(
                model.feature_importances_,
                index=self.feature_names,
                name=f'{component}_importance'
            ).sort_values(ascending=False)
        else:
            return None