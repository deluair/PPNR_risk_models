"""
Trading Revenue Model

Forecasts trading revenue based on:
- Market risk factors (equity, FX, credit, commodity volatility)
- VaR-based revenue modeling
- Correlation analysis across trading desks
- Stress testing under extreme market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from scipy import stats
import warnings

from .base_model import BaseModel

class TradingRevenueModel(BaseModel):
    """
    Trading revenue forecasting model with risk factor integration.
    
    Features:
    - Multi-factor risk model
    - VaR-based revenue forecasting
    - Desk-level correlation modeling
    - Extreme scenario stress testing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Trading Revenue model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config, "Trading_Revenue_Model")
        
        # Trading revenue specific configuration
        self.trading_config = config.get('models', {}).get('trading_revenue', {})
        self.risk_factors = self.trading_config.get('risk_factors', [])
        self.var_confidence = self.trading_config.get('var_confidence_level', 0.99)
        self.var_holding_period = self.trading_config.get('var_holding_period', 1)
        
        # Model components
        self.scaler = RobustScaler()
        self.var_models = {}
        self.correlation_matrix = None
        
    def prepare_risk_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare risk factor features for trading revenue modeling.
        
        Args:
            data: Raw market data
            
        Returns:
            Risk factor feature matrix
        """
        features = pd.DataFrame(index=data.index)
        
        # Equity risk factors
        equity_factors = ['sp500_return', 'nasdaq_return', 'russell2000_return', 'vix']
        for factor in equity_factors:
            if factor in data.columns:
                features[factor] = data[factor]
                features[f'{factor}_lag1'] = data[factor].shift(1)
                features[f'{factor}_volatility'] = data[factor].rolling(20).std()
                features[f'{factor}_skewness'] = data[factor].rolling(60).skew()
                features[f'{factor}_kurtosis'] = data[factor].rolling(60).kurt()
        
        # FX risk factors
        fx_factors = ['dxy_return', 'eurusd_return', 'gbpusd_return', 'usdjpy_return']
        for factor in fx_factors:
            if factor in data.columns:
                features[factor] = data[factor]
                features[f'{factor}_volatility'] = data[factor].rolling(20).std()
                features[f'{factor}_momentum'] = data[factor].rolling(10).mean()
        
        # Credit risk factors
        credit_factors = ['credit_spreads', 'high_yield_spreads', 'cds_index']
        for factor in credit_factors:
            if factor in data.columns:
                features[factor] = data[factor]
                features[f'{factor}_change'] = data[factor].diff()
                features[f'{factor}_percentile'] = data[factor].rolling(252).rank(pct=True)
        
        # Commodity risk factors
        commodity_factors = ['oil_return', 'gold_return', 'copper_return']
        for factor in commodity_factors:
            if factor in data.columns:
                features[factor] = data[factor]
                features[f'{factor}_volatility'] = data[factor].rolling(20).std()
        
        # Interest rate risk factors
        if 'treasury_10y' in data.columns and 'treasury_2y' in data.columns:
            features['yield_curve_slope'] = data['treasury_10y'] - data['treasury_2y']
            features['yield_curve_slope_change'] = features['yield_curve_slope'].diff()
        
        # Market regime indicators
        if 'vix' in data.columns:
            features['high_vol_regime'] = (data['vix'] > data['vix'].rolling(252).quantile(0.75)).astype(int)
            features['crisis_regime'] = (data['vix'] > 30).astype(int)
        
        # Cross-asset correlations (rolling)
        if 'sp500_return' in data.columns and 'treasury_10y' in data.columns:
            features['equity_bond_correlation'] = data['sp500_return'].rolling(60).corr(data['treasury_10y'])
        
        return features.fillna(method='ffill').fillna(0)
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = None, 
                     holding_period: int = None) -> Dict[str, float]:
        """
        Calculate Value at Risk metrics.
        
        Args:
            returns: Return series
            confidence_level: VaR confidence level
            holding_period: Holding period in days
            
        Returns:
            Dictionary of VaR metrics
        """
        if confidence_level is None:
            confidence_level = self.var_confidence
        if holding_period is None:
            holding_period = self.var_holding_period
        
        # Historical VaR
        var_historical = np.percentile(returns.dropna(), (1 - confidence_level) * 100)
        
        # Parametric VaR (assuming normal distribution)
        mean_return = returns.mean()
        std_return = returns.std()
        var_parametric = mean_return + stats.norm.ppf(1 - confidence_level) * std_return
        
        # Expected Shortfall (Conditional VaR)
        es = returns[returns <= var_historical].mean()
        
        # Adjust for holding period
        var_historical_adjusted = var_historical * np.sqrt(holding_period)
        var_parametric_adjusted = var_parametric * np.sqrt(holding_period)
        es_adjusted = es * np.sqrt(holding_period)
        
        return {
            'var_historical': var_historical_adjusted,
            'var_parametric': var_parametric_adjusted,
            'expected_shortfall': es_adjusted,
            'volatility': std_return * np.sqrt(252),  # Annualized
            'max_drawdown': (returns.cumsum() - returns.cumsum().expanding().max()).min()
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series, trading_desks: List[str] = None, **kwargs) -> 'TradingRevenueModel':
        """
        Fit the trading revenue model.
        
        Args:
            X: Feature matrix containing market data
            y: Target trading revenue
            trading_desks: List of trading desk names for desk-level modeling
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Fitting Trading Revenue model...")
        
        # Validate data
        self.validate_data(X, y)
        
        # Prepare risk factors
        risk_features = self.prepare_risk_factors(X)
        
        # Scale features
        risk_features_scaled = pd.DataFrame(
            self.scaler.fit_transform(risk_features),
            columns=risk_features.columns,
            index=risk_features.index
        )
        
        # Store feature names
        self.feature_names = risk_features.columns.tolist()
        self.target_name = y.name if y.name else "trading_revenue"
        
        # Fit main trading revenue model
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(risk_features_scaled, y)
        
        # Calculate VaR models for different market factors
        for factor in ['sp500_return', 'dxy_return', 'credit_spreads']:
            if factor in X.columns:
                self.var_models[factor] = self.calculate_var(X[factor])
        
        # Calculate correlation matrix for risk factors
        numeric_features = risk_features.select_dtypes(include=[np.number])
        self.correlation_matrix = numeric_features.corr()
        
        self.is_fitted = True
        self.logger.info("Trading Revenue model fitted successfully")
        
        return self
    
    def predict(self, X: pd.DataFrame, include_var_adjustment: bool = True, **kwargs) -> np.ndarray:
        """
        Generate trading revenue predictions.
        
        Args:
            X: Feature matrix for prediction
            include_var_adjustment: Whether to include VaR-based adjustments
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of trading revenue predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare risk factors
        risk_features = self.prepare_risk_factors(X)
        risk_features_scaled = pd.DataFrame(
            self.scaler.transform(risk_features),
            columns=risk_features.columns,
            index=risk_features.index
        )
        
        # Generate base predictions
        base_predictions = self.model.predict(risk_features_scaled)
        
        if not include_var_adjustment:
            return base_predictions
        
        # Apply VaR-based adjustments
        adjusted_predictions = base_predictions.copy()
        
        # Adjust for high volatility periods
        if 'high_vol_regime' in risk_features.columns:
            high_vol_mask = risk_features['high_vol_regime'] == 1
            adjusted_predictions[high_vol_mask] *= 0.8  # Reduce revenue in high vol periods
        
        # Adjust for crisis periods
        if 'crisis_regime' in risk_features.columns:
            crisis_mask = risk_features['crisis_regime'] == 1
            adjusted_predictions[crisis_mask] *= 0.6  # Significant reduction in crisis
        
        return adjusted_predictions
    
    def stress_test_trading_revenue(self, X: pd.DataFrame, stress_scenarios: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Perform stress testing on trading revenue.
        
        Args:
            X: Base feature matrix
            stress_scenarios: Dictionary of stress scenarios
            
        Returns:
            DataFrame of stressed trading revenue predictions
        """
        results = pd.DataFrame(index=X.index)
        
        # Base case
        base_pred = self.predict(X)
        results['base_case'] = base_pred
        
        # Apply stress scenarios
        for scenario_name, stress_factors in stress_scenarios.items():
            self.logger.info(f"Running stress scenario: {scenario_name}")
            
            # Create stressed features
            stressed_features = self.prepare_risk_factors(X).copy()
            
            # Apply stress factors
            for factor, shock in stress_factors.items():
                if factor in stressed_features.columns:
                    if 'return' in factor:
                        # Apply return shock
                        stressed_features[factor] += shock
                    elif factor == 'vix':
                        # Apply volatility shock (multiplicative)
                        stressed_features[factor] *= (1 + shock)
                    elif 'spread' in factor:
                        # Apply spread widening
                        stressed_features[factor] += shock
            
            # Recalculate derived features
            if 'vix' in stressed_features.columns:
                stressed_features['high_vol_regime'] = (stressed_features['vix'] > 30).astype(int)
                stressed_features['crisis_regime'] = (stressed_features['vix'] > 40).astype(int)
            
            # Scale and predict
            stressed_features_scaled = pd.DataFrame(
                self.scaler.transform(stressed_features),
                columns=stressed_features.columns,
                index=stressed_features.index
            )
            
            stressed_pred = self.model.predict(stressed_features_scaled)
            results[f'{scenario_name}_stressed'] = stressed_pred
            
            # Calculate impact
            results[f'{scenario_name}_impact_pct'] = ((stressed_pred - base_pred) / base_pred) * 100
        
        return results
    
    def calculate_desk_correlations(self, desk_revenues: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlations between trading desk revenues.
        
        Args:
            desk_revenues: DataFrame with desk-level revenue data
            
        Returns:
            Correlation matrix between desks
        """
        return desk_revenues.corr()
    
    def estimate_diversification_benefit(self, desk_revenues: pd.DataFrame) -> Dict[str, float]:
        """
        Estimate diversification benefits across trading desks.
        
        Args:
            desk_revenues: DataFrame with desk-level revenue data
            
        Returns:
            Dictionary of diversification metrics
        """
        # Calculate individual desk volatilities
        individual_vols = desk_revenues.std()
        
        # Calculate portfolio volatility
        correlation_matrix = desk_revenues.corr()
        weights = np.ones(len(desk_revenues.columns)) / len(desk_revenues.columns)  # Equal weights
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(correlation_matrix, weights))) * np.sqrt(252)
        
        # Calculate weighted average of individual volatilities
        weighted_avg_vol = np.dot(weights, individual_vols) * np.sqrt(252)
        
        # Diversification ratio
        diversification_ratio = weighted_avg_vol / portfolio_vol
        
        return {
            'portfolio_volatility': portfolio_vol,
            'weighted_avg_volatility': weighted_avg_vol,
            'diversification_ratio': diversification_ratio,
            'diversification_benefit_pct': (1 - 1/diversification_ratio) * 100
        }
    
    def get_risk_attribution(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Perform risk attribution analysis for trading revenue.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with risk factor contributions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before risk attribution")
        
        # Get feature importance
        feature_importance = self.get_feature_importance()
        
        if feature_importance is None:
            return None
        
        # Prepare features
        risk_features = self.prepare_risk_factors(X)
        
        # Calculate risk contributions
        risk_attribution = pd.DataFrame(index=X.index)
        
        # Group features by risk category
        risk_categories = {
            'equity': [col for col in feature_importance.index if any(eq in col.lower() for eq in ['sp500', 'nasdaq', 'russell', 'vix'])],
            'fx': [col for col in feature_importance.index if any(fx in col.lower() for fx in ['dxy', 'eur', 'gbp', 'jpy'])],
            'credit': [col for col in feature_importance.index if any(cr in col.lower() for cr in ['credit', 'spread', 'cds'])],
            'commodity': [col for col in feature_importance.index if any(com in col.lower() for com in ['oil', 'gold', 'copper'])],
            'rates': [col for col in feature_importance.index if any(rt in col.lower() for rt in ['yield', 'curve', 'treasury'])]
        }
        
        # Calculate category contributions
        for category, features in risk_categories.items():
            category_importance = feature_importance[feature_importance.index.isin(features)].sum()
            risk_attribution[f'{category}_contribution'] = category_importance
        
        return risk_attribution