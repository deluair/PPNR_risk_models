"""
Economic Indicators Processor for PPNR Risk Models

Specialized processing for economic indicators including:
- GDP and employment data
- Inflation and monetary policy indicators
- Housing and consumer metrics
- Leading economic indicators
- Stress scenario construction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class EconomicIndicatorsProcessor:
    """
    Comprehensive economic indicators processor for PPNR risk modeling.
    
    Features:
    - Economic data transformation and normalization
    - Leading indicator construction
    - Economic regime identification
    - Stress scenario generation
    - Forecasting support
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize economic indicators processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.economic_config = config.get('economic_indicators', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.EconomicIndicatorsProcessor")
        
        # Processing parameters
        self.transformation_method = self.economic_config.get('transformation_method', 'yoy_change')
        self.smoothing_window = self.economic_config.get('smoothing_window', 3)
        self.leading_indicators_window = self.economic_config.get('leading_indicators_window', 12)
        
        # Processed data storage
        self.processed_indicators = {}
        self.leading_indicators = {}
        self.economic_regimes = {}
        
        # Define indicator categories
        self.indicator_categories = {
            'growth': ['GDP', 'INDUSTRIAL_PRODUCTION', 'RETAIL_SALES', 'PERSONAL_INCOME'],
            'employment': ['UNEMPLOYMENT_RATE', 'NONFARM_PAYROLLS', 'JOBLESS_CLAIMS'],
            'inflation': ['CPI', 'PPI', 'PCE', 'CORE_CPI'],
            'monetary': ['FED_FUNDS_RATE', 'M2_MONEY_SUPPLY', 'TREASURY_YIELD_10Y'],
            'housing': ['HOUSING_STARTS', 'HOME_SALES', 'CASE_SHILLER_INDEX'],
            'consumer': ['CONSUMER_CONFIDENCE', 'CONSUMER_SENTIMENT', 'PERSONAL_SPENDING'],
            'business': ['BUSINESS_CONFIDENCE', 'CAPEX', 'INVENTORY_CHANGE'],
            'financial': ['CREDIT_SPREADS', 'STOCK_MARKET_INDEX', 'DOLLAR_INDEX']
        }
    
    def process_economic_indicators(self, economic_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw economic indicators into analysis-ready format.
        
        Args:
            economic_data: Raw economic indicators DataFrame
            
        Returns:
            Processed economic indicators
        """
        self.logger.info("Processing economic indicators...")
        
        # Ensure data is properly formatted
        processed_data = self._format_economic_data(economic_data)
        
        # Apply transformations
        processed_data = self._apply_transformations(processed_data)
        
        # Calculate derived indicators
        processed_data = self._calculate_derived_indicators(processed_data)
        
        # Add smoothed versions
        processed_data = self._add_smoothed_indicators(processed_data)
        
        # Store processed data
        self.processed_indicators = processed_data
        
        self.logger.info(f"Economic indicators processing completed: {processed_data.shape}")
        return processed_data
    
    def _format_economic_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Format and validate economic data structure."""
        formatted_data = data.copy()
        
        # Ensure date column is datetime
        if 'date' in formatted_data.columns:
            formatted_data['date'] = pd.to_datetime(formatted_data['date'])
            formatted_data = formatted_data.sort_values('date')
            formatted_data = formatted_data.set_index('date')
        
        # Handle missing values
        formatted_data = self._handle_missing_economic_data(formatted_data)
        
        # Ensure monthly frequency for economic data
        formatted_data = self._ensure_monthly_frequency(formatted_data)
        
        return formatted_data
    
    def _handle_missing_economic_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in economic data."""
        # Forward fill for most economic indicators (released with lag)
        data = data.fillna(method='ffill', limit=3)
        
        # Interpolate for smooth series like GDP
        smooth_indicators = ['GDP', 'PERSONAL_INCOME', 'INDUSTRIAL_PRODUCTION']
        for indicator in smooth_indicators:
            if indicator in data.columns:
                data[indicator] = data[indicator].interpolate(method='linear')
        
        # Log missing data
        missing_summary = data.isnull().sum()
        if missing_summary.sum() > 0:
            self.logger.warning(f"Missing data summary:\n{missing_summary[missing_summary > 0]}")
        
        return data
    
    def _ensure_monthly_frequency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure data has consistent monthly frequency."""
        # Create monthly date range
        start_date = data.index.min()
        end_date = data.index.max()
        monthly_index = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Reindex to monthly frequency
        data = data.reindex(monthly_index)
        
        return data
    
    def _apply_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply standard transformations to economic indicators."""
        result_data = data.copy()
        
        # Year-over-year changes
        for col in data.columns:
            if col in data.columns:
                result_data[f'{col}_yoy'] = data[col].pct_change(periods=12) * 100
        
        # Month-over-month changes
        for col in data.columns:
            if col in data.columns:
                result_data[f'{col}_mom'] = data[col].pct_change(periods=1) * 100
        
        # Quarter-over-quarter changes (annualized)
        for col in data.columns:
            if col in data.columns:
                result_data[f'{col}_qoq_ann'] = (data[col].pct_change(periods=3) * 4) * 100
        
        # Level changes for rates and indices
        rate_indicators = ['UNEMPLOYMENT_RATE', 'FED_FUNDS_RATE', 'TREASURY_YIELD_10Y']
        for indicator in rate_indicators:
            if indicator in data.columns:
                result_data[f'{indicator}_change'] = data[indicator].diff()
                result_data[f'{indicator}_change_12m'] = data[indicator].diff(periods=12)
        
        # Log transformations for level series
        level_indicators = ['GDP', 'INDUSTRIAL_PRODUCTION', 'M2_MONEY_SUPPLY']
        for indicator in level_indicators:
            if indicator in data.columns and (data[indicator] > 0).all():
                result_data[f'{indicator}_log'] = np.log(data[indicator])
        
        return result_data
    
    def _calculate_derived_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived economic indicators."""
        result_data = data.copy()
        
        # Real vs nominal adjustments
        if 'CPI' in data.columns:
            # Real personal income
            if 'PERSONAL_INCOME' in data.columns:
                result_data['REAL_PERSONAL_INCOME'] = (
                    data['PERSONAL_INCOME'] / data['CPI'] * 100
                )
            
            # Real retail sales
            if 'RETAIL_SALES' in data.columns:
                result_data['REAL_RETAIL_SALES'] = (
                    data['RETAIL_SALES'] / data['CPI'] * 100
                )
        
        # Yield curve indicators
        if 'TREASURY_YIELD_10Y' in data.columns and 'FED_FUNDS_RATE' in data.columns:
            result_data['YIELD_CURVE_SLOPE'] = (
                data['TREASURY_YIELD_10Y'] - data['FED_FUNDS_RATE']
            )
        
        # Employment indicators
        if 'NONFARM_PAYROLLS' in data.columns:
            result_data['PAYROLLS_3M_AVG'] = data['NONFARM_PAYROLLS'].rolling(3).mean()
            result_data['PAYROLLS_MOMENTUM'] = (
                data['NONFARM_PAYROLLS'].rolling(3).mean() - 
                data['NONFARM_PAYROLLS'].rolling(3).mean().shift(3)
            )
        
        # Consumer strength composite
        consumer_indicators = ['CONSUMER_CONFIDENCE', 'PERSONAL_SPENDING', 'RETAIL_SALES']
        available_consumer = [ind for ind in consumer_indicators if ind in data.columns]
        if len(available_consumer) >= 2:
            # Standardize and average
            consumer_data = data[available_consumer]
            standardized = (consumer_data - consumer_data.mean()) / consumer_data.std()
            result_data['CONSUMER_STRENGTH_INDEX'] = standardized.mean(axis=1)
        
        # Housing market composite
        housing_indicators = ['HOUSING_STARTS', 'HOME_SALES', 'CASE_SHILLER_INDEX']
        available_housing = [ind for ind in housing_indicators if ind in data.columns]
        if len(available_housing) >= 2:
            housing_data = data[available_housing]
            standardized = (housing_data - housing_data.mean()) / housing_data.std()
            result_data['HOUSING_MARKET_INDEX'] = standardized.mean(axis=1)
        
        # Inflation expectations proxy
        if 'TREASURY_YIELD_10Y' in data.columns and 'FED_FUNDS_RATE' in data.columns:
            result_data['INFLATION_EXPECTATIONS_PROXY'] = (
                data['TREASURY_YIELD_10Y'] - data['FED_FUNDS_RATE'] - 2.0  # Assume 2% real rate
            )
        
        return result_data
    
    def _add_smoothed_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add smoothed versions of key indicators."""
        result_data = data.copy()
        
        # Moving averages for volatile series
        volatile_indicators = ['JOBLESS_CLAIMS', 'CONSUMER_CONFIDENCE', 'BUSINESS_CONFIDENCE']
        for indicator in volatile_indicators:
            if indicator in data.columns:
                result_data[f'{indicator}_3M_MA'] = data[indicator].rolling(3).mean()
                result_data[f'{indicator}_6M_MA'] = data[indicator].rolling(6).mean()
        
        # Exponentially weighted moving averages
        for indicator in data.columns:
            if not indicator.endswith(('_MA', '_yoy', '_mom', '_qoq_ann', '_change')):
                result_data[f'{indicator}_EWMA'] = data[indicator].ewm(span=6).mean()
        
        return result_data
    
    def construct_leading_indicators(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Construct leading economic indicators.
        
        Args:
            processed_data: Processed economic data
            
        Returns:
            DataFrame with leading indicators
        """
        self.logger.info("Constructing leading economic indicators...")
        
        leading_indicators = pd.DataFrame(index=processed_data.index)
        
        # 1. Employment leading indicators
        leading_indicators = self._construct_employment_leading_indicators(
            leading_indicators, processed_data
        )
        
        # 2. Growth leading indicators
        leading_indicators = self._construct_growth_leading_indicators(
            leading_indicators, processed_data
        )
        
        # 3. Inflation leading indicators
        leading_indicators = self._construct_inflation_leading_indicators(
            leading_indicators, processed_data
        )
        
        # 4. Financial conditions leading indicators
        leading_indicators = self._construct_financial_leading_indicators(
            leading_indicators, processed_data
        )
        
        # 5. Composite leading indicators
        leading_indicators = self._construct_composite_leading_indicators(
            leading_indicators, processed_data
        )
        
        # Store leading indicators
        self.leading_indicators = leading_indicators
        
        self.logger.info(f"Leading indicators constructed: {leading_indicators.shape}")
        return leading_indicators
    
    def _construct_employment_leading_indicators(self, leading_indicators: pd.DataFrame,
                                               data: pd.DataFrame) -> pd.DataFrame:
        """Construct employment leading indicators."""
        # Jobless claims trend
        if 'JOBLESS_CLAIMS' in data.columns:
            leading_indicators['JOBLESS_CLAIMS_TREND'] = (
                data['JOBLESS_CLAIMS'].rolling(4).mean() / 
                data['JOBLESS_CLAIMS'].rolling(12).mean()
            )
        
        # Employment momentum
        if 'NONFARM_PAYROLLS_yoy' in data.columns:
            leading_indicators['EMPLOYMENT_MOMENTUM'] = (
                data['NONFARM_PAYROLLS_yoy'].rolling(3).mean()
            )
        
        return leading_indicators
    
    def _construct_growth_leading_indicators(self, leading_indicators: pd.DataFrame,
                                           data: pd.DataFrame) -> pd.DataFrame:
        """Construct growth leading indicators."""
        # Industrial production momentum
        if 'INDUSTRIAL_PRODUCTION_yoy' in data.columns:
            leading_indicators['INDUSTRIAL_PRODUCTION_MOMENTUM'] = (
                data['INDUSTRIAL_PRODUCTION_yoy'] - 
                data['INDUSTRIAL_PRODUCTION_yoy'].rolling(6).mean()
            )
        
        # Consumer spending trend
        if 'PERSONAL_SPENDING_yoy' in data.columns:
            leading_indicators['CONSUMER_SPENDING_TREND'] = (
                data['PERSONAL_SPENDING_yoy'].rolling(3).mean()
            )
        
        return leading_indicators
    
    def _construct_inflation_leading_indicators(self, leading_indicators: pd.DataFrame,
                                              data: pd.DataFrame) -> pd.DataFrame:
        """Construct inflation leading indicators."""
        # Core inflation trend
        if 'CORE_CPI_yoy' in data.columns:
            leading_indicators['CORE_INFLATION_TREND'] = (
                data['CORE_CPI_yoy'].rolling(3).mean()
            )
        
        # Wage pressure indicator
        if 'PERSONAL_INCOME_yoy' in data.columns and 'CPI_yoy' in data.columns:
            leading_indicators['WAGE_PRESSURE'] = (
                data['PERSONAL_INCOME_yoy'] - data['CPI_yoy']
            )
        
        return leading_indicators
    
    def _construct_financial_leading_indicators(self, leading_indicators: pd.DataFrame,
                                              data: pd.DataFrame) -> pd.DataFrame:
        """Construct financial conditions leading indicators."""
        # Yield curve steepness change
        if 'YIELD_CURVE_SLOPE' in data.columns:
            leading_indicators['YIELD_CURVE_STEEPNESS_CHANGE'] = (
                data['YIELD_CURVE_SLOPE'].diff(periods=3)
            )
        
        # Credit conditions proxy
        if 'CREDIT_SPREADS' in data.columns:
            leading_indicators['CREDIT_CONDITIONS'] = (
                -data['CREDIT_SPREADS'].rolling(3).mean()  # Negative because lower spreads = better conditions
            )
        
        return leading_indicators
    
    def _construct_composite_leading_indicators(self, leading_indicators: pd.DataFrame,
                                              data: pd.DataFrame) -> pd.DataFrame:
        """Construct composite leading indicators."""
        # Economic momentum composite
        momentum_indicators = [col for col in leading_indicators.columns if 'MOMENTUM' in col]
        if len(momentum_indicators) >= 2:
            momentum_data = leading_indicators[momentum_indicators].dropna()
            if not momentum_data.empty:
                standardized = (momentum_data - momentum_data.mean()) / momentum_data.std()
                leading_indicators['ECONOMIC_MOMENTUM_COMPOSITE'] = standardized.mean(axis=1)
        
        # Financial conditions composite
        financial_indicators = [col for col in leading_indicators.columns 
                              if any(term in col for term in ['YIELD_CURVE', 'CREDIT_CONDITIONS'])]
        if len(financial_indicators) >= 2:
            financial_data = leading_indicators[financial_indicators].dropna()
            if not financial_data.empty:
                standardized = (financial_data - financial_data.mean()) / financial_data.std()
                leading_indicators['FINANCIAL_CONDITIONS_COMPOSITE'] = standardized.mean(axis=1)
        
        return leading_indicators
    
    def identify_economic_regimes(self, processed_data: pd.DataFrame,
                                method: str = 'growth_inflation') -> pd.DataFrame:
        """
        Identify economic regimes.
        
        Args:
            processed_data: Processed economic data
            method: Regime identification method
            
        Returns:
            DataFrame with regime indicators
        """
        self.logger.info(f"Identifying economic regimes using {method} method...")
        
        regimes = processed_data.copy()
        
        if method == 'growth_inflation':
            regimes = self._identify_growth_inflation_regimes(regimes)
        elif method == 'business_cycle':
            regimes = self._identify_business_cycle_regimes(regimes)
        elif method == 'financial_conditions':
            regimes = self._identify_financial_conditions_regimes(regimes)
        else:
            raise ValueError(f"Unknown regime identification method: {method}")
        
        # Store regimes
        self.economic_regimes = regimes
        
        return regimes
    
    def _identify_growth_inflation_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify regimes based on growth and inflation."""
        result = data.copy()
        
        # Use GDP growth and CPI inflation
        growth_indicator = 'GDP_yoy' if 'GDP_yoy' in data.columns else 'INDUSTRIAL_PRODUCTION_yoy'
        inflation_indicator = 'CPI_yoy' if 'CPI_yoy' in data.columns else 'CORE_CPI_yoy'
        
        if growth_indicator not in data.columns or inflation_indicator not in data.columns:
            self.logger.warning("Required indicators not found for growth-inflation regime identification")
            return result
        
        growth = data[growth_indicator]
        inflation = data[inflation_indicator]
        
        # Define thresholds
        growth_median = growth.median()
        inflation_median = inflation.median()
        
        # Assign regimes
        result['economic_regime'] = 'unknown'
        
        # High growth, low inflation - Goldilocks
        mask = (growth >= growth_median) & (inflation <= inflation_median)
        result.loc[mask, 'economic_regime'] = 'goldilocks'
        
        # High growth, high inflation - Overheating
        mask = (growth >= growth_median) & (inflation > inflation_median)
        result.loc[mask, 'economic_regime'] = 'overheating'
        
        # Low growth, low inflation - Slowdown
        mask = (growth < growth_median) & (inflation <= inflation_median)
        result.loc[mask, 'economic_regime'] = 'slowdown'
        
        # Low growth, high inflation - Stagflation
        mask = (growth < growth_median) & (inflation > inflation_median)
        result.loc[mask, 'economic_regime'] = 'stagflation'
        
        # Create regime indicators
        for regime in ['goldilocks', 'overheating', 'slowdown', 'stagflation']:
            result[f'regime_{regime}'] = (result['economic_regime'] == regime).astype(int)
        
        return result
    
    def _identify_business_cycle_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify business cycle regimes."""
        result = data.copy()
        
        # Use multiple indicators for business cycle identification
        cycle_indicators = []
        
        for indicator in ['GDP_yoy', 'INDUSTRIAL_PRODUCTION_yoy', 'UNEMPLOYMENT_RATE']:
            if indicator in data.columns:
                cycle_indicators.append(indicator)
        
        if len(cycle_indicators) < 2:
            self.logger.warning("Insufficient indicators for business cycle regime identification")
            return result
        
        # Create composite business cycle indicator
        cycle_data = data[cycle_indicators].dropna()
        
        # Standardize indicators (invert unemployment rate)
        standardized_data = cycle_data.copy()
        for col in cycle_data.columns:
            if 'UNEMPLOYMENT' in col:
                standardized_data[col] = -cycle_data[col]  # Invert unemployment
            standardized_data[col] = (standardized_data[col] - standardized_data[col].mean()) / standardized_data[col].std()
        
        # Create composite indicator
        composite_indicator = standardized_data.mean(axis=1)
        
        # Define regime thresholds
        expansion_threshold = composite_indicator.quantile(0.6)
        recession_threshold = composite_indicator.quantile(0.2)
        
        # Assign regimes
        regime_series = pd.Series(index=composite_indicator.index, dtype='object')
        regime_series[:] = 'normal'
        regime_series[composite_indicator >= expansion_threshold] = 'expansion'
        regime_series[composite_indicator <= recession_threshold] = 'recession'
        
        # Align with result DataFrame
        result['business_cycle_regime'] = regime_series.reindex(result.index)
        
        # Create regime indicators
        for regime in ['expansion', 'normal', 'recession']:
            result[f'cycle_{regime}'] = (result['business_cycle_regime'] == regime).astype(int)
        
        # Store composite indicator
        result['business_cycle_composite'] = composite_indicator.reindex(result.index)
        
        return result
    
    def _identify_financial_conditions_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify financial conditions regimes."""
        result = data.copy()
        
        # Use financial indicators
        financial_indicators = []
        
        for indicator in ['YIELD_CURVE_SLOPE', 'CREDIT_SPREADS', 'STOCK_MARKET_INDEX_yoy']:
            if indicator in data.columns:
                financial_indicators.append(indicator)
        
        if len(financial_indicators) < 2:
            self.logger.warning("Insufficient indicators for financial conditions regime identification")
            return result
        
        # Create financial conditions index
        financial_data = data[financial_indicators].dropna()
        
        # Standardize indicators (invert credit spreads)
        standardized_data = financial_data.copy()
        for col in financial_data.columns:
            if 'CREDIT_SPREADS' in col:
                standardized_data[col] = -financial_data[col]  # Invert credit spreads
            standardized_data[col] = (standardized_data[col] - standardized_data[col].mean()) / standardized_data[col].std()
        
        # Create financial conditions index
        financial_conditions_index = standardized_data.mean(axis=1)
        
        # Define regime thresholds
        loose_threshold = financial_conditions_index.quantile(0.67)
        tight_threshold = financial_conditions_index.quantile(0.33)
        
        # Assign regimes
        regime_series = pd.Series(index=financial_conditions_index.index, dtype='object')
        regime_series[:] = 'neutral'
        regime_series[financial_conditions_index >= loose_threshold] = 'loose'
        regime_series[financial_conditions_index <= tight_threshold] = 'tight'
        
        # Align with result DataFrame
        result['financial_conditions_regime'] = regime_series.reindex(result.index)
        
        # Create regime indicators
        for regime in ['loose', 'neutral', 'tight']:
            result[f'financial_{regime}'] = (result['financial_conditions_regime'] == regime).astype(int)
        
        # Store financial conditions index
        result['financial_conditions_index'] = financial_conditions_index.reindex(result.index)
        
        return result
    
    def generate_stress_scenarios(self, processed_data: pd.DataFrame,
                                scenario_type: str = 'adverse') -> pd.DataFrame:
        """
        Generate stress scenarios for economic indicators.
        
        Args:
            processed_data: Processed economic data
            scenario_type: Type of stress scenario ('adverse', 'severely_adverse')
            
        Returns:
            DataFrame with stress scenario values
        """
        self.logger.info(f"Generating {scenario_type} stress scenarios...")
        
        # Get latest values as baseline
        baseline = processed_data.iloc[-1:].copy()
        
        # Define stress scenario parameters
        if scenario_type == 'adverse':
            stress_params = self._get_adverse_scenario_params()
        elif scenario_type == 'severely_adverse':
            stress_params = self._get_severely_adverse_scenario_params()
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        # Apply stress to indicators
        stressed_scenario = self._apply_stress_to_indicators(baseline, stress_params)
        
        return stressed_scenario
    
    def _get_adverse_scenario_params(self) -> Dict[str, float]:
        """Get parameters for adverse scenario."""
        return {
            'GDP_yoy': -2.0,  # 2% GDP decline
            'UNEMPLOYMENT_RATE': 2.0,  # 2pp increase in unemployment
            'TREASURY_YIELD_10Y': -1.0,  # 100bp decrease in 10Y yield
            'STOCK_MARKET_INDEX': -20.0,  # 20% stock market decline
            'HOUSING_STARTS': -15.0,  # 15% decline in housing starts
            'CONSUMER_CONFIDENCE': -20.0,  # 20 point decline in confidence
            'CREDIT_SPREADS': 1.5,  # 150bp increase in credit spreads
        }
    
    def _get_severely_adverse_scenario_params(self) -> Dict[str, float]:
        """Get parameters for severely adverse scenario."""
        return {
            'GDP_yoy': -4.0,  # 4% GDP decline
            'UNEMPLOYMENT_RATE': 4.0,  # 4pp increase in unemployment
            'TREASURY_YIELD_10Y': -2.0,  # 200bp decrease in 10Y yield
            'STOCK_MARKET_INDEX': -35.0,  # 35% stock market decline
            'HOUSING_STARTS': -30.0,  # 30% decline in housing starts
            'CONSUMER_CONFIDENCE': -35.0,  # 35 point decline in confidence
            'CREDIT_SPREADS': 3.0,  # 300bp increase in credit spreads
        }
    
    def _apply_stress_to_indicators(self, baseline: pd.DataFrame,
                                  stress_params: Dict[str, float]) -> pd.DataFrame:
        """Apply stress parameters to baseline indicators."""
        stressed = baseline.copy()
        
        for indicator, stress_value in stress_params.items():
            if indicator in stressed.columns:
                if indicator in ['UNEMPLOYMENT_RATE', 'CREDIT_SPREADS', 'TREASURY_YIELD_10Y']:
                    # Additive stress for rates and spreads
                    stressed[indicator] = stressed[indicator] + stress_value
                else:
                    # Multiplicative stress for growth rates and levels
                    if '_yoy' in indicator:
                        stressed[indicator] = stress_value  # Replace with stress value
                    else:
                        stressed[indicator] = stressed[indicator] * (1 + stress_value / 100)
        
        return stressed
    
    def forecast_indicators(self, processed_data: pd.DataFrame,
                          horizon: int = 12,
                          method: str = 'trend') -> pd.DataFrame:
        """
        Generate forecasts for economic indicators.
        
        Args:
            processed_data: Processed economic data
            horizon: Forecast horizon in months
            method: Forecasting method ('trend', 'seasonal', 'ar')
            
        Returns:
            DataFrame with forecasted values
        """
        self.logger.info(f"Forecasting indicators for {horizon} months using {method} method...")
        
        # Get last date and create forecast dates
        last_date = processed_data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq='M'
        )
        
        forecast_data = pd.DataFrame(index=forecast_dates)
        
        # Apply forecasting method to each indicator
        for col in processed_data.select_dtypes(include=[np.number]).columns:
            if method == 'trend':
                forecast_data[col] = self._trend_forecast(processed_data[col], horizon)
            elif method == 'seasonal':
                forecast_data[col] = self._seasonal_forecast(processed_data[col], horizon)
            elif method == 'ar':
                forecast_data[col] = self._ar_forecast(processed_data[col], horizon)
        
        return forecast_data
    
    def _trend_forecast(self, series: pd.Series, horizon: int) -> pd.Series:
        """Simple trend-based forecast."""
        # Use last 12 months for trend calculation
        recent_data = series.dropna().tail(12)
        
        if len(recent_data) < 2:
            # If insufficient data, use last value
            return pd.Series([series.iloc[-1]] * horizon)
        
        # Calculate linear trend
        x = np.arange(len(recent_data))
        y = recent_data.values
        
        # Fit linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Generate forecast
        forecast_x = np.arange(len(recent_data), len(recent_data) + horizon)
        forecast_values = slope * forecast_x + intercept
        
        return pd.Series(forecast_values)
    
    def _seasonal_forecast(self, series: pd.Series, horizon: int) -> pd.Series:
        """Seasonal forecast using historical patterns."""
        # Calculate seasonal pattern (12-month cycle)
        seasonal_data = series.dropna()
        
        if len(seasonal_data) < 24:  # Need at least 2 years
            return self._trend_forecast(series, horizon)
        
        # Calculate monthly seasonal factors
        monthly_data = seasonal_data.groupby(seasonal_data.index.month).mean()
        
        # Apply trend and seasonal pattern
        trend_forecast = self._trend_forecast(series, horizon)
        
        # Get seasonal factors for forecast months
        forecast_months = [(i % 12) + 1 for i in range(horizon)]
        seasonal_factors = [monthly_data.get(month, 1.0) for month in forecast_months]
        
        # Combine trend and seasonal
        base_level = series.dropna().tail(12).mean()
        seasonal_forecast = trend_forecast + (np.array(seasonal_factors) - base_level)
        
        return pd.Series(seasonal_forecast)
    
    def _ar_forecast(self, series: pd.Series, horizon: int) -> pd.Series:
        """Simple AR(1) forecast."""
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            return self._trend_forecast(series, horizon)
        
        # Estimate AR(1) parameters
        y = clean_series.values[1:]
        x = clean_series.values[:-1]
        
        # Simple AR(1): y_t = alpha + beta * y_{t-1} + error
        beta = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        alpha = np.mean(y) - beta * np.mean(x)
        
        # Generate forecast
        forecast_values = []
        last_value = clean_series.iloc[-1]
        
        for _ in range(horizon):
            next_value = alpha + beta * last_value
            forecast_values.append(next_value)
            last_value = next_value
        
        return pd.Series(forecast_values)
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processed economic indicators."""
        summary = {
            'processed_indicators_shape': self.processed_indicators.shape if hasattr(self, 'processed_indicators') else None,
            'leading_indicators_shape': self.leading_indicators.shape if hasattr(self, 'leading_indicators') else None,
            'economic_regimes_shape': self.economic_regimes.shape if hasattr(self, 'economic_regimes') else None,
            'processing_parameters': {
                'transformation_method': self.transformation_method,
                'smoothing_window': self.smoothing_window,
                'leading_indicators_window': self.leading_indicators_window
            },
            'indicator_categories': self.indicator_categories
        }
        
        if hasattr(self, 'processed_indicators') and not self.processed_indicators.empty:
            summary['processed_columns'] = list(self.processed_indicators.columns)
            summary['date_range'] = {
                'start': str(self.processed_indicators.index.min()),
                'end': str(self.processed_indicators.index.max())
            }
        
        if hasattr(self, 'leading_indicators') and not self.leading_indicators.empty:
            summary['leading_indicator_columns'] = list(self.leading_indicators.columns)
        
        return summary