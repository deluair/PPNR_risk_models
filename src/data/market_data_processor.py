"""
Market Data Processor for PPNR Risk Models

Specialized processing for market data including:
- Price and return calculations
- Volatility estimation
- Risk factor construction
- Market regime identification
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
import matplotlib.pyplot as plt

class MarketDataProcessor:
    """
    Comprehensive market data processor for PPNR risk modeling.
    
    Features:
    - Return and volatility calculations
    - Risk factor construction
    - Market regime identification
    - Correlation analysis
    - Data transformation and normalization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize market data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.market_config = config.get('market_data', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.MarketDataProcessor")
        
        # Processing parameters
        self.return_frequency = self.market_config.get('return_frequency', 'daily')
        self.volatility_window = self.market_config.get('volatility_window', 252)
        self.correlation_window = self.market_config.get('correlation_window', 252)
        
        # Processed data storage
        self.processed_data = {}
        self.risk_factors = {}
        
    def process_market_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw market data into analysis-ready format.
        
        Args:
            market_data: Raw market data DataFrame
            
        Returns:
            Processed market data with returns, volatilities, etc.
        """
        self.logger.info("Processing market data...")
        
        # Ensure data is properly formatted
        processed_data = self._format_market_data(market_data)
        
        # Calculate returns
        processed_data = self._calculate_returns(processed_data)
        
        # Calculate volatilities
        processed_data = self._calculate_volatilities(processed_data)
        
        # Add technical indicators
        processed_data = self._add_technical_indicators(processed_data)
        
        # Store processed data
        self.processed_data = processed_data
        
        self.logger.info(f"Market data processing completed: {processed_data.shape}")
        return processed_data
    
    def _format_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Format and validate market data structure."""
        formatted_data = data.copy()
        
        # Ensure date column is datetime
        if 'date' in formatted_data.columns:
            formatted_data['date'] = pd.to_datetime(formatted_data['date'])
            formatted_data = formatted_data.sort_values(['date', 'symbol'])
        
        # Validate required columns
        required_cols = ['date', 'symbol', 'price']
        missing_cols = [col for col in required_cols if col not in formatted_data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Handle missing values
        formatted_data = self._handle_missing_values(formatted_data)
        
        return formatted_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in market data."""
        # Forward fill prices within each symbol
        if 'symbol' in data.columns:
            data['price'] = data.groupby('symbol')['price'].fillna(method='ffill')
        
        # Drop remaining missing values
        initial_rows = len(data)
        data = data.dropna(subset=['price'])
        dropped_rows = initial_rows - len(data)
        
        if dropped_rows > 0:
            self.logger.warning(f"Dropped {dropped_rows} rows due to missing price data")
        
        return data
    
    def _calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate various types of returns."""
        result_data = data.copy()
        
        # Calculate log returns for each symbol
        if 'symbol' in result_data.columns:
            result_data['log_return'] = result_data.groupby('symbol')['price'].transform(
                lambda x: np.log(x / x.shift(1))
            )
            result_data['simple_return'] = result_data.groupby('symbol')['price'].transform(
                lambda x: x.pct_change()
            )
        else:
            result_data['log_return'] = np.log(result_data['price'] / result_data['price'].shift(1))
            result_data['simple_return'] = result_data['price'].pct_change()
        
        # Calculate multi-period returns
        for period in [5, 21, 63]:  # Weekly, monthly, quarterly
            if 'symbol' in result_data.columns:
                result_data[f'return_{period}d'] = result_data.groupby('symbol')['price'].transform(
                    lambda x: (x / x.shift(period)) - 1
                )
            else:
                result_data[f'return_{period}d'] = (result_data['price'] / result_data['price'].shift(period)) - 1
        
        return result_data
    
    def _calculate_volatilities(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling volatilities."""
        result_data = data.copy()
        
        # Rolling volatility windows
        vol_windows = [21, 63, 252]  # Monthly, quarterly, annual
        
        for window in vol_windows:
            if 'symbol' in result_data.columns:
                result_data[f'volatility_{window}d'] = result_data.groupby('symbol')['log_return'].transform(
                    lambda x: x.rolling(window=window, min_periods=window//2).std() * np.sqrt(252)
                )
            else:
                result_data[f'volatility_{window}d'] = result_data['log_return'].rolling(
                    window=window, min_periods=window//2
                ).std() * np.sqrt(252)
        
        # GARCH-like volatility (exponentially weighted)
        alpha = 0.94  # Decay factor
        if 'symbol' in result_data.columns:
            result_data['ewm_volatility'] = result_data.groupby('symbol')['log_return'].transform(
                lambda x: x.ewm(alpha=alpha).std() * np.sqrt(252)
            )
        else:
            result_data['ewm_volatility'] = result_data['log_return'].ewm(alpha=alpha).std() * np.sqrt(252)
        
        return result_data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators relevant for risk modeling."""
        result_data = data.copy()
        
        # Moving averages
        ma_windows = [20, 50, 200]
        for window in ma_windows:
            if 'symbol' in result_data.columns:
                result_data[f'ma_{window}'] = result_data.groupby('symbol')['price'].transform(
                    lambda x: x.rolling(window=window).mean()
                )
            else:
                result_data[f'ma_{window}'] = result_data['price'].rolling(window=window).mean()
        
        # Price momentum indicators
        momentum_windows = [10, 20, 60]
        for window in momentum_windows:
            if 'symbol' in result_data.columns:
                result_data[f'momentum_{window}d'] = result_data.groupby('symbol')['price'].transform(
                    lambda x: (x / x.shift(window)) - 1
                )
            else:
                result_data[f'momentum_{window}d'] = (result_data['price'] / result_data['price'].shift(window)) - 1
        
        # RSI-like indicator
        if 'symbol' in result_data.columns:
            result_data['rsi_14'] = result_data.groupby('symbol').apply(
                lambda group: self._calculate_rsi(group['price'], 14)
            ).reset_index(level=0, drop=True)
        else:
            result_data['rsi_14'] = self._calculate_rsi(result_data['price'], 14)
        
        return result_data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def construct_risk_factors(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Construct risk factors from market data.
        
        Args:
            processed_data: Processed market data
            
        Returns:
            DataFrame with constructed risk factors
        """
        self.logger.info("Constructing risk factors...")
        
        # Pivot data to have symbols as columns
        if 'symbol' in processed_data.columns:
            returns_pivot = processed_data.pivot(index='date', columns='symbol', values='log_return')
            prices_pivot = processed_data.pivot(index='date', columns='symbol', values='price')
        else:
            returns_pivot = processed_data.set_index('date')[['log_return']]
            prices_pivot = processed_data.set_index('date')[['price']]
        
        risk_factors = pd.DataFrame(index=returns_pivot.index)
        
        # 1. Equity risk factors
        risk_factors = self._construct_equity_factors(risk_factors, returns_pivot, prices_pivot)
        
        # 2. Interest rate risk factors
        risk_factors = self._construct_interest_rate_factors(risk_factors, returns_pivot, prices_pivot)
        
        # 3. Credit risk factors
        risk_factors = self._construct_credit_factors(risk_factors, returns_pivot, prices_pivot)
        
        # 4. FX risk factors
        risk_factors = self._construct_fx_factors(risk_factors, returns_pivot, prices_pivot)
        
        # 5. Commodity risk factors
        risk_factors = self._construct_commodity_factors(risk_factors, returns_pivot, prices_pivot)
        
        # 6. Volatility and correlation factors
        risk_factors = self._construct_volatility_factors(risk_factors, returns_pivot)
        
        # Store risk factors
        self.risk_factors = risk_factors
        
        self.logger.info(f"Risk factors constructed: {risk_factors.shape}")
        return risk_factors
    
    def _construct_equity_factors(self, risk_factors: pd.DataFrame,
                                returns_pivot: pd.DataFrame,
                                prices_pivot: pd.DataFrame) -> pd.DataFrame:
        """Construct equity risk factors."""
        # Market return (SP500 if available)
        if 'SP500' in returns_pivot.columns:
            risk_factors['equity_market_return'] = returns_pivot['SP500']
            risk_factors['equity_market_volatility'] = returns_pivot['SP500'].rolling(21).std() * np.sqrt(252)
        
        # VIX (volatility index)
        if 'VIX' in prices_pivot.columns:
            risk_factors['vix_level'] = prices_pivot['VIX']
            risk_factors['vix_change'] = prices_pivot['VIX'].pct_change()
        
        # Market momentum
        if 'SP500' in prices_pivot.columns:
            risk_factors['equity_momentum_1m'] = (prices_pivot['SP500'] / prices_pivot['SP500'].shift(21)) - 1
            risk_factors['equity_momentum_3m'] = (prices_pivot['SP500'] / prices_pivot['SP500'].shift(63)) - 1
        
        return risk_factors
    
    def _construct_interest_rate_factors(self, risk_factors: pd.DataFrame,
                                       returns_pivot: pd.DataFrame,
                                       prices_pivot: pd.DataFrame) -> pd.DataFrame:
        """Construct interest rate risk factors."""
        # Yield levels
        for tenor in ['USD_3M', 'USD_2Y', 'USD_10Y']:
            if tenor in prices_pivot.columns:
                risk_factors[f'{tenor.lower()}_yield'] = prices_pivot[tenor]
                risk_factors[f'{tenor.lower()}_change'] = prices_pivot[tenor].diff()
        
        # Yield curve factors
        if 'USD_10Y' in prices_pivot.columns and 'USD_2Y' in prices_pivot.columns:
            risk_factors['yield_curve_slope'] = prices_pivot['USD_10Y'] - prices_pivot['USD_2Y']
            risk_factors['yield_curve_slope_change'] = risk_factors['yield_curve_slope'].diff()
        
        if 'USD_10Y' in prices_pivot.columns and 'USD_3M' in prices_pivot.columns:
            risk_factors['term_spread'] = prices_pivot['USD_10Y'] - prices_pivot['USD_3M']
        
        # Interest rate volatility
        for tenor in ['USD_2Y', 'USD_10Y']:
            if tenor in returns_pivot.columns:
                risk_factors[f'{tenor.lower()}_volatility'] = returns_pivot[tenor].rolling(21).std() * np.sqrt(252)
        
        return risk_factors
    
    def _construct_credit_factors(self, risk_factors: pd.DataFrame,
                                returns_pivot: pd.DataFrame,
                                prices_pivot: pd.DataFrame) -> pd.DataFrame:
        """Construct credit risk factors."""
        # Credit spread levels
        for grade in ['CREDIT_IG', 'CREDIT_HY']:
            if grade in prices_pivot.columns:
                risk_factors[f'{grade.lower()}_spread'] = prices_pivot[grade]
                risk_factors[f'{grade.lower()}_change'] = prices_pivot[grade].diff()
        
        # Credit spread volatility
        for grade in ['CREDIT_IG', 'CREDIT_HY']:
            if grade in returns_pivot.columns:
                risk_factors[f'{grade.lower()}_volatility'] = returns_pivot[grade].rolling(21).std() * np.sqrt(252)
        
        # Credit spread differential
        if 'CREDIT_HY' in prices_pivot.columns and 'CREDIT_IG' in prices_pivot.columns:
            risk_factors['credit_spread_differential'] = prices_pivot['CREDIT_HY'] - prices_pivot['CREDIT_IG']
        
        return risk_factors
    
    def _construct_fx_factors(self, risk_factors: pd.DataFrame,
                            returns_pivot: pd.DataFrame,
                            prices_pivot: pd.DataFrame) -> pd.DataFrame:
        """Construct FX risk factors."""
        # FX rates
        for pair in ['USD_EUR', 'USD_JPY']:
            if pair in prices_pivot.columns:
                risk_factors[f'{pair.lower()}_rate'] = prices_pivot[pair]
                risk_factors[f'{pair.lower()}_return'] = returns_pivot[pair]
                risk_factors[f'{pair.lower()}_volatility'] = returns_pivot[pair].rolling(21).std() * np.sqrt(252)
        
        # Dollar strength index (if multiple FX pairs available)
        fx_pairs = [col for col in prices_pivot.columns if 'USD_' in col and col not in ['USD_2Y', 'USD_10Y', 'USD_3M']]
        if len(fx_pairs) >= 2:
            # Simple dollar index (equal weighted)
            fx_returns = returns_pivot[fx_pairs].mean(axis=1)
            risk_factors['dollar_index_return'] = fx_returns
            risk_factors['dollar_index_volatility'] = fx_returns.rolling(21).std() * np.sqrt(252)
        
        return risk_factors
    
    def _construct_commodity_factors(self, risk_factors: pd.DataFrame,
                                   returns_pivot: pd.DataFrame,
                                   prices_pivot: pd.DataFrame) -> pd.DataFrame:
        """Construct commodity risk factors."""
        # Oil prices
        if 'OIL_WTI' in prices_pivot.columns:
            risk_factors['oil_price'] = prices_pivot['OIL_WTI']
            risk_factors['oil_return'] = returns_pivot['OIL_WTI']
            risk_factors['oil_volatility'] = returns_pivot['OIL_WTI'].rolling(21).std() * np.sqrt(252)
        
        return risk_factors
    
    def _construct_volatility_factors(self, risk_factors: pd.DataFrame,
                                    returns_pivot: pd.DataFrame) -> pd.DataFrame:
        """Construct volatility and correlation factors."""
        # Market-wide volatility measures
        if len(returns_pivot.columns) > 1:
            # Average correlation
            corr_window = 63  # Quarterly correlation
            rolling_corr = returns_pivot.rolling(window=corr_window).corr()
            
            # Extract average correlation (excluding diagonal)
            avg_correlations = []
            for date in returns_pivot.index[corr_window-1:]:
                corr_matrix = rolling_corr.loc[date]
                if isinstance(corr_matrix, pd.DataFrame):
                    # Get upper triangle excluding diagonal
                    mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
                    correlations = corr_matrix.values[mask]
                    avg_corr = np.nanmean(correlations)
                    avg_correlations.append(avg_corr)
                else:
                    avg_correlations.append(np.nan)
            
            # Align with risk_factors index
            corr_series = pd.Series(avg_correlations, index=returns_pivot.index[corr_window-1:])
            risk_factors['average_correlation'] = corr_series.reindex(risk_factors.index)
        
        # Realized volatility of market portfolio
        if len(returns_pivot.columns) > 1:
            # Equal-weighted portfolio return
            portfolio_return = returns_pivot.mean(axis=1)
            risk_factors['portfolio_volatility'] = portfolio_return.rolling(21).std() * np.sqrt(252)
        
        return risk_factors
    
    def identify_market_regimes(self, risk_factors: pd.DataFrame,
                              method: str = 'volatility') -> pd.DataFrame:
        """
        Identify market regimes based on risk factors.
        
        Args:
            risk_factors: Risk factors DataFrame
            method: Regime identification method ('volatility', 'correlation', 'pca')
            
        Returns:
            DataFrame with regime indicators
        """
        self.logger.info(f"Identifying market regimes using {method} method...")
        
        regimes = risk_factors.copy()
        
        if method == 'volatility':
            regimes = self._identify_volatility_regimes(regimes)
        elif method == 'correlation':
            regimes = self._identify_correlation_regimes(regimes)
        elif method == 'pca':
            regimes = self._identify_pca_regimes(regimes)
        else:
            raise ValueError(f"Unknown regime identification method: {method}")
        
        return regimes
    
    def _identify_volatility_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify regimes based on volatility levels."""
        result = data.copy()
        
        # Use VIX or portfolio volatility for regime identification
        if 'vix_level' in result.columns:
            vol_indicator = result['vix_level']
        elif 'portfolio_volatility' in result.columns:
            vol_indicator = result['portfolio_volatility']
        else:
            self.logger.warning("No volatility indicator found for regime identification")
            return result
        
        # Define regime thresholds (percentiles)
        low_threshold = vol_indicator.quantile(0.33)
        high_threshold = vol_indicator.quantile(0.67)
        
        # Assign regimes
        result['volatility_regime'] = 'medium'
        result.loc[vol_indicator <= low_threshold, 'volatility_regime'] = 'low'
        result.loc[vol_indicator >= high_threshold, 'volatility_regime'] = 'high'
        
        # Create regime indicators
        for regime in ['low', 'medium', 'high']:
            result[f'vol_regime_{regime}'] = (result['volatility_regime'] == regime).astype(int)
        
        return result
    
    def _identify_correlation_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify regimes based on correlation levels."""
        result = data.copy()
        
        if 'average_correlation' not in result.columns:
            self.logger.warning("Average correlation not found for regime identification")
            return result
        
        corr_indicator = result['average_correlation']
        
        # Define regime thresholds
        low_threshold = corr_indicator.quantile(0.33)
        high_threshold = corr_indicator.quantile(0.67)
        
        # Assign regimes
        result['correlation_regime'] = 'medium'
        result.loc[corr_indicator <= low_threshold, 'correlation_regime'] = 'low'
        result.loc[corr_indicator >= high_threshold, 'correlation_regime'] = 'high'
        
        # Create regime indicators
        for regime in ['low', 'medium', 'high']:
            result[f'corr_regime_{regime}'] = (result['correlation_regime'] == regime).astype(int)
        
        return result
    
    def _identify_pca_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify regimes using PCA on risk factors."""
        result = data.copy()
        
        # Select numeric columns for PCA
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        pca_data = result[numeric_cols].dropna()
        
        if len(pca_data) < 50:  # Need sufficient data for PCA
            self.logger.warning("Insufficient data for PCA regime identification")
            return result
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_data)
        
        # Apply PCA
        pca = PCA(n_components=min(3, len(numeric_cols)))
        pca_components = pca.fit_transform(scaled_data)
        
        # Use first principal component for regime identification
        pc1 = pca_components[:, 0]
        
        # Define regime thresholds
        low_threshold = np.percentile(pc1, 33)
        high_threshold = np.percentile(pc1, 67)
        
        # Create regime series aligned with original data
        regime_series = pd.Series(index=pca_data.index, dtype='object')
        regime_series[:] = 'medium'
        regime_series[pc1 <= low_threshold] = 'low'
        regime_series[pc1 >= high_threshold] = 'high'
        
        # Align with result DataFrame
        result['pca_regime'] = regime_series.reindex(result.index)
        
        # Create regime indicators
        for regime in ['low', 'medium', 'high']:
            result[f'pca_regime_{regime}'] = (result['pca_regime'] == regime).astype(int)
        
        # Store PCA information
        result['pca_component_1'] = pd.Series(pc1, index=pca_data.index).reindex(result.index)
        
        return result
    
    def calculate_correlation_matrix(self, data: pd.DataFrame,
                                   window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling correlation matrix.
        
        Args:
            data: Input data
            window: Rolling window size
            
        Returns:
            Correlation matrix
        """
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Calculate rolling correlation
        rolling_corr = numeric_data.rolling(window=window).corr()
        
        return rolling_corr
    
    def detect_outliers(self, data: pd.DataFrame,
                       method: str = 'iqr',
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in market data.
        
        Args:
            data: Input data
            method: Outlier detection method ('iqr', 'zscore', 'isolation')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier indicators
        """
        result = data.copy()
        
        # Select numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                result[f'{col}_outlier'] = (
                    (result[col] < lower_bound) | (result[col] > upper_bound)
                ).astype(int)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(result[col].dropna()))
                outlier_mask = z_scores > threshold
                
                # Align with original data
                outlier_series = pd.Series(0, index=result.index)
                outlier_series.loc[result[col].dropna().index[outlier_mask]] = 1
                result[f'{col}_outlier'] = outlier_series
        
        return result
    
    def transform_data(self, data: pd.DataFrame,
                      transformations: Dict[str, str]) -> pd.DataFrame:
        """
        Apply transformations to data.
        
        Args:
            data: Input data
            transformations: Dictionary mapping column names to transformation types
            
        Returns:
            Transformed data
        """
        result = data.copy()
        
        for col, transform_type in transformations.items():
            if col not in result.columns:
                continue
            
            if transform_type == 'log':
                result[f'{col}_log'] = np.log(result[col].clip(lower=1e-10))
            elif transform_type == 'sqrt':
                result[f'{col}_sqrt'] = np.sqrt(result[col].clip(lower=0))
            elif transform_type == 'standardize':
                result[f'{col}_std'] = (result[col] - result[col].mean()) / result[col].std()
            elif transform_type == 'normalize':
                result[f'{col}_norm'] = (result[col] - result[col].min()) / (result[col].max() - result[col].min())
            elif transform_type == 'winsorize':
                lower = result[col].quantile(0.01)
                upper = result[col].quantile(0.99)
                result[f'{col}_winsorized'] = result[col].clip(lower=lower, upper=upper)
        
        return result
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processed data."""
        summary = {
            'processed_data_shape': self.processed_data.shape if hasattr(self, 'processed_data') else None,
            'risk_factors_shape': self.risk_factors.shape if hasattr(self, 'risk_factors') else None,
            'processing_parameters': {
                'return_frequency': self.return_frequency,
                'volatility_window': self.volatility_window,
                'correlation_window': self.correlation_window
            }
        }
        
        if hasattr(self, 'processed_data') and not self.processed_data.empty:
            summary['processed_columns'] = list(self.processed_data.columns)
            summary['date_range'] = {
                'start': str(self.processed_data['date'].min()) if 'date' in self.processed_data.columns else None,
                'end': str(self.processed_data['date'].max()) if 'date' in self.processed_data.columns else None
            }
        
        if hasattr(self, 'risk_factors') and not self.risk_factors.empty:
            summary['risk_factor_columns'] = list(self.risk_factors.columns)
        
        return summary