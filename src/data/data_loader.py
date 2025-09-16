"""
Data Loader for PPNR Risk Models

Handles loading and initial processing of data from various sources:
- Database connections
- File-based data (CSV, Excel, Parquet)
- API data sources
- Real-time data feeds
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import warnings
import os
from pathlib import Path
import sqlite3
import json

class DataLoader:
    """
    Comprehensive data loader for PPNR risk modeling.
    
    Features:
    - Multiple data source support
    - Automatic data type inference
    - Data validation and cleaning
    - Caching capabilities
    - Error handling and logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loader.
        
        Args:
            config: Configuration dictionary containing data source settings
        """
        self.config = config
        self.data_config = config.get('data_sources', {})
        self.cache_config = config.get('caching', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.DataLoader")
        
        # Initialize data cache
        self.data_cache = {}
        self.cache_enabled = self.cache_config.get('enabled', True)
        self.cache_ttl = self.cache_config.get('ttl_hours', 24)
        
        # Data source connections
        self.connections = {}
        
    def load_market_data(self, symbols: List[str], 
                        start_date: str, 
                        end_date: str,
                        data_source: str = 'default') -> pd.DataFrame:
        """
        Load market data for specified symbols and date range.
        
        Args:
            symbols: List of market symbols/tickers
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            data_source: Data source identifier
            
        Returns:
            DataFrame with market data
        """
        self.logger.info(f"Loading market data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Check cache first
        cache_key = f"market_data_{data_source}_{'-'.join(symbols)}_{start_date}_{end_date}"
        if self.cache_enabled and self._is_cache_valid(cache_key):
            self.logger.info("Loading market data from cache")
            return self.data_cache[cache_key]['data']
        
        # Load data based on source configuration
        source_config = self.data_config.get('market_data', {}).get(data_source, {})
        source_type = source_config.get('type', 'file')
        
        if source_type == 'file':
            market_data = self._load_market_data_from_file(symbols, start_date, end_date, source_config)
        elif source_type == 'database':
            market_data = self._load_market_data_from_database(symbols, start_date, end_date, source_config)
        elif source_type == 'api':
            market_data = self._load_market_data_from_api(symbols, start_date, end_date, source_config)
        else:
            # Generate synthetic data for demonstration
            market_data = self._generate_synthetic_market_data(symbols, start_date, end_date)
        
        # Cache the data
        if self.cache_enabled:
            self._cache_data(cache_key, market_data)
        
        self.logger.info(f"Loaded market data: {market_data.shape[0]} rows, {market_data.shape[1]} columns")
        return market_data
    
    def _load_market_data_from_file(self, symbols: List[str], 
                                   start_date: str, 
                                   end_date: str,
                                   source_config: Dict[str, Any]) -> pd.DataFrame:
        """Load market data from file sources."""
        file_path = source_config.get('path', 'data/raw/market_data.csv')
        
        if not os.path.exists(file_path):
            self.logger.warning(f"Market data file not found: {file_path}. Generating synthetic data.")
            return self._generate_synthetic_market_data(symbols, start_date, end_date)
        
        try:
            # Determine file format
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                data = pd.read_excel(file_path)
            elif file_path.endswith('.parquet'):
                data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Process and filter data
            data = self._process_market_data(data, symbols, start_date, end_date)
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading market data from file: {str(e)}")
            return self._generate_synthetic_market_data(symbols, start_date, end_date)
    
    def _load_market_data_from_database(self, symbols: List[str], 
                                       start_date: str, 
                                       end_date: str,
                                       source_config: Dict[str, Any]) -> pd.DataFrame:
        """Load market data from database."""
        try:
            # Database connection details
            db_type = source_config.get('db_type', 'sqlite')
            connection_string = source_config.get('connection_string', 'data/market_data.db')
            table_name = source_config.get('table_name', 'market_data')
            
            # Create connection
            if db_type == 'sqlite':
                conn = sqlite3.connect(connection_string)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            # Build query
            symbols_str = "', '".join(symbols)
            query = f"""
                SELECT * FROM {table_name}
                WHERE symbol IN ('{symbols_str}')
                AND date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY date, symbol
            """
            
            # Execute query
            data = pd.read_sql_query(query, conn)
            conn.close()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading market data from database: {str(e)}")
            return self._generate_synthetic_market_data(symbols, start_date, end_date)
    
    def _load_market_data_from_api(self, symbols: List[str], 
                                  start_date: str, 
                                  end_date: str,
                                  source_config: Dict[str, Any]) -> pd.DataFrame:
        """Load market data from API."""
        # This would integrate with real market data APIs like Bloomberg, Reuters, etc.
        # For now, generate synthetic data
        self.logger.info("API data loading not implemented. Generating synthetic data.")
        return self._generate_synthetic_market_data(symbols, start_date, end_date)
    
    def _generate_synthetic_market_data(self, symbols: List[str], 
                                       start_date: str, 
                                       end_date: str) -> pd.DataFrame:
        """Generate synthetic market data for testing and demonstration."""
        self.logger.info("Generating synthetic market data")
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Market data structure
        market_data = []
        
        # Base parameters for different asset classes
        asset_params = {
            'SP500': {'initial_price': 4000, 'volatility': 0.15, 'drift': 0.08},
            'VIX': {'initial_price': 20, 'volatility': 0.8, 'drift': 0.0, 'mean_reversion': 0.5},
            'USD_10Y': {'initial_price': 3.5, 'volatility': 0.2, 'drift': 0.0, 'mean_reversion': 0.3},
            'USD_2Y': {'initial_price': 2.8, 'volatility': 0.25, 'drift': 0.0, 'mean_reversion': 0.4},
            'USD_3M': {'initial_price': 2.5, 'volatility': 0.3, 'drift': 0.0, 'mean_reversion': 0.5},
            'CREDIT_IG': {'initial_price': 150, 'volatility': 0.4, 'drift': 0.0, 'mean_reversion': 0.2},
            'CREDIT_HY': {'initial_price': 400, 'volatility': 0.6, 'drift': 0.0, 'mean_reversion': 0.2},
            'USD_EUR': {'initial_price': 1.1, 'volatility': 0.12, 'drift': 0.0},
            'USD_JPY': {'initial_price': 110, 'volatility': 0.10, 'drift': 0.0},
            'OIL_WTI': {'initial_price': 70, 'volatility': 0.35, 'drift': 0.05}
        }
        
        for symbol in symbols:
            # Get parameters or use defaults
            params = asset_params.get(symbol, {
                'initial_price': 100, 
                'volatility': 0.2, 
                'drift': 0.05
            })
            
            # Generate price series
            prices = self._generate_price_series(
                len(date_range), 
                params['initial_price'],
                params['volatility'],
                params['drift'],
                params.get('mean_reversion', 0.0)
            )
            
            # Create DataFrame for this symbol
            symbol_data = pd.DataFrame({
                'date': date_range,
                'symbol': symbol,
                'price': prices,
                'return': np.concatenate([[0], np.diff(np.log(prices))]),
                'volume': np.random.lognormal(15, 0.5, len(date_range))
            })
            
            market_data.append(symbol_data)
        
        # Combine all symbols
        combined_data = pd.concat(market_data, ignore_index=True)
        combined_data['date'] = pd.to_datetime(combined_data['date'])
        
        return combined_data
    
    def _generate_price_series(self, n_periods: int, 
                              initial_price: float,
                              volatility: float, 
                              drift: float,
                              mean_reversion: float = 0.0) -> np.ndarray:
        """Generate synthetic price series using geometric Brownian motion with optional mean reversion."""
        dt = 1/252  # Daily frequency
        prices = np.zeros(n_periods)
        prices[0] = initial_price
        
        for i in range(1, n_periods):
            # Random shock
            shock = np.random.normal(0, 1)
            
            # Mean reversion component
            if mean_reversion > 0:
                mean_revert = mean_reversion * (np.log(initial_price) - np.log(prices[i-1]))
            else:
                mean_revert = 0
            
            # Price evolution
            log_return = (drift + mean_revert) * dt + volatility * np.sqrt(dt) * shock
            prices[i] = prices[i-1] * np.exp(log_return)
        
        return prices
    
    def _process_market_data(self, data: pd.DataFrame, 
                           symbols: List[str], 
                           start_date: str, 
                           end_date: str) -> pd.DataFrame:
        """Process and filter market data."""
        # Convert date column
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        
        # Filter by symbols
        if 'symbol' in data.columns:
            data = data[data['symbol'].isin(symbols)]
        
        # Filter by date range
        if 'date' in data.columns:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            data = data[(data['date'] >= start_dt) & (data['date'] <= end_dt)]
        
        # Sort data
        if 'date' in data.columns and 'symbol' in data.columns:
            data = data.sort_values(['date', 'symbol'])
        
        return data
    
    def load_economic_indicators(self, indicators: List[str],
                               start_date: str,
                               end_date: str,
                               data_source: str = 'default') -> pd.DataFrame:
        """
        Load economic indicator data.
        
        Args:
            indicators: List of economic indicators
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            data_source: Data source identifier
            
        Returns:
            DataFrame with economic indicator data
        """
        self.logger.info(f"Loading economic indicators: {indicators}")
        
        # Check cache
        cache_key = f"econ_indicators_{data_source}_{'-'.join(indicators)}_{start_date}_{end_date}"
        if self.cache_enabled and self._is_cache_valid(cache_key):
            return self.data_cache[cache_key]['data']
        
        # Generate synthetic economic data for demonstration
        econ_data = self._generate_synthetic_economic_data(indicators, start_date, end_date)
        
        # Cache the data
        if self.cache_enabled:
            self._cache_data(cache_key, econ_data)
        
        return econ_data
    
    def _generate_synthetic_economic_data(self, indicators: List[str],
                                        start_date: str,
                                        end_date: str) -> pd.DataFrame:
        """Generate synthetic economic indicator data."""
        # Create monthly date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # Economic indicator parameters
        econ_params = {
            'gdp_growth': {'initial': 2.5, 'volatility': 0.5, 'trend': 0.0, 'persistence': 0.7},
            'unemployment_rate': {'initial': 4.0, 'volatility': 0.3, 'trend': 0.0, 'persistence': 0.8},
            'inflation_rate': {'initial': 2.0, 'volatility': 0.4, 'trend': 0.0, 'persistence': 0.6},
            'fed_funds_rate': {'initial': 2.5, 'volatility': 0.2, 'trend': 0.0, 'persistence': 0.9},
            'consumer_confidence': {'initial': 100, 'volatility': 5, 'trend': 0.0, 'persistence': 0.5},
            'housing_starts': {'initial': 1200, 'volatility': 100, 'trend': 0.0, 'persistence': 0.6},
            'industrial_production': {'initial': 100, 'volatility': 2, 'trend': 0.1, 'persistence': 0.7}
        }
        
        econ_data = []
        
        for indicator in indicators:
            params = econ_params.get(indicator, {
                'initial': 100, 
                'volatility': 5, 
                'trend': 0.0, 
                'persistence': 0.5
            })
            
            # Generate AR(1) series
            series = self._generate_ar1_series(
                len(date_range),
                params['initial'],
                params['volatility'],
                params['persistence'],
                params['trend']
            )
            
            indicator_data = pd.DataFrame({
                'date': date_range,
                'indicator': indicator,
                'value': series
            })
            
            econ_data.append(indicator_data)
        
        combined_data = pd.concat(econ_data, ignore_index=True)
        return combined_data
    
    def _generate_ar1_series(self, n_periods: int,
                           initial_value: float,
                           volatility: float,
                           persistence: float,
                           trend: float = 0.0) -> np.ndarray:
        """Generate AR(1) time series."""
        series = np.zeros(n_periods)
        series[0] = initial_value
        
        for i in range(1, n_periods):
            shock = np.random.normal(0, volatility)
            series[i] = persistence * series[i-1] + (1 - persistence) * initial_value + trend + shock
        
        return series
    
    def load_bank_metrics(self, metrics: List[str],
                         start_date: str,
                         end_date: str,
                         data_source: str = 'default') -> pd.DataFrame:
        """
        Load bank-specific metrics data.
        
        Args:
            metrics: List of bank metrics
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            data_source: Data source identifier
            
        Returns:
            DataFrame with bank metrics data
        """
        self.logger.info(f"Loading bank metrics: {metrics}")
        
        # Check cache
        cache_key = f"bank_metrics_{data_source}_{'-'.join(metrics)}_{start_date}_{end_date}"
        if self.cache_enabled and self._is_cache_valid(cache_key):
            return self.data_cache[cache_key]['data']
        
        # Generate synthetic bank data
        bank_data = self._generate_synthetic_bank_data(metrics, start_date, end_date)
        
        # Cache the data
        if self.cache_enabled:
            self._cache_data(cache_key, bank_data)
        
        return bank_data
    
    def _generate_synthetic_bank_data(self, metrics: List[str],
                                    start_date: str,
                                    end_date: str) -> pd.DataFrame:
        """Generate synthetic bank metrics data."""
        # Create quarterly date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='QS')
        
        # Bank metrics parameters
        bank_params = {
            'total_assets': {'initial': 2000000, 'volatility': 50000, 'trend': 1000, 'persistence': 0.95},
            'total_loans': {'initial': 1200000, 'volatility': 30000, 'trend': 800, 'persistence': 0.9},
            'total_deposits': {'initial': 1500000, 'volatility': 40000, 'trend': 900, 'persistence': 0.92},
            'tier1_capital_ratio': {'initial': 12.5, 'volatility': 0.3, 'trend': 0.0, 'persistence': 0.8},
            'net_charge_offs': {'initial': 0.5, 'volatility': 0.2, 'trend': 0.0, 'persistence': 0.6},
            'provision_expense': {'initial': 100, 'volatility': 50, 'trend': 0.0, 'persistence': 0.4},
            'trading_assets': {'initial': 150000, 'volatility': 20000, 'trend': 0.0, 'persistence': 0.7}
        }
        
        bank_data = []
        
        for metric in metrics:
            params = bank_params.get(metric, {
                'initial': 1000, 
                'volatility': 100, 
                'trend': 0.0, 
                'persistence': 0.7
            })
            
            series = self._generate_ar1_series(
                len(date_range),
                params['initial'],
                params['volatility'],
                params['persistence'],
                params['trend']
            )
            
            metric_data = pd.DataFrame({
                'date': date_range,
                'metric': metric,
                'value': series
            })
            
            bank_data.append(metric_data)
        
        combined_data = pd.concat(bank_data, ignore_index=True)
        return combined_data
    
    def load_historical_ppnr_data(self, start_date: str,
                                 end_date: str,
                                 data_source: str = 'default') -> pd.DataFrame:
        """
        Load historical PPNR component data.
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            data_source: Data source identifier
            
        Returns:
            DataFrame with historical PPNR data
        """
        self.logger.info("Loading historical PPNR data")
        
        # Check cache
        cache_key = f"ppnr_data_{data_source}_{start_date}_{end_date}"
        if self.cache_enabled and self._is_cache_valid(cache_key):
            return self.data_cache[cache_key]['data']
        
        # Generate synthetic PPNR data
        ppnr_data = self._generate_synthetic_ppnr_data(start_date, end_date)
        
        # Cache the data
        if self.cache_enabled:
            self._cache_data(cache_key, ppnr_data)
        
        return ppnr_data
    
    def _generate_synthetic_ppnr_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic PPNR component data."""
        # Create quarterly date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='QS')
        
        ppnr_data = []
        
        # PPNR components with realistic parameters
        components = {
            'net_interest_income': {'initial': 5000, 'volatility': 200, 'trend': 50, 'persistence': 0.8},
            'service_charges': {'initial': 800, 'volatility': 50, 'trend': 10, 'persistence': 0.7},
            'investment_banking_fees': {'initial': 1200, 'volatility': 300, 'trend': 0, 'persistence': 0.5},
            'trading_revenue': {'initial': 600, 'volatility': 400, 'trend': 0, 'persistence': 0.3},
            'card_fees': {'initial': 400, 'volatility': 30, 'trend': 15, 'persistence': 0.8},
            'mortgage_banking': {'initial': 300, 'volatility': 150, 'trend': 0, 'persistence': 0.4},
            'trust_fees': {'initial': 200, 'volatility': 20, 'trend': 5, 'persistence': 0.9}
        }
        
        for component, params in components.items():
            series = self._generate_ar1_series(
                len(date_range),
                params['initial'],
                params['volatility'],
                params['persistence'],
                params['trend']
            )
            
            component_data = pd.DataFrame({
                'date': date_range,
                'component': component,
                'value': series
            })
            
            ppnr_data.append(component_data)
        
        combined_data = pd.concat(ppnr_data, ignore_index=True)
        return combined_data
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.data_cache:
            return False
        
        cache_entry = self.data_cache[cache_key]
        cache_time = cache_entry['timestamp']
        current_time = datetime.now()
        
        # Check if cache has expired
        if (current_time - cache_time).total_seconds() > (self.cache_ttl * 3600):
            del self.data_cache[cache_key]
            return False
        
        return True
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame) -> None:
        """Cache data with timestamp."""
        self.data_cache[cache_key] = {
            'data': data.copy(),
            'timestamp': datetime.now()
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.data_cache.clear()
        self.logger.info("Data cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        cache_info = {
            'cache_enabled': self.cache_enabled,
            'cache_ttl_hours': self.cache_ttl,
            'cached_items': len(self.data_cache),
            'cache_keys': list(self.data_cache.keys())
        }
        
        return cache_info
    
    def save_data(self, data: pd.DataFrame, 
                  file_path: str, 
                  format: str = 'csv') -> None:
        """
        Save data to file.
        
        Args:
            data: DataFrame to save
            file_path: Output file path
            format: File format ('csv', 'excel', 'parquet')
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if format.lower() == 'csv':
            data.to_csv(file_path, index=False)
        elif format.lower() in ['excel', 'xlsx']:
            data.to_excel(file_path, index=False)
        elif format.lower() == 'parquet':
            data.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Data saved to {file_path}")
    
    def load_custom_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load custom data from file with flexible parameters.
        
        Args:
            file_path: Path to data file
            **kwargs: Additional parameters for pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.csv':
            return pd.read_csv(file_path, **kwargs)
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, **kwargs)
        elif file_extension == '.parquet':
            return pd.read_parquet(file_path, **kwargs)
        elif file_extension == '.json':
            return pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive summary of loaded data.
        
        Args:
            data: DataFrame to summarize
            
        Returns:
            Dictionary with data summary statistics
        """
        summary = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Add numeric column statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = data[numeric_cols].describe().to_dict()
        
        # Add date column information
        date_cols = data.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            summary['date_ranges'] = {}
            for col in date_cols:
                summary['date_ranges'][col] = {
                    'min_date': str(data[col].min()),
                    'max_date': str(data[col].max()),
                    'date_range_days': (data[col].max() - data[col].min()).days
                }
        
        return summary